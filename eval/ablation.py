"""Ablations.

1. newmm on/off — BM25 recall@10 on the same queries, tokenized with vs
   without newmm word segmentation. If these two numbers are close, BM25
   is silently broken: an un-segmented Thai paragraph collapses into
   roughly one giant token, so BM25 degenerates into near-random ranking.
   This is meant to run in CI on every change to text_utils.py or
   indexing.py, not just once and forgotten.

2. bm25-only / vector-only / hybrid (RRF) — proves the fusion in
   nodes/fusion.py earns its complexity instead of just adding latency.
   Needs a live Postgres+pgvector (PgVectorStore), unlike #1.

    python -m eval.ablation --mode sample --local-vector             # both, no Postgres needed
    python -m eval.ablation --mode sample --skip-retriever-ablation  # #1 only, no Postgres either way
    python -m eval.ablation --mode sample                            # both, #2 needs a live Postgres+pgvector
"""
from __future__ import annotations

import argparse

from config import AppConfig
from data.loaders import load_nomiracl_subset
from indexing import build_bm25_index, iter_corpus
from eval.metrics import recall_at_k
from text_utils import thai_tokenize

MIN_RECALL_GAP = 0.05  # newmm recall must beat no-newmm recall by at least this much


def _gold_queries_in_corpus(config: AppConfig, doc_id_set: set[str]) -> list[dict]:
    """dev.relevant queries whose gold passage actually landed inside the
    (possibly sampled) corpus we indexed — recall@k is undefined otherwise."""
    root = config.data.nomiracl_local_dir
    records = load_nomiracl_subset(root, config.data.nomiracl_lang, "dev", "relevant")
    queries = []
    for r in records:
        relevant_ids = {p["docid"] for p in r["passages"] if p.get("relevance", 0) > 0}
        overlap = relevant_ids & doc_id_set
        if overlap:
            queries.append({"query": r["query"], "relevant_ids": overlap})
    return queries


def _bm25_recall(tokenized_corpus: list[list[str]], doc_ids: list[str], queries: list[dict], use_newmm: bool, k: int = 10) -> float:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized_corpus)
    recalls = []
    for q in queries:
        qtokens = thai_tokenize(q["query"], use_newmm=use_newmm)
        scored = list(zip(doc_ids, bm25.get_scores(qtokens)))
        scored.sort(key=lambda ds: ds[1], reverse=True)
        ranked = [d for d, _ in scored[:k]]
        recalls.append(recall_at_k(ranked, q["relevant_ids"], k))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def newmm_ablation(config: AppConfig) -> dict:
    passages = list(iter_corpus(config))
    doc_ids = [p["doc_id"] for p in passages]
    doc_id_set = set(doc_ids)

    queries = _gold_queries_in_corpus(config, doc_id_set)
    if not queries:
        raise RuntimeError(
            "no nomiracl dev.relevant queries have their gold passage inside the current corpus "
            f"({len(passages)} passages, data_mode={config.data.data_mode}) — run with --mode full, "
            "or index a corpus slice that actually includes nomiracl's gold docs"
        )

    tokenized_no_newmm = [thai_tokenize(p["text"], use_newmm=False) for p in passages]
    tokenized_newmm = [thai_tokenize(p["text"], use_newmm=True) for p in passages]

    recall_no_newmm = _bm25_recall(tokenized_no_newmm, doc_ids, queries, use_newmm=False)
    recall_newmm = _bm25_recall(tokenized_newmm, doc_ids, queries, use_newmm=True)
    gap = recall_newmm - recall_no_newmm

    print(f"[ablation] newmm on/off — n_queries={len(queries)} (gold doc present in corpus)")
    print(f"[ablation] BM25 recall@10  no-newmm={recall_no_newmm:.3f}  newmm={recall_newmm:.3f}  gap={gap:+.3f}")
    if gap < MIN_RECALL_GAP:
        print(
            f"[ablation] FAIL: gap {gap:.3f} < {MIN_RECALL_GAP} — Thai word segmentation isn't "
            "meaningfully improving BM25 recall. Likely bug: check that indexing.py and "
            "nodes/retrieval.py both call text_utils.thai_tokenize with the same use_newmm setting."
        )
    else:
        print("[ablation] PASS: newmm segmentation meaningfully improves BM25 recall")

    return {"recall_no_newmm": recall_no_newmm, "recall_newmm": recall_newmm, "gap": gap, "n_queries": len(queries)}


def retriever_ablation(config: AppConfig, local_vector: bool = False) -> dict:
    """bm25-only vs vector-only vs hybrid(RRF) recall@10. Needs Postgres
    unless local_vector=True (see local_vector_store.py)."""
    from nodes.fusion import make_fusion_node

    passages = list(iter_corpus(config))
    bm25, doc_ids, corpus = build_bm25_index(passages, config)
    doc_id_set = set(doc_ids)

    if local_vector:
        from local_vector_store import LocalVectorStore

        cache_path = config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl"
        if cache_path.exists():
            store = LocalVectorStore.load(config, cache_path)
        else:
            store = LocalVectorStore(config)
            store.build(passages)
            store.save(cache_path)
    else:
        from indexing import PgVectorStore

        store = PgVectorStore(config)
        store.build(passages)
    fusion = make_fusion_node(config)

    queries = _gold_queries_in_corpus(config, doc_id_set)
    if not queries:
        raise RuntimeError("no nomiracl dev.relevant queries have their gold passage inside the current corpus")

    results = {"bm25_only": [], "vector_only": [], "hybrid": []}
    for q in queries:
        query_text = q["query"]
        relevant_ids = q["relevant_ids"]

        qtokens = thai_tokenize(query_text, use_newmm=config.retrieval.use_newmm)
        scored = sorted(zip(doc_ids, bm25.get_scores(qtokens)), key=lambda ds: ds[1], reverse=True)[:10]
        bm25_hits = [{"doc_id": d, "score": float(s)} for d, s in scored]

        vector_hits = store.similarity_search(query_text, k=10)

        fused = fusion({"bm25_hits": bm25_hits, "vector_hits": vector_hits})["fused_docs"]

        results["bm25_only"].append(recall_at_k([h["doc_id"] for h in bm25_hits], relevant_ids, 10))
        results["vector_only"].append(recall_at_k([h["doc_id"] for h in vector_hits], relevant_ids, 10))
        results["hybrid"].append(recall_at_k([d["doc_id"] for d in fused], relevant_ids, 10))

    summary = {name: (sum(v) / len(v) if v else float("nan")) for name, v in results.items()}
    print(
        f"[ablation] retriever recall@10  bm25_only={summary['bm25_only']:.3f}  "
        f"vector_only={summary['vector_only']:.3f}  hybrid={summary['hybrid']:.3f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "full"], default=None)
    parser.add_argument("--skip-retriever-ablation", action="store_true", help="skip the bm25/vector/hybrid half")
    parser.add_argument("--local-vector", action="store_true",
                         help="retriever ablation: use an in-memory NumPy vector store instead of Postgres")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.mode:
        config.data.data_mode = args.mode

    newmm_ablation(config)
    if not args.skip_retriever_ablation:
        retriever_ablation(config, local_vector=args.local_vector)


if __name__ == "__main__":
    main()
