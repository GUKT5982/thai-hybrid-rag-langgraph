"""Mode B — full pipeline eval: real retrieval (BM25 + pgvector) through the
compiled LangGraph, scored against NoMIRACL qrels. Needs indexing.py to
have been run first (a live Postgres+pgvector), unlike Mode A.

    python -m eval.mode_b --mode sample --split dev --limit 30 --local-vector   # no Postgres needed
    python -m eval.mode_b --mode sample --split dev --limit 30                  # needs a live Postgres+pgvector

Caveat, read before trusting the aggregate numbers: NoMIRACL's
`non_relevant` queries are ones where MIRACL's ORIGINAL retriever found
nothing relevant for — not proof the Thai Wikipedia corpus has no answer.
Mode A sidesteps this (it's handed a fixed passage set, period). Mode B
can't: a better retriever than that original baseline may legitimately
surface a real answer for a query labeled non_relevant, and that would
look like a "hallucination" in the aggregate stats when it isn't. That's
why every run also dumps individual cases to a markdown file — read them,
especially the non_relevant rows that didn't abstain.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from config import AppConfig
from data.loaders import load_nomiracl_subset
from graph import build_graph
from indexing import PgVectorStore, build_bm25_index, iter_corpus
from local_vector_store import LocalVectorStore
from eval.metrics import error_rate, hallucination_rate, ndcg_at_k, recall_at_k
from nodes.generate import is_abstain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "full"], default=None)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=30, help="cap per subset")
    parser.add_argument("--dump-examples", type=int, default=25)
    parser.add_argument("--out", default="eval/results/mode_b.json")
    parser.add_argument("--local-vector", action="store_true",
                         help="use an in-memory NumPy vector store instead of PgVectorStore — no Postgres needed, "
                              "see local_vector_store.py for why/when this is and isn't an appropriate substitute")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.mode:
        config.data.data_mode = args.mode

    passages = list(iter_corpus(config))
    bm25, doc_ids, corpus = build_bm25_index(passages, config)

    if args.local_vector:
        cache_path = config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl"
        if cache_path.exists():
            vector_store = LocalVectorStore.load(config, cache_path)
        else:
            vector_store = LocalVectorStore(config)
            vector_store.build(passages)
            vector_store.save(cache_path)
    else:
        vector_store = PgVectorStore(config)
        vector_store.build(passages)

    app = build_graph(
        config, bm25, doc_ids, corpus, vector_store,
        llm=config.get_llm(), cross_encoder=config.get_reranker(),
    )

    root = config.data.nomiracl_local_dir
    relevant = load_nomiracl_subset(root, config.data.nomiracl_lang, args.split, "relevant")
    non_relevant = load_nomiracl_subset(root, config.data.nomiracl_lang, args.split, "non_relevant")

    queued = [("relevant", r) for r in relevant[: args.limit]] + [
        ("non_relevant", r) for r in non_relevant[: args.limit]
    ]

    rows = []
    for subset, record in queued:
        gold = {p["docid"]: p.get("relevance", 0) for p in record["passages"]}
        relevant_ids = {d for d, rel in gold.items() if rel > 0}

        result = app.invoke(
            {"question": record["query"], "retry_count": 0},
            config={"configurable": {"thread_id": record["query_id"]}, "recursion_limit": 50},
        )
        retrieved_ids = [d["doc_id"] for d in result.get("reranked_docs", [])]
        answer = result.get("answer", "")

        rows.append(
            {
                "subset": subset,
                "query_id": record["query_id"],
                "query": record["query"],
                "answer": answer,
                "abstained": is_abstain(answer),
                "retrieved_ids": retrieved_ids,
                "gold_relevant_ids": sorted(relevant_ids),
                "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10) if relevant_ids else None,
                # Gate on relevant_ids, not gold: a non_relevant query's gold
                # is all relevance-0, so its ideal DCG is 0 and ndcg_at_k
                # returns NaN — which then turned the whole mean into NaN.
                "ndcg_at_10": ndcg_at_k(retrieved_ids, gold, 10) if relevant_ids else None,
            }
        )

    recalls = [r["recall_at_10"] for r in rows if r["recall_at_10"] is not None]
    ndcgs = [r["ndcg_at_10"] for r in rows if r["ndcg_at_10"] is not None]
    err = error_rate([r["answer"] for r in rows if r["subset"] == "relevant"])
    halluc = hallucination_rate([r["answer"] for r in rows if r["subset"] == "non_relevant"])
    print("\n=== Mode B results ===")
    print(f"recall@10 (mean over {len(recalls)} relevant-subset queries): "
          f"{sum(recalls) / len(recalls):.3f}" if recalls else "recall@10: n/a")
    print(f"nDCG@10 (mean over {len(ndcgs)} relevant-subset queries): "
          f"{sum(ndcgs) / len(ndcgs):.3f}" if ndcgs else "nDCG@10: n/a")
    print(f"error_rate (relevant): {err:.3f}")
    print(f"hallucination_rate (non_relevant): {halluc:.3f}  <- read the examples file before trusting this, see docstring")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mode_b] full results -> {out_path}")

    sample_path = out_path.with_name(out_path.stem + "_examples.md")
    examples = random.sample(rows, k=min(args.dump_examples, len(rows)))
    with open(sample_path, "w", encoding="utf-8") as fh:
        fh.write("# Mode B — manual review sample\n\n")
        fh.write(
            "Read these by hand, especially `non_relevant` rows that answered instead of "
            "abstaining — that can mean our retriever legitimately found something MIRACL's "
            "original baseline missed, not that the model hallucinated. See module docstring.\n\n"
        )
        for r in examples:
            fh.write(f"## [{r['subset']}] {r['query_id']}\n\n")
            fh.write(f"**Q:** {r['query']}\n\n**A:** {r['answer']}\n\n")
            fh.write(f"retrieved: {r['retrieved_ids']}  \ngold relevant: {r['gold_relevant_ids']}\n\n---\n\n")
    print(f"[mode_b] {len(examples)} examples for manual review -> {sample_path}")


if __name__ == "__main__":
    main()
