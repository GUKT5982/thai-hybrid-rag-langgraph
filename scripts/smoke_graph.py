"""Dev utility: run the FULL compiled graph.py pipeline against real
NoMIRACL questions, using real components everywhere — real BM25, real
vector search (LocalVectorStore, an in-memory NumPy store — see
local_vector_store.py; no Postgres needed), real LLM (whatever
LLM_PROVIDER/LLM_MODEL is set to, e.g. Ollama's llama3), real cross-encoder
reranker.

Not part of the eval/ suite (no metrics) — this is for eyeballing that
query_rewrite -> bm25_search + vector_search -> fusion -> rerank ->
grade_documents -> generate all actually cooperate with a real, slow,
occasionally-off-script local LLM, before spending time on eval/mode_b.py.

Usage:
    python indexing.py --mode sample --skip-vector   # build the BM25 index first
    python -m scripts.smoke_graph --limit 3
    python -m scripts.smoke_graph --limit 3 --subset non_relevant
    python -m scripts.smoke_graph --limit 3 --no-vector   # BM25-only, skip building the vector store too
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import AppConfig
from data.loaders import load_nomiracl_subset
from graph import build_graph
from indexing import load_bm25_index
from local_vector_store import LocalVectorStore
from nodes.generate import is_abstain


class NullVectorStore:
    """Stands in for a vector store when --no-vector is passed — always
    returns zero hits, so vector_search never breaks the graph, it just
    contributes nothing to fusion."""

    def similarity_search(self, query: str, k: int) -> list[dict]:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--subset", choices=["relevant", "non_relevant"], default="relevant")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--bm25-index", default=None, help="defaults to data_cache/bm25_<DATA_MODE>.pkl")
    parser.add_argument("--no-vector", action="store_true", help="skip building LocalVectorStore, BM25-only run")
    args = parser.parse_args()

    # print(..., flush=True) everywhere in this function: Windows consoles
    # sometimes fully buffer a non-interactive stdout instead of line-
    # buffering it, so without flush=True nothing appears for minutes at a
    # time even though the script is actively working (loading torch,
    # downloading the reranker model, waiting on a slow local LLM call) —
    # confirmed by watching python.exe's RSS climb in Task Manager while
    # the console stayed blank.
    config = AppConfig.from_env()
    bm25_path = Path(args.bm25_index) if args.bm25_index else config.data.cache_dir / f"bm25_{config.data.data_mode}.pkl"
    print(f"[smoke_graph] loading BM25 index from {bm25_path}", flush=True)
    bm25, doc_ids, corpus = load_bm25_index(bm25_path)

    if args.no_vector:
        vector_store = NullVectorStore()
        print("[smoke_graph] --no-vector: vector_search will return 0 hits every time", flush=True)
    else:
        vector_cache = config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl"
        t_vec = time.time()
        if vector_cache.exists():
            vector_store = LocalVectorStore.load(config, vector_cache)
        else:
            print(f"[smoke_graph] no cached vector store at {vector_cache} — building one now "
                  "(embeds every passage, ~161ms/passage on this machine's CPU — can take a while)...", flush=True)
            vector_store = LocalVectorStore(config)
            vector_store.build({"doc_id": d, **corpus[d]} for d in doc_ids)
            vector_store.save(vector_cache)
        print(f"[smoke_graph] LocalVectorStore ready ({time.time() - t_vec:.1f}s)", flush=True)

    print(f"[smoke_graph] LLM: provider={config.model.llm_provider} model={config.model.llm_model}", flush=True)
    print(f"[smoke_graph] loading reranker {config.model.rerank_model} "
          "(imports torch + downloads the model on first use — can take a minute)...", flush=True)
    t_load = time.time()
    cross_encoder = config.get_reranker()
    print(f"[smoke_graph] reranker ready ({time.time() - t_load:.1f}s)", flush=True)
    app = build_graph(config, bm25, doc_ids, corpus, vector_store, llm=config.get_llm(), cross_encoder=cross_encoder)

    root = config.data.nomiracl_local_dir
    records = load_nomiracl_subset(root, config.data.nomiracl_lang, args.split, args.subset)[: args.limit]
    print(f"[smoke_graph] running {len(records)} real {args.split}.{args.subset} queries through the full graph\n", flush=True)

    for record in records:
        print(f"[smoke_graph] query_id={record['query_id']} — calling the graph (query_rewrite -> ... -> "
              "generate; each LLM node can take up to ~1 min on a slow local model)...", flush=True)
        t0 = time.time()
        result = app.invoke(
            {"question": record["query"], "retry_count": 0},
            config={"configurable": {"thread_id": record["query_id"]}, "recursion_limit": 50},
        )
        elapsed = time.time() - t0
        answer = result.get("answer", "")

        print(f"--- query_id={record['query_id']}  ({elapsed:.1f}s) ---", flush=True)
        print("Q:", record["query"], flush=True)
        print("rewritten_queries:", result.get("rewritten_queries"), flush=True)
        print("bm25_hits:", len(result.get("bm25_hits", [])), " vector_hits:", len(result.get("vector_hits", [])), flush=True)
        print("reranked_docs:", [d["doc_id"] for d in result.get("reranked_docs", [])], flush=True)
        print("documents_sufficient:", result.get("documents_sufficient"), " retry_count:", result.get("retry_count"), flush=True)
        print("abstained:", is_abstain(answer), flush=True)
        print("A:", answer, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
