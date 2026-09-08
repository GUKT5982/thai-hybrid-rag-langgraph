"""Interactive CLI: type your own question, watch every stage of the graph
(query_rewrite -> bm25_search + vector_search -> fusion -> rerank ->
grade_documents -> generate) with real doc_ids, scores, and content
snippets at each step — not just the final answer.

Uses the same components as scripts/smoke_graph.py (real BM25, real
LocalVectorStore, real reranker, real LLM) but loops on stdin instead of
iterating NoMIRACL records, so you can ask anything.

Usage:
    python indexing.py --mode sample --skip-vector      # build the BM25 index first, if not already
    python indexing.py --mode sample --local-vector      # build the vector store too (or reuse the cache)
    python -u -m scripts.ask
    python -u -m scripts.ask --no-vector                  # BM25-only, skip loading/building the vector store
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import AppConfig
from graph import build_graph
from indexing import load_bm25_index
from local_vector_store import LocalVectorStore
from nodes.generate import is_abstain


class NullVectorStore:
    def similarity_search(self, query: str, k: int) -> list[dict]:
        return []


def _snippet(text: str, n: int = 100) -> str:
    text = (text or "").replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def _print_hits(label: str, hits: list[dict]) -> None:
    print(f"\n{label} ({len(hits)}):", flush=True)
    if not hits:
        print("  (none)", flush=True)
    for h in hits:
        score = h.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        print(f"  [{h['doc_id']}] score={score_str}  {h.get('title', '')!r}", flush=True)
        print(f"      {_snippet(h.get('content', ''))}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-index", default=None, help="defaults to data_cache/bm25_<DATA_MODE>.pkl")
    parser.add_argument("--no-vector", action="store_true", help="skip vector search entirely (BM25-only)")
    args = parser.parse_args()

    config = AppConfig.from_env()
    bm25_path = Path(args.bm25_index) if args.bm25_index else config.data.cache_dir / f"bm25_{config.data.data_mode}.pkl"
    print(f"[ask] loading BM25 index from {bm25_path}", flush=True)
    bm25, doc_ids, corpus = load_bm25_index(bm25_path)

    if args.no_vector:
        vector_store = NullVectorStore()
        print("[ask] --no-vector: vector_search will return 0 hits every time", flush=True)
    else:
        vector_cache = config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl"
        if vector_cache.exists():
            vector_store = LocalVectorStore.load(config, vector_cache)
        else:
            print(f"[ask] no cached vector store at {vector_cache} — building one now "
                  "(can take a while on CPU; run `python indexing.py --local-vector` ahead of time to avoid this)...",
                  flush=True)
            vector_store = LocalVectorStore(config)
            vector_store.build({"doc_id": d, **corpus[d]} for d in doc_ids)
            vector_store.save(vector_cache)

    print(f"[ask] LLM: provider={config.model.llm_provider} model={config.model.llm_model}", flush=True)
    print(f"[ask] loading reranker {config.model.rerank_model}...", flush=True)
    cross_encoder = config.get_reranker()
    print("[ask] reranker ready", flush=True)

    app = build_graph(config, bm25, doc_ids, corpus, vector_store, llm=config.get_llm(), cross_encoder=cross_encoder)

    print("\n=== พร้อมแล้ว — พิมพ์คำถาม (Enter ว่างเพื่อออก) ===", flush=True)
    print("(แต่ละคำถามเรียก LLM หลายรอบ — ถ้าใช้โมเดล local อาจใช้เวลาหลายนาทีต่อคำถาม)\n", flush=True)

    turn = 0
    while True:
        try:
            question = input("คำถาม> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ask] bye")
            break
        if not question:
            print("[ask] bye")
            break

        turn += 1
        t0 = time.time()
        result = app.invoke(
            {"question": question, "retry_count": 0},
            config={"configurable": {"thread_id": f"ask-{turn}"}, "recursion_limit": 50},
        )
        elapsed = time.time() - t0

        print(f"\n--- ผลลัพธ์ ({elapsed:.1f}s) ---", flush=True)
        print("rewritten_queries:", result.get("rewritten_queries"), flush=True)

        _print_hits("BM25 hits", result.get("bm25_hits", []))
        _print_hits("Vector hits", result.get("vector_hits", []))
        _print_hits("Fused (RRF) — top of what goes into rerank", result.get("fused_docs", []))
        _print_hits("Reranked — top of what goes into generate", result.get("reranked_docs", []))

        print(f"\ndocuments_sufficient: {result.get('documents_sufficient')}"
              f"  (grade_reason: {result.get('grade_reason')!r})"
              f"  retry_count: {result.get('retry_count')}", flush=True)

        answer = result.get("answer", "")
        print(f"\nabstained: {is_abstain(answer)}", flush=True)
        print(f"\nคำตอบ:\n{answer}\n", flush=True)
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
