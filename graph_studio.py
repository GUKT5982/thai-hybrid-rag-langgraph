"""Entry point for `langgraph dev` / LangGraph Studio — not used by
eval/*.py or scripts/*.py, which all call graph.build_graph() directly
with indexes they load/build themselves. This module exists purely
because the `langgraph dev` CLI needs a zero-argument, importable
*compiled* graph object to serve, and build_graph() takes the BM25/vector
indexes as arguments — something the CLI has no way to supply.

Loads the cached sample-mode BM25 + LocalVectorStore that `indexing.py
--local-vector` builds (see README). Run that first if data_cache/ is
empty.

    langgraph dev
"""
from __future__ import annotations

from config import AppConfig
from graph import build_graph
from indexing import load_bm25_index
from local_vector_store import LocalVectorStore

config = AppConfig.from_env()

_bm25_path = config.data.cache_dir / f"bm25_{config.data.data_mode}.pkl"
if not _bm25_path.exists():
    raise FileNotFoundError(
        f"{_bm25_path} not found. Build it first:\n"
        f"    python indexing.py --mode {config.data.data_mode} --local-vector\n"
        "(or --skip-vector for BM25 only, if you don't need vector search in Studio)"
    )
bm25, doc_ids, corpus = load_bm25_index(_bm25_path)

_vector_path = config.data.cache_dir / f"local_vectors_{config.data.data_mode}.pkl"
if _vector_path.exists():
    vector_store = LocalVectorStore.load(config, _vector_path)
else:
    print(f"[graph_studio] no cached vector store at {_vector_path} — vector_search will "
          "return 0 hits every time. Run `python indexing.py --local-vector` to build one.")

    class _NullVectorStore:
        def similarity_search(self, query: str, k: int) -> list[dict]:
            return []

    vector_store = _NullVectorStore()

# The object `langgraph.json`'s "graphs" entry points at — a compiled,
# checkpointed StateGraph, ready for Studio to invoke.
app = build_graph(config, bm25, doc_ids, corpus, vector_store, llm=config.get_llm(), cross_encoder=config.get_reranker())
