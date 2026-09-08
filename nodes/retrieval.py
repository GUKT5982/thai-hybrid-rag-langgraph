"""Retrieval nodes: bm25_search and vector_search.

Both read state["rewritten_queries"] only — neither reads the other's
output — so graph.py fans them out from query_rewrite and fans them back
in at fusion, and LangGraph runs them concurrently rather than chained.
"""
from __future__ import annotations

from config import AppConfig
from state import RAGState
from text_utils import thai_tokenize


def make_bm25_search_node(bm25_index, doc_ids: list[str], corpus: dict[str, dict], config: AppConfig):
    """bm25_index: rank_bm25.BM25Okapi built over `corpus`, tokenized with
    the same use_newmm setting used here — see indexing.build_bm25_index.
    corpus: doc_id -> {"title", "text"} lookup for hydrating hits.

    Reads:  state["rewritten_queries"] (falls back to state["question"])
    Writes: state["bm25_hits"]
    """

    def bm25_search(state: RAGState) -> dict:
        queries = state.get("rewritten_queries") or [state["question"]]
        scored: dict[str, float] = {}
        for q in queries:
            tokens = thai_tokenize(q, use_newmm=config.retrieval.use_newmm)
            scores = bm25_index.get_scores(tokens)
            for doc_id, score in zip(doc_ids, scores):
                if score <= 0:
                    continue
                scored[doc_id] = max(scored.get(doc_id, 0.0), float(score))

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[: config.retrieval.bm25_k]
        hits = [
            {
                "doc_id": doc_id,
                "score": score,
                "content": corpus[doc_id]["text"],
                "title": corpus[doc_id].get("title", ""),
            }
            for doc_id, score in ranked
        ]
        return {"bm25_hits": hits}

    return bm25_search


def make_vector_search_node(vector_store, config: AppConfig):
    """vector_store: object exposing `.similarity_search(query, k) -> list[dict]`
    with cosine similarity scores — see indexing.PgVectorStore.

    Reads:  state["rewritten_queries"] (falls back to state["question"])
    Writes: state["vector_hits"]
    """

    def vector_search(state: RAGState) -> dict:
        queries = state.get("rewritten_queries") or [state["question"]]
        best: dict[str, dict] = {}
        for q in queries:
            for hit in vector_store.similarity_search(q, k=config.retrieval.vector_k):
                doc_id = hit["doc_id"]
                if doc_id not in best or hit["score"] > best[doc_id]["score"]:
                    best[doc_id] = hit

        ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)[: config.retrieval.vector_k]
        return {"vector_hits": ranked}

    return vector_search
