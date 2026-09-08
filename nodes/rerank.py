"""rerank node — cross-encoder rerank of the fused candidate list.

Reads:  state["question"], state["fused_docs"]
Writes: state["reranked_docs"]
"""
from __future__ import annotations

from config import AppConfig
from state import RAGState


def make_rerank_node(config: AppConfig, cross_encoder=None):
    """`cross_encoder` is injectable (must expose `.predict(list[(q, doc)]) ->
    list[float]`, matching sentence_transformers.CrossEncoder) so this node
    is testable without downloading a real model."""
    encoder = cross_encoder or config.get_reranker()

    def rerank(state: RAGState) -> dict:
        docs = state.get("fused_docs") or []
        if not docs:
            return {"reranked_docs": []}

        question = state["question"]
        pairs = [(question, d["content"]) for d in docs]
        scores = encoder.predict(pairs)

        reranked = sorted(zip(docs, scores), key=lambda ds: ds[1], reverse=True)
        top = [{**d, "score": float(s)} for d, s in reranked[: config.retrieval.rerank_top_k]]
        return {"reranked_docs": top}

    return rerank
