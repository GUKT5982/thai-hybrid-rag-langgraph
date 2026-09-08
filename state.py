"""LangGraph state shape for the query pipeline (graph.py)."""
from __future__ import annotations

from typing import TypedDict


class RetrievedDoc(TypedDict, total=False):
    doc_id: str
    score: float
    content: str
    title: str


class RAGState(TypedDict, total=False):
    question: str                       # original user question
    rewritten_queries: list[str]        # question + 2-3 paraphrased variants
    bm25_hits: list[RetrievedDoc]       # from nodes/retrieval.py bm25_search
    vector_hits: list[RetrievedDoc]     # from nodes/retrieval.py vector_search
    fused_docs: list[RetrievedDoc]      # after RRF (nodes/fusion.py)
    reranked_docs: list[RetrievedDoc]   # after cross-encoder rerank (nodes/rerank.py)
    answer: str                         # final generated answer
    retry_count: int                    # bumped by graph.py's bump_retry node

    # Extra fields beyond the brief's minimal state, needed to make the
    # grade -> retry conditional edge in graph.py actually routable:
    documents_sufficient: bool          # set by nodes/grade.py
    grade_reason: str                   # LLM's raw grading response, for debugging
