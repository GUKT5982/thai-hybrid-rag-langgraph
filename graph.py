"""Assembles the LangGraph StateGraph for the hybrid RAG query pipeline.

Indexing lives in indexing.py — this file only wires nodes together over
indexes that already exist.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config import AppConfig
from nodes.fusion import make_fusion_node
from nodes.generate import make_generate_node
from nodes.grade import make_grade_documents_node
from nodes.rerank import make_rerank_node
from nodes.retrieval import make_bm25_search_node, make_vector_search_node
from nodes.rewrite import make_query_rewrite_node
from state import RAGState


def _bump_retry(state: RAGState) -> dict:
    """Owns retry_count so grade_documents itself stays a pure judge with
    no routing side effects (see nodes/grade.py docstring)."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph(config: AppConfig, bm25_index, doc_ids, corpus, vector_store, llm=None, cross_encoder=None):
    """Wire all nodes into a compiled, checkpointed StateGraph.

    bm25_index / doc_ids / corpus / vector_store come from indexing.py
    (build_bm25_index / PgVectorStore). llm / cross_encoder are injectable
    for testing; default to config.get_llm() / config.get_reranker().
    """
    graph = StateGraph(RAGState)

    graph.add_node("query_rewrite", make_query_rewrite_node(config, llm=llm))
    graph.add_node("bm25_search", make_bm25_search_node(bm25_index, doc_ids, corpus, config))
    graph.add_node("vector_search", make_vector_search_node(vector_store, config))
    graph.add_node("fusion", make_fusion_node(config))
    graph.add_node("rerank", make_rerank_node(config, cross_encoder=cross_encoder))
    graph.add_node("grade_documents", make_grade_documents_node(config, llm=llm))
    graph.add_node("bump_retry", _bump_retry)
    graph.add_node("generate", make_generate_node(config, llm=llm))

    graph.add_edge(START, "query_rewrite")
    # Fan-out: both searches read state["rewritten_queries"] independently
    # of one another, so LangGraph runs them concurrently.
    graph.add_edge("query_rewrite", "bm25_search")
    graph.add_edge("query_rewrite", "vector_search")
    # Fan-in at fusion.
    graph.add_edge("bm25_search", "fusion")
    graph.add_edge("vector_search", "fusion")
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "grade_documents")

    def route(state: RAGState) -> str:
        if state.get("documents_sufficient"):
            return "generate"
        if state.get("retry_count", 0) >= config.graph.max_retries:
            # Give up gracefully rather than looping forever; generate()
            # will abstain on its own if reranked_docs is empty/weak.
            return "generate"
        return "bump_retry"

    graph.add_conditional_edges("grade_documents", route, {"bump_retry": "bump_retry", "generate": "generate"})
    graph.add_edge("bump_retry", "query_rewrite")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=MemorySaver())
