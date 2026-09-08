"""fusion node — Reciprocal Rank Fusion (RRF).

Reads:  state["bm25_hits"], state["vector_hits"]
Writes: state["fused_docs"]

RRF score for a doc = sum, over each ranked list it appears in, of
1 / (k + rank) where rank is its 1-indexed position in that list. k
defaults to 60 (the value from the original RRF paper; Cormack et al. 2009
— tune it via config.retrieval.rrf_k). A doc that ranks #1 in one list and
is absent from the other still beats a doc that ranks #5 in both, which is
usually what you want: strong agreement from a single reliable signal
still counts.

We deliberately do NOT mix raw BM25 and cosine scores directly — they live
on unrelated scales (BM25 is an unbounded relevance score, cosine is
bounded [-1, 1]), so summing them would just let whichever score happens
to have the larger magnitude dominate the fused ranking. RRF only looks at
*rank*, sidestepping the scale problem entirely.
"""
from __future__ import annotations

from config import AppConfig
from state import RAGState


def make_fusion_node(config: AppConfig):
    def fusion(state: RAGState) -> dict:
        k = config.retrieval.rrf_k
        rrf_scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for hits in (state.get("bm25_hits") or [], state.get("vector_hits") or []):
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                docs.setdefault(doc_id, hit)

        ranked_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        fused = [{**docs[doc_id], "score": rrf_scores[doc_id]} for doc_id in ranked_ids]
        return {"fused_docs": fused[: config.retrieval.fusion_top_k]}

    return fusion
