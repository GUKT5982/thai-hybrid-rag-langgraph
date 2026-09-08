"""Metric functions shared by eval/mode_a.py, eval/mode_b.py, eval/ablation.py.
Pure functions over already-loaded results — no I/O here.
"""
from __future__ import annotations

import math

from nodes.generate import is_abstain


def hallucination_rate(answers: list[str]) -> float:
    """`answers` are model outputs for non_relevant-subset queries (no
    passage in context actually answers the question). Correct behavior is
    abstaining; hallucination_rate = fraction that answered anyway instead
    of abstaining."""
    if not answers:
        return float("nan")
    hallucinated = sum(1 for a in answers if not is_abstain(a))
    return hallucinated / len(answers)


def error_rate(answers: list[str]) -> float:
    """`answers` are model outputs for relevant-subset queries (some
    passage in context does answer the question). Correct behavior is
    answering; error_rate = fraction that abstained anyway (over-cautious
    refusal, the metric that trades off against hallucination_rate)."""
    if not answers:
        return float("nan")
    refused = sum(1 for a in answers if is_abstain(a))
    return refused / len(answers)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    ideal_order = sorted(relevance, key=lambda d: relevance[d], reverse=True)
    ideal = dcg_at_k(ideal_order, relevance, k)
    if ideal == 0:
        return float("nan")
    return dcg_at_k(retrieved_ids, relevance, k) / ideal
