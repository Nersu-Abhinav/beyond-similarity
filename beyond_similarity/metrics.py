"""Ranking metrics for issue-to-code localization."""
from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = set(relevant_ids)
    if not ranked_ids:
        return 0.0
    selected = ranked_ids[:k]
    return sum(item in relevant for item in selected) / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, node_id in enumerate(ranked_ids, start=1):
        if node_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
