"""Small, dependency-free retrieval methods for code-localization research.

This is an interpretable baseline, not a neural embedding model or production agent.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\\d+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Tokenize text and retain complete identifiers plus identifier parts."""
    result: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.lower()
        result.append(token)
        parts = [part for part in token.split("_") if part]
        if len(parts) == 1:
            parts = [part for part in _CAMEL_BOUNDARY.sub("_", raw).lower().split("_") if part]
        if len(parts) > 1:
            result.extend(parts)
    return result


@dataclass(frozen=True)
class CodeNode:
    """Searchable repository component."""
    id: str
    name: str
    text: str = ""

    @property
    def searchable_text(self) -> str:
        return f"{self.name} {self.text}".strip()


@dataclass(frozen=True)
class RankedCandidate:
    node_id: str
    score: float
    lexical_score: float
    graph_score: float


class RepositoryIndex:
    """Searchable nodes plus an undirected dependency graph."""

    def __init__(
        self,
        nodes: Sequence[CodeNode],
        edges: Iterable[tuple[str, str]] = (),
    ) -> None:
        if not nodes:
            raise ValueError("nodes must contain at least one code component")
        self.nodes: dict[str, CodeNode] = {}
        for node in nodes:
            if not node.id:
                raise ValueError("node IDs must be non-empty")
            if node.id in self.nodes:
                raise ValueError(f"duplicate node ID: {node.id}")
            self.nodes[node.id] = node

        self.adjacency: dict[str, set[str]] = {
            node_id: set() for node_id in self.nodes
        }
        for left, right in edges:
            if left not in self.nodes or right not in self.nodes:
                raise ValueError(f"edge references unknown node: {left!r}, {right!r}")
            self.adjacency[left].add(right)
            self.adjacency[right].add(left)

        self._documents = {
            node_id: Counter(tokenize(node.searchable_text))
            for node_id, node in self.nodes.items()
        }
        document_frequency: Counter[str] = Counter()
        for counts in self._documents.values():
            document_frequency.update(counts.keys())
        total = len(self.nodes)
        self._idf = {
            token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._doc_lengths = {
            node_id: sum(counts.values())
            for node_id, counts in self._documents.items()
        }
        self._average_doc_length = (
            sum(self._doc_lengths.values()) / total if total else 0.0
        )

    def lexical_scores(self, query: str) -> dict[str, float]:
        """Return IDF-weighted query token scores normalized by document length."""
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return {node_id: 0.0 for node_id in self.nodes}

        raw: dict[str, float] = {}
        for node_id, document in self._documents.items():
            length = self._doc_lengths[node_id]
            norm = 0.75 + 0.25 * (
                length / self._average_doc_length
                if self._average_doc_length > 0
                else 0.0
            )
            score = 0.0
            for token, query_frequency in query_tokens.items():
                frequency = document.get(token, 0)
                if frequency:
                    score += (
                        query_frequency
                        * self._idf.get(token, 0.0)
                        * frequency
                        / norm
                    )
            raw[node_id] = score

        maximum = max(raw.values(), default=0.0)
        if maximum <= 0.0:
            return {node_id: 0.0 for node_id in self.nodes}
        return {node_id: score / maximum for node_id, score in raw.items()}

    def graph_proximity(
        self, seeds: Iterable[str], max_depth: int = 2
    ) -> dict[str, float]:
        """Score nodes by shortest distance from seed candidates.

        Distance 0 gets score 1.0; distance d > 0 gets score 1/(d+1).
        Nodes beyond max_depth or disconnected from seeds receive 0.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        seed_set = set(seeds)
        unknown = seed_set - self.nodes.keys()
        if unknown:
            raise ValueError(f"unknown seed nodes: {sorted(unknown)}")

        distance: dict[str, int] = {}
        queue: deque[str] = deque()
        for seed in seed_set:
            distance[seed] = 0
            queue.append(seed)

        while queue:
            current = queue.popleft()
            if distance[current] >= max_depth:
                continue
            for neighbor in self.adjacency[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)

        return {
            node_id: (
                1.0 / (distance[node_id] + 1) if node_id in distance else 0.0
            )
            for node_id in self.nodes
        }

    def rank_lexical(self, query: str) -> list[RankedCandidate]:
        lexical = self.lexical_scores(query)
        return [
            RankedCandidate(node_id, lexical[node_id], lexical[node_id], 0.0)
            for node_id in sorted(self.nodes, key=lambda item: (-lexical[item], item))
        ]

    def rank_graph_guided(
        self,
        query: str,
        *,
        graph_weight: float = 0.25,
        max_depth: int = 2,
        seed_limit: int = 5,
    ) -> list[RankedCandidate]:
        """Rank candidates using lexical relevance and graph proximity.

        Graph proximity is computed around the top lexical candidates. This is a
        transparent baseline: graph_weight must be tuned on development data.
        """
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("graph_weight must be between 0 and 1")
        if seed_limit < 1:
            raise ValueError("seed_limit must be at least 1")

        lexical = self.lexical_scores(query)
        seeds = [
            node_id
            for node_id in sorted(self.nodes, key=lambda item: (-lexical[item], item))
            if lexical[node_id] > 0.0
        ][:seed_limit]
        graph = self.graph_proximity(seeds, max_depth=max_depth) if seeds else {
            node_id: 0.0 for node_id in self.nodes
        }

        scores = {
            node_id: (1.0 - graph_weight) * lexical[node_id]
            + graph_weight * graph[node_id]
            for node_id in self.nodes
        }
        return [
            RankedCandidate(node_id, scores[node_id], lexical[node_id], graph[node_id])
            for node_id in sorted(self.nodes, key=lambda item: (-scores[item], item))
        ]


def rank_ids(candidates: Sequence[RankedCandidate]) -> list[str]:
    return [candidate.node_id for candidate in candidates]
