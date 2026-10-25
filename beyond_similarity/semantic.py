"""Optional semantic retrieval adapter powered by Sentence Transformers.

Install with: pip install -e ".[semantic]"
The core package remains usable without this dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .retrieval import CodeNode


@dataclass(frozen=True)
class SemanticResult:
    node_id: str
    score: float


class SemanticRetriever:
    """Cosine-similarity retriever with lazy optional dependency loading."""

    def __init__(
        self,
        nodes: Sequence[CodeNode],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'Semantic retrieval needs the optional extra: pip install -e ".[semantic]"'
            ) from exc
        if not nodes:
            raise ValueError("nodes must not be empty")
        self.nodes = list(nodes)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode(
            [node.searchable_text for node in self.nodes],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

    def search(self, query: str, k: int = 10) -> list[SemanticResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        order = sorted(
            range(len(self.nodes)),
            key=lambda index: (-float(scores[index]), self.nodes[index].id),
        )[:k]
        return [
            SemanticResult(self.nodes[index].id, float(scores[index]))
            for index in order
        ]
