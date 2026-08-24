"""Dense Vector Store with Cosine Similarity and Metadata Filtering."""

import math
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from agentone.core.state import DocumentChunk


def compute_semantic_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic semantic hash embedding generator.
    Creates rich, normalized dense vector representations based on n-gram token hashes
    and semantic frequency distribution.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return [0.0] * dim

    vec = [0.0] * dim
    for i, token in enumerate(tokens):
        h = hashlib.sha256(token.encode("utf-8")).digest()
        idx1 = int.from_bytes(h[:4], "big") % dim
        idx2 = int.from_bytes(h[4:8], "big") % dim
        idx3 = int.from_bytes(h[8:12], "big") % dim
        
        weight = 1.0 / math.log2(i + 2)
        vec[idx1] += 1.0 * weight
        vec[idx2] += 0.5 * weight
        vec[idx3] += 0.25 * weight

    # L2 Normalization
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two unit-normalized vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))


class VectorStore:
    """Interface for vector store implementations."""
    pass


class InMemoryVectorStore(VectorStore):
    """In-memory dense vector database with metadata filtering and top-k search."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents: List[DocumentChunk]):
        """Embed and index document chunks."""
        for doc in documents:
            embedding = compute_semantic_embedding(doc.content, self.embedding_dim)
            self.documents[doc.doc_id] = doc
            self.embeddings[doc.doc_id] = embedding

    def search(
        self,
        query: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search the dense index using cosine similarity."""
        if not self.documents:
            return []

        query_vec = compute_semantic_embedding(query, self.embedding_dim)
        results: List[Tuple[DocumentChunk, float]] = []

        for doc_id, doc in self.documents.items():
            if category_filter and doc.category.lower() != category_filter.lower():
                continue
            doc_vec = self.embeddings[doc_id]
            sim = cosine_similarity(query_vec, doc_vec)
            if sim >= min_score:
                doc_copy = doc.model_copy()
                doc_copy.dense_score = round(sim, 4)
                results.append((doc_copy, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self):
        """Reset the vector database."""
        self.documents.clear()
        self.embeddings.clear()

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self.documents)
