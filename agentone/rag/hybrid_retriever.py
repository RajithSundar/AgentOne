"""Hybrid Retrieval Engine uniting Dense Semantic Vectors and Sparse BM25 via Reciprocal Rank Fusion (RRF)."""

from typing import Dict, List, Optional
from agentone.core.state import DocumentChunk
from agentone.rag.vector_store import InMemoryVectorStore
from agentone.rag.bm25_search import BM25Index


class HybridRetriever:
    """Production hybrid search combining dense embeddings and BM25 lexical rankers."""

    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.vector_store = vector_store or InMemoryVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self.alpha = alpha  # Weight balance between dense (alpha) and sparse (1 - alpha)

    def add_documents(self, documents: List[DocumentChunk]):
        """Index chunks into both dense and sparse representations."""
        self.vector_store.add_documents(documents)
        self.bm25_index.add_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> List[DocumentChunk]:
        """
        Perform hybrid retrieval and Reciprocal Rank Fusion (RRF).
        RRF formula: RRF_score(d) = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_sparse))
        """
        current_alpha = alpha if alpha is not None else self.alpha

        dense_candidates = self.vector_store.search(
            query=query,
            top_k=top_k * 3,
            category_filter=category_filter,
        )

        sparse_candidates = self.bm25_index.search(
            query=query,
            top_k=top_k * 3,
            category_filter=category_filter,
        )

        rrf_scores: Dict[str, float] = {}
        merged_chunks: Dict[str, DocumentChunk] = {}

        # Process dense ranking
        for rank, (chunk, score) in enumerate(dense_candidates, start=1):
            doc_id = chunk.doc_id
            merged_chunks[doc_id] = chunk
            dense_contrib = current_alpha * (1.0 / (self.rrf_k + rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_contrib

        # Process sparse ranking
        for rank, (chunk, score) in enumerate(sparse_candidates, start=1):
            doc_id = chunk.doc_id
            if doc_id not in merged_chunks:
                merged_chunks[doc_id] = chunk
            else:
                # Merge sparse score into existing chunk model
                merged_chunks[doc_id].sparse_score = chunk.sparse_score

            sparse_contrib = (1.0 - current_alpha) * (1.0 / (self.rrf_k + rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_contrib

        # Build sorted list
        ranked_docs: List[DocumentChunk] = []
        for doc_id, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = merged_chunks[doc_id]
            chunk.rrf_score = round(rrf, 6)
            ranked_docs.append(chunk)

        return ranked_docs[:top_k]

    def count(self) -> int:
        """Return total document count."""
        return self.vector_store.count()
