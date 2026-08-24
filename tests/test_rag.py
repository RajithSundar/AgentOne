"""Unit tests for Hybrid RAG: Vector Store, BM25 Index, and RRF Retriever."""

import pytest
from agentone.core.state import DocumentChunk
from agentone.rag.vector_store import InMemoryVectorStore
from agentone.rag.bm25_search import BM25Index
from agentone.rag.hybrid_retriever import HybridRetriever
from agentone.rag.document_loader import DocumentChunker


def test_document_chunker():
    chunker = DocumentChunker(target_chunk_size=50)
    md = """# SLA Guide
Enterprise SLA guarantees 99.95% availability.

## Incident Policy
Critical incidents resolve within 2 hours."""
    chunks = chunker.chunk_markdown(md, source_name="sla_guide.md", category="sla")
    assert len(chunks) >= 2
    assert chunks[0].source == "sla_guide.md"


def test_dense_vector_store():
    vs = InMemoryVectorStore()
    docs = [
        DocumentChunk(doc_id="c1", content="Enterprise SLA uptime is 99.95 percent.", source="doc1", category="sla"),
        DocumentChunk(doc_id="c2", content="Refunds over 500 dollars require manager approval.", source="doc2", category="billing"),
    ]
    vs.add_documents(docs)
    assert vs.count() == 2

    results = vs.search("What is the uptime SLA?", top_k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "c1"


def test_bm25_index_exact_match():
    bm25 = BM25Index()
    docs = [
        DocumentChunk(doc_id="c1", content="Invoice INV-98214 was settled via Stripe ACH.", source="doc1", category="billing"),
        DocumentChunk(doc_id="c2", content="Kubernetes cluster ingress timeout is 60s.", source="doc2", category="infra"),
    ]
    bm25.add_documents(docs)

    results = bm25.search("INV-98214", top_k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "c1"


def test_hybrid_retriever_rrf():
    retriever = HybridRetriever(alpha=0.5)
    docs = [
        DocumentChunk(doc_id="c1", content="Enterprise SLA 99.95% uptime with 15min response for P1.", source="sla", category="sla"),
        DocumentChunk(doc_id="c2", content="Refunds up to 500 dollars processed automatically within 48h.", source="billing", category="billing"),
    ]
    retriever.add_documents(docs)

    results = retriever.retrieve("refund dollar limit policy", top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "c2"
    assert results[0].rrf_score > 0.0
