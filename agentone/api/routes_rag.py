"""RAG knowledge base ingestion and hybrid search REST endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from agentone.core.state import DocumentChunk
from agentone.agents.rag_agent import get_global_retriever
from agentone.rag.document_loader import DocumentChunker

router = APIRouter(prefix="/api/rag", tags=["RAG & Knowledge Base"])


class DocumentIngestRequest(BaseModel):
    """Payload to ingest raw text or markdown into knowledge base."""
    title: str = Field(..., description="Document title / file name")
    content: str = Field(..., description="Raw text or markdown content")
    category: str = Field(default="general", description="Domain category tag")


class DocumentIngestResponse(BaseModel):
    """Ingestion confirmation."""
    status: str = "success"
    chunks_created: int
    total_indexed_documents: int


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """Chunk and index document into dense and sparse representations."""
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    chunker = DocumentChunker()
    chunks = chunker.chunk_markdown(
        text=request.content,
        source_name=request.title,
        category=request.category,
    )

    retriever = get_global_retriever()
    retriever.add_documents(chunks)

    return DocumentIngestResponse(
        status="success",
        chunks_created=len(chunks),
        total_indexed_documents=retriever.count(),
    )


@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=4, ge=1, le=20),
    category: Optional[str] = None,
    alpha: Optional[float] = Query(default=0.5, ge=0.0, le=1.0),
):
    """Perform hybrid retrieval with dense vector + BM25 + Reciprocal Rank Fusion (RRF)."""
    retriever = get_global_retriever()
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        category_filter=category,
        alpha=alpha,
    )

    return {
        "query": query,
        "results_count": len(results),
        "results": [chunk.model_dump() for chunk in results],
    }


@router.get("/stats")
async def get_rag_stats():
    """Return knowledge base status and total chunk counts."""
    retriever = get_global_retriever()
    return {
        "status": "ready",
        "total_chunks": retriever.count(),
        "dense_engine": "Cosine Semantic Hash / L2-Norm",
        "sparse_engine": "Okapi BM25 (k1=1.5, b=0.75)",
        "merging_algorithm": "Reciprocal Rank Fusion (RRF, k=60)",
    }
