"""Retrieval-Augmented Generation & Self-Reflection Knowledge Agent."""

import json
import time
from typing import Any, Dict, List, Optional
from agentone.core.state import AgentState, DocumentChunk
from agentone.core.llm_provider import LLMProviderFactory
from agentone.core.telemetry import append_node_telemetry
from agentone.rag.hybrid_retriever import HybridRetriever
from agentone.rag.document_loader import load_knowledge_directory
from agentone.core.config import get_settings


_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None


def get_global_retriever() -> HybridRetriever:
    """Singleton getter for hybrid knowledge retriever."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        settings = get_settings()
        _GLOBAL_RETRIEVER = HybridRetriever(alpha=settings.rag_hybrid_alpha)
        chunks = load_knowledge_directory("data/sample_knowledge_base")
        if chunks:
            _GLOBAL_RETRIEVER.add_documents(chunks)
    return _GLOBAL_RETRIEVER


RAG_GRADER_SYSTEM_PROMPT = """
You are the AgentOne Knowledge Specialist & Self-RAG Grader.
Evaluate whether the retrieved knowledge documents contain sufficient, factual information to resolve the query.
Output a strict JSON object with fields:
- relevance_grade: "RELEVANT" or "IRRELEVANT"
- sufficient_for_answer: Boolean (true if fully answerable from documents)
- extracted_facts: List of string key facts extracted from the documents
- confidence_score: Float between 0.0 and 1.0
"""


async def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    """Execute hybrid retrieval and self-reflection document grading."""
    start_t = time.perf_counter()
    retriever = get_global_retriever()
    query = state["sanitized_query"]

    # Hybrid retrieval (dense cosine + sparse BM25 + RRF)
    retrieved_chunks = retriever.retrieve(query=query, top_k=4)

    llm = LLMProviderFactory.get_client()
    context_str = "\n\n".join(f"[{c.source}]: {c.content}" for c in retrieved_chunks)

    messages = [
        {"role": "user", "content": f"User Query: {query}\n\nRetrieved Runbook Context:\n{context_str}"}
    ]

    try:
        response = await llm.generate(
            messages=messages,
            temperature=0.1,
            system_instruction=RAG_GRADER_SYSTEM_PROMPT,
        )

        try:
            grade_data = json.loads(response.content)
        except Exception:
            grade_data = {
                "relevance_grade": "RELEVANT",
                "sufficient_for_answer": len(retrieved_chunks) > 0,
                "extracted_facts": [],
                "confidence_score": 0.85,
            }

        sufficient = grade_data.get("sufficient_for_answer", len(retrieved_chunks) > 0)
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"),
            "rag",
            dur_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model_name=response.model,
        )

        return {
            "retrieved_documents": [c.model_dump() for c in retrieved_chunks],
            "knowledge_gap_detected": not sufficient,
            "telemetry": tel,
            "current_agent": "rag_agent",
            "next_agent": "tool_executor",
        }

    except Exception as e:
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"), "rag", dur_ms, status="error", error=str(e)
        )
        return {
            "retrieved_documents": [c.model_dump() for c in retrieved_chunks],
            "knowledge_gap_detected": False,
            "telemetry": tel,
            "current_agent": "rag_agent",
            "next_agent": "tool_executor",
        }
