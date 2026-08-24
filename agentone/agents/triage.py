"""Triage and Intent Classification Agent Node."""

import json
import time
from typing import Any, Dict
from agentone.core.state import AgentState
from agentone.core.llm_provider import LLMProviderFactory
from agentone.core.telemetry import append_node_telemetry


TRIAGE_SYSTEM_PROMPT = """
You are the AgentOne Triage Specialist.
Analyze the incoming request and output a strictly formatted JSON object with the following fields:
- intent: Category of the request (e.g. BILLING_REFUND_INQUIRY, TECHNICAL_INCIDENT, ACCOUNT_ACCESS, GENERAL_INQUIRY)
- priority: One of P1_CRITICAL, P2_HIGH, P3_MEDIUM, P4_LOW
- sentiment_score: Float between -1.0 (extremely negative/upset) and 1.0 (positive)
- customer_id: Inferred or generated customer identifier (e.g. CUST-10492)
- recommended_route: Next optimal agent step ("rag_retrieval" or "tool_action")
- reasoning: Short explanation of your classification.
"""


async def triage_node(state: AgentState) -> Dict[str, Any]:
    """Execute triage classification on sanitized query."""
    start_t = time.perf_counter()
    llm = LLMProviderFactory.get_client()
    messages = [{"role": "user", "content": state["sanitized_query"]}]

    try:
        response = await llm.generate(
            messages=messages,
            temperature=0.1,
            system_instruction=TRIAGE_SYSTEM_PROMPT,
        )

        try:
            data = json.loads(response.content)
        except Exception:
            data = {
                "intent": "GENERAL_INQUIRY",
                "priority": "P3_MEDIUM",
                "sentiment_score": 0.0,
                "customer_id": "CUST-DEFAULT",
                "recommended_route": "rag_retrieval",
                "reasoning": "Fallback classification",
            }

        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"),
            "triage",
            dur_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model_name=response.model,
        )

        return {
            "intent": data.get("intent"),
            "priority": data.get("priority", "P3_MEDIUM"),
            "sentiment_score": data.get("sentiment_score"),
            "customer_id": data.get("customer_id"),
            "telemetry": tel,
            "current_agent": "triage_agent",
            "next_agent": data.get("recommended_route", "rag_retrieval"),
        }

    except Exception as e:
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"), "triage", dur_ms, status="error", error=str(e)
        )
        return {
            "intent": "GENERAL_INQUIRY",
            "priority": "P3_MEDIUM",
            "telemetry": tel,
            "current_agent": "triage_agent",
            "next_agent": "rag_retrieval",
        }
