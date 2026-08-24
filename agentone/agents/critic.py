"""Critic, Compliance, and Response Synthesis Agent Node."""

import json
import time
from typing import Any, Dict, List
from agentone.core.state import AgentState, DocumentChunk
from agentone.core.llm_provider import LLMProviderFactory
from agentone.core.telemetry import append_node_telemetry
from agentone.guardrails.output_verifier import OutputVerifier


CRITIC_SYNTHESIS_PROMPT = """
You are the AgentOne Lead Operations Synthesizer & Reviewer.
Synthesize a professional, concise, and accurate response to the user query based on:
1. Retrieved Runbooks and SLA specifications
2. Executed or pending tool operations
3. Policy constraints

Do NOT use generic boilerplate phrases like "as an AI language model".
State facts directly, clearly outline actions taken, and present actionable next steps.
"""


async def critic_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize final client response and verify output compliance."""
    start_t = time.perf_counter()
    llm = LLMProviderFactory.get_client()
    query = state["sanitized_query"]
    intent = state.get("intent", "GENERAL_INQUIRY")
    docs = [DocumentChunk(**d) for d in state.get("retrieved_documents", [])]
    executed = state.get("executed_actions", [])
    pending = state.get("pending_actions", [])

    context_summary = "\n".join(f"- {d.source}: {d.content[:300]}" for d in docs)
    actions_summary = f"Executed: {len(executed)}, Pending Approvals: {len(pending)}"

    messages = [
        {
            "role": "user",
            "content": (
                f"Query: {query}\n"
                f"Intent: {intent}\n"
                f"Actions status: {actions_summary}\n"
                f"Retrieved Documentation:\n{context_summary}"
            ),
        }
    ]

    try:
        response = await llm.generate(
            messages=messages,
            temperature=0.2,
            system_instruction=CRITIC_SYNTHESIS_PROMPT,
        )

        final_text = response.content

        # Output guardrail verification
        verifier = OutputVerifier()
        verification = verifier.verify(
            generated_output=final_text,
            retrieved_documents=docs,
        )

        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"),
            "critic",
            dur_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model_name=response.model,
        )

        return {
            "final_response": verification.sanitized_output,
            "critic_notes": "Passed verification." if verification.is_valid else f"Violations: {verification.violations}",
            "critic_approved": verification.is_valid,
            "is_complete": True,
            "telemetry": tel,
            "current_agent": "critic_agent",
            "next_agent": None,
        }

    except Exception as e:
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"), "critic", dur_ms, status="error", error=str(e)
        )
        return {
            "final_response": "Your request has been processed by the operations system.",
            "critic_notes": f"Fallback synthesis due to: {e}",
            "critic_approved": False,
            "is_complete": True,
            "telemetry": tel,
            "current_agent": "critic_agent",
            "next_agent": None,
        }
