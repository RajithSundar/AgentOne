"""Supervisor and Orchestration Node for AgentOne Multi-Agent Graph."""

import time
from typing import Any, Dict
from agentone.core.state import AgentState
from agentone.core.telemetry import append_node_telemetry
from agentone.guardrails.input_shield import InputShield
from agentone.guardrails.pii_redactor import PIIRedactor


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor entry point:
    1. Runs InputShield against prompt injection/jailbreak.
    2. Runs PIIRedactor to sanitize sensitive customer data.
    3. Increments iteration count and routes to Triage Specialist.
    """
    start_t = time.perf_counter()
    original_query = state.get("original_query", "")
    iteration = state.get("iteration_count", 0) + 1

    # Layer 1: Prompt injection shield
    shield = InputShield()
    injection_verdict = shield.inspect(original_query)

    # Layer 2: PII Redaction
    redactor = PIIRedactor()
    pii_result = redactor.redact(injection_verdict.sanitized_input)

    guardrail_verdicts = list(state.get("guardrail_verdicts", []))
    guardrail_verdicts.append({
        "shield_type": "INPUT_SHIELD",
        "passed": injection_verdict.is_safe,
        "risk_score": injection_verdict.risk_score,
        "flags": injection_verdict.flags,
    })
    guardrail_verdicts.append({
        "shield_type": "PII_REDACTOR",
        "pii_found": pii_result.pii_found,
        "entity_counts": pii_result.entity_counts,
    })

    dur_ms = (time.perf_counter() - start_t) * 1000.0
    tel = append_node_telemetry(state.get("telemetry"), "supervisor", dur_ms)

    return {
        "sanitized_query": pii_result.redacted_text,
        "iteration_count": iteration,
        "guardrail_verdicts": guardrail_verdicts,
        "telemetry": tel,
        "current_agent": "supervisor_agent",
        "next_agent": "triage_agent",
    }
