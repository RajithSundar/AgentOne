"""Integration tests for LangGraph multi-agent workflow nodes."""

import pytest
from agentone.agents.graph import compile_agent_graph


@pytest.mark.asyncio
async def test_full_agent_workflow_execution():
    app = compile_agent_graph()

    initial_state = {
        "thread_id": "test-thread-001",
        "original_query": "Can I get a refund of $350 for invoice INV-102? My email is user@domain.com",
        "sanitized_query": "",
        "messages": [{"role": "user", "content": "Can I get a refund of $350 for invoice INV-102?"}],
        "current_agent": "supervisor",
        "next_agent": None,
        "iteration_count": 0,
        "is_complete": False,
        "status": "running",
        "intent": None,
        "priority": None,
        "customer_id": None,
        "sentiment_score": None,
        "retrieved_documents": [],
        "knowledge_gap_detected": False,
        "web_search_needed": False,
        "pending_actions": [],
        "executed_actions": [],
        "approval_requests": [],
        "requires_human_approval": False,
        "guardrail_verdicts": [],
        "critic_notes": None,
        "critic_approved": False,
        "final_response": None,
        "structured_output": None,
        "audit_trail": [],
        "telemetry": {},
    }

    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": "test-thread-001"}},
    )

    assert result["is_complete"] is True
    assert result["sanitized_query"] != ""
    assert "[EMAIL_1]" in result["sanitized_query"]
    assert result["final_response"] is not None
    assert len(result["final_response"]) > 0
    assert result["intent"] is not None
    assert result["telemetry"]["spans_count"] >= 4
