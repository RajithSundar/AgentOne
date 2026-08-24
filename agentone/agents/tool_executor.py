"""Action dispatcher and safe enterprise tool execution agent node."""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from agentone.core.state import (
    AgentState,
    AgentAction,
    ActionRiskLevel,
    ApprovalRequest,
)
from agentone.core.llm_provider import LLMProviderFactory
from agentone.core.telemetry import append_node_telemetry
from agentone.core.config import get_settings


TOOL_EXEC_SYSTEM_PROMPT = """
You are the AgentOne Action Specialist.
Based on the user request, triage metadata, and retrieved runbooks, determine the operational action to execute.
Output a strict JSON object with fields:
- action_name: Name of the action (e.g. "refund_process", "create_support_ticket", "fetch_account_status", "send_notification")
- parameters: Dictionary of action arguments
- risk_level: One of "low", "medium", "high", "critical"
- requires_approval: Boolean (true if high-risk financial or destructive mutation)
- rationale: Clear explanation of the intended operation
"""


def execute_mock_tool(action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate execution of external system tools & APIs."""
    if action_name == "refund_process":
        amount = parameters.get("amount", 0.0)
        return {
            "status": "success",
            "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
            "amount_refunded": amount,
            "currency": parameters.get("currency", "USD"),
            "ledger_updated": True,
            "message": f"Refund of ${amount} initiated via payment gateway.",
        }
    elif action_name == "create_support_ticket":
        ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "queue": parameters.get("queue", "Tier-2 Support"),
            "priority": parameters.get("priority", "P2"),
            "assigned_lead": "Ops-Dispatcher-01",
            "message": f"Support ticket {ticket_id} created successfully.",
        }
    elif action_name == "fetch_account_status":
        user_id = parameters.get("user_id", "USR-DEFAULT")
        return {
            "status": "active",
            "user_id": user_id,
            "plan": "Enterprise Pro",
            "mrr": 499.00,
            "open_tickets": 0,
            "health_score": 98,
        }
    else:
        return {
            "status": "executed",
            "action": action_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": parameters,
        }


async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """Determine, validate, and execute operational actions with safety gates."""
    start_t = time.perf_counter()
    settings = get_settings()
    llm = LLMProviderFactory.get_client()

    query = state["sanitized_query"]
    intent = state.get("intent", "GENERAL_INQUIRY")
    docs = state.get("retrieved_documents", [])
    docs_summary = "\n".join(d.get("content", "")[:200] for d in docs)

    messages = [
        {
            "role": "user",
            "content": f"Query: {query}\nIntent: {intent}\nRunbook context:\n{docs_summary}",
        }
    ]

    try:
        response = await llm.generate(
            messages=messages,
            temperature=0.1,
            system_instruction=TOOL_EXEC_SYSTEM_PROMPT,
        )

        try:
            tool_data = json.loads(response.content)
        except Exception:
            tool_data = {
                "action_name": "fetch_account_status",
                "parameters": {"user_id": state.get("customer_id", "USR-1001")},
                "risk_level": "low",
                "requires_approval": False,
                "rationale": "Default account verification action",
            }

        action_name = tool_data.get("action_name", "fetch_account_status")
        params = tool_data.get("parameters", {})
        risk_str = tool_data.get("risk_level", "low").lower()
        risk_level = ActionRiskLevel(risk_str if risk_str in ActionRiskLevel._value2member_map_ else "low")

        requires_approval = tool_data.get("requires_approval", False)
        if action_name in settings.hitl_action_list:
            requires_approval = True

        if action_name == "refund_process":
            amount = float(params.get("amount", 0.0))
            if amount >= settings.hitl_refund_threshold_usd:
                requires_approval = True
                risk_level = ActionRiskLevel.HIGH

        action_id = f"ACT-{uuid.uuid4().hex[:8]}"
        action = AgentAction(
            action_id=action_id,
            tool_name=action_name,
            tool_input=params,
            risk_level=risk_level,
            description=tool_data.get("rationale", f"Execution of {action_name}"),
            requires_approval=requires_approval,
            status="pending" if requires_approval else "executed",
        )

        pending_actions = list(state.get("pending_actions", []))
        executed_actions = list(state.get("executed_actions", []))
        approval_requests = list(state.get("approval_requests", []))

        if requires_approval:
            approval_id = f"APP-{uuid.uuid4().hex[:8]}"
            approval_req = ApprovalRequest(
                approval_id=approval_id,
                thread_id=state.get("thread_id", "default"),
                proposed_action=action,
                reason_for_review=f"Action '{action_name}' classified as {risk_level.value.upper()} risk.",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            approval_requests.append(approval_req.model_dump())
            pending_actions.append(action.model_dump())
            needs_human = True
            
            # Register with global approval manager
            from agentone.hitl.approval_manager import get_approval_manager
            get_approval_manager().submit_request(approval_req)
        else:
            result = execute_mock_tool(action_name, params)
            action.result = result
            action.executed_at = datetime.now(timezone.utc).isoformat()
            executed_actions.append(action.model_dump())
            needs_human = False

        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"),
            "tool_executor",
            dur_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model_name=response.model,
        )

        return {
            "pending_actions": pending_actions,
            "executed_actions": executed_actions,
            "approval_requests": approval_requests,
            "requires_human_approval": needs_human,
            "telemetry": tel,
            "current_agent": "tool_executor",
            "next_agent": "critic_agent",
        }

    except Exception as e:
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        tel = append_node_telemetry(
            state.get("telemetry"), "tool_executor", dur_ms, status="error", error=str(e)
        )
        return {
            "requires_human_approval": False,
            "telemetry": tel,
            "current_agent": "tool_executor",
            "next_agent": "critic_agent",
        }
