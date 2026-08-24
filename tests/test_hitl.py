"""Unit tests for Human-in-the-Loop approval workflows."""

import pytest
from datetime import datetime, timezone
from agentone.hitl.approval_manager import ApprovalManager
from agentone.core.state import ApprovalRequest, AgentAction, ActionRiskLevel


def test_hitl_submission_and_approval():
    manager = ApprovalManager()
    action = AgentAction(
        action_id="act-test-01",
        tool_name="refund_process",
        tool_input={"amount": 1250.0, "currency": "USD"},
        risk_level=ActionRiskLevel.HIGH,
        description="High risk refund over $500",
        requires_approval=True,
    )
    req = ApprovalRequest(
        approval_id="app-test-01",
        thread_id="thread-test",
        proposed_action=action,
        reason_for_review="Exceeds $500 limit",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    manager.submit_request(req)
    assert len(manager.get_pending()) == 1

    approved = manager.approve("app-test-01", reviewer_comment="Authorized by finance lead.")
    assert approved is not None
    assert approved.status == "approved"
    assert approved.proposed_action.status == "executed"
    assert len(manager.get_pending()) == 0


def test_hitl_rejection():
    manager = ApprovalManager()
    action = AgentAction(
        action_id="act-test-02",
        tool_name="database_schema_migration",
        tool_input={"table": "users"},
        risk_level=ActionRiskLevel.CRITICAL,
        description="Destructive table schema drop",
        requires_approval=True,
    )
    req = ApprovalRequest(
        approval_id="app-test-02",
        thread_id="thread-test",
        proposed_action=action,
        reason_for_review="Critical action requires lead sign-off",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    manager.submit_request(req)
    rejected = manager.reject("app-test-02", reviewer_comment="Rejected due to maintenance window.")
    assert rejected is not None
    assert rejected.status == "rejected"
    assert rejected.proposed_action.status == "rejected"
    assert len(manager.get_pending()) == 0
