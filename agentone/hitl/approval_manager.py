"""Human-in-the-Loop (HITL) approval governance and checkpoint manager."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from agentone.core.state import ApprovalRequest, AgentAction


class ApprovalManager:
    """Central store and state manager for human-reviewed actions."""

    def __init__(self):
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.resolved_approvals: Dict[str, ApprovalRequest] = {}

    def submit_request(self, request: ApprovalRequest):
        """Register a new action requiring human review."""
        self.pending_approvals[request.approval_id] = request

    def get_pending(self) -> List[ApprovalRequest]:
        """Return all outstanding approval tickets."""
        return list(self.pending_approvals.values())

    def get_by_id(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Lookup approval by ID."""
        return self.pending_approvals.get(approval_id) or self.resolved_approvals.get(approval_id)

    def approve(self, approval_id: str, reviewer_comment: Optional[str] = None) -> Optional[ApprovalRequest]:
        """Approve a pending action and mark for execution."""
        req = self.pending_approvals.pop(approval_id, None)
        if not req:
            return None

        req.status = "approved"
        req.decided_at = datetime.now(timezone.utc).isoformat()
        req.reviewer_comment = reviewer_comment or "Approved by operator."
        req.proposed_action.status = "approved"

        # Execute approved action
        from agentone.agents.tool_executor import execute_mock_tool
        result = execute_mock_tool(req.proposed_action.tool_name, req.proposed_action.tool_input)
        req.proposed_action.result = result
        req.proposed_action.executed_at = datetime.now(timezone.utc).isoformat()
        req.proposed_action.status = "executed"

        self.resolved_approvals[approval_id] = req
        return req

    def reject(self, approval_id: str, reviewer_comment: Optional[str] = None) -> Optional[ApprovalRequest]:
        """Reject a pending action."""
        req = self.pending_approvals.pop(approval_id, None)
        if not req:
            return None

        req.status = "rejected"
        req.decided_at = datetime.now(timezone.utc).isoformat()
        req.reviewer_comment = reviewer_comment or "Rejected by operator policy."
        req.proposed_action.status = "rejected"

        self.resolved_approvals[approval_id] = req
        return req


_GLOBAL_APPROVAL_MANAGER = ApprovalManager()


def get_approval_manager() -> ApprovalManager:
    """Return singleton instance of approval manager."""
    return _GLOBAL_APPROVAL_MANAGER
