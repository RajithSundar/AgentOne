"""Human-in-the-Loop (HITL) API routes for action governance."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
from agentone.hitl.approval_manager import get_approval_manager
from agentone.core.state import ApprovalRequest

router = APIRouter(prefix="/api/hitl", tags=["Human-in-the-Loop Governance"])


class ApprovalActionPayload(BaseModel):
    """Payload to approve or reject a pending agent action."""
    reviewer_comment: Optional[str] = Field(default=None, description="Operator explanation / audit notes")


@router.get("/pending", response_model=List[ApprovalRequest])
async def list_pending_approvals():
    """Retrieve all high-risk agent operations currently paused awaiting approval."""
    manager = get_approval_manager()
    return manager.get_pending()


@router.get("/{approval_id}", response_model=ApprovalRequest)
async def get_approval_details(approval_id: str = Path(...)):
    """Fetch details of a specific approval ticket."""
    manager = get_approval_manager()
    req = manager.get_by_id(approval_id)
    if not req:
        raise HTTPException(status_code=404, detail="Approval ticket not found.")
    return req


@router.post("/{approval_id}/approve", response_model=ApprovalRequest)
async def approve_action(approval_id: str, payload: ApprovalActionPayload):
    """Approve a paused high-risk operation and trigger immediate execution."""
    manager = get_approval_manager()
    approved = manager.approve(approval_id, payload.reviewer_comment)
    if not approved:
        raise HTTPException(status_code=404, detail="Approval ticket not found or already resolved.")
    return approved


@router.post("/{approval_id}/reject", response_model=ApprovalRequest)
async def reject_action(approval_id: str, payload: ApprovalActionPayload):
    """Reject a paused agent action."""
    manager = get_approval_manager()
    rejected = manager.reject(approval_id, payload.reviewer_comment)
    if not rejected:
        raise HTTPException(status_code=404, detail="Approval ticket not found or already resolved.")
    return rejected
