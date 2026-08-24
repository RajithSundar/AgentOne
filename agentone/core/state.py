"""Typed state and domain models for LangGraph multi-agent orchestration."""

from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class ActionRiskLevel(str, Enum):
    """Risk tier for proposed agent operations."""
    LOW = "low"            # Read-only operations, public info lookup
    MEDIUM = "medium"      # Ticket creation, non-critical notification
    HIGH = "high"          # Financial mutation, account status change, DB write
    CRITICAL = "critical"  # Data deletion, RBAC permission elevation


class TaskPriority(str, Enum):
    """Urgency tier evaluated by the Triage Agent."""
    P1_CRITICAL = "P1_CRITICAL"
    P2_HIGH = "P2_HIGH"
    P3_MEDIUM = "P3_MEDIUM"
    P4_LOW = "P4_LOW"


class AgentAction(BaseModel):
    """Represents a discrete executable action proposed or taken by an agent."""
    action_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    risk_level: ActionRiskLevel = ActionRiskLevel.LOW
    description: str
    requires_approval: bool = False
    status: str = "pending"  # pending, approved, rejected, executed, failed
    result: Optional[Any] = None
    executed_at: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Structured human-in-the-loop checkpoint payload."""
    approval_id: str
    thread_id: str
    proposed_action: AgentAction
    reason_for_review: str
    created_at: str
    decided_at: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected
    reviewer_comment: Optional[str] = None


class GuardrailVerdict(BaseModel):
    """Evaluation result from input/output security guardrails."""
    passed: bool = True
    shield_type: str  # e.g., "PII_REDACTOR", "INJECTION_SHIELD", "OUTPUT_VERIFIER"
    risk_score: float = 0.0
    detected_patterns: List[str] = Field(default_factory=list)
    sanitized_text: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Retrieved knowledge chunk with hybrid scores and metadata."""
    doc_id: str
    content: str
    source: str
    category: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_grade: Optional[str] = None


class AuditEntry(BaseModel):
    """Audit log entry capturing state transitions and decision traces."""
    timestamp: str
    node_name: str
    event_type: str
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Message passing unit across agent nodes."""
    sender: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """Complete graph execution state passed between LangGraph nodes."""

    # Thread and conversation context
    thread_id: str
    original_query: str
    sanitized_query: str
    messages: List[Dict[str, Any]]

    # Routing & orchestration control
    current_agent: str
    next_agent: Optional[str]
    iteration_count: int
    is_complete: bool
    status: str

    # Intent and triage metadata
    intent: Optional[str]
    priority: Optional[str]
    customer_id: Optional[str]
    sentiment_score: Optional[float]

    # Knowledge & RAG context
    retrieved_documents: List[Dict[str, Any]]
    knowledge_gap_detected: bool
    web_search_needed: bool

    # Action execution & Safety
    pending_actions: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    approval_requests: List[Dict[str, Any]]
    requires_human_approval: bool

    # Guardrails & Critic review
    guardrail_verdicts: List[Dict[str, Any]]
    critic_notes: Optional[str]
    critic_approved: bool

    # Output & Telemetry (all pure serializable dicts/lists)
    final_response: Optional[str]
    structured_output: Optional[Dict[str, Any]]
    audit_trail: List[Dict[str, Any]]
    telemetry: Dict[str, Any]
