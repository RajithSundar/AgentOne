"""Telemetry, latency measurement, token usage tracking, and audit logging."""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPricing:
    """Estimated cost per million tokens (USD)."""
    PRICING = {
        "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
        "mock": {"prompt": 0.0, "completion": 0.0},
    }

    @classmethod
    def calculate_cost(cls, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated USD cost for token consumption."""
        rates = cls.PRICING.get(model_name, {"prompt": 1.00, "completion": 3.00})
        prompt_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
        completion_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
        return round(prompt_cost + completion_cost, 6)


class NodeExecutionSpan(BaseModel):
    """Detailed trace span for an individual agent node execution."""
    node_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "running"
    error: Optional[str] = None


class TelemetryTracker:
    """Collects and aggregates performance, cost, and trace metrics."""

    def __init__(self):
        self.spans: List[NodeExecutionSpan] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_duration_ms: float = 0.0
        self.audit_log: List[Dict[str, Any]] = []

    def start_span(self, node_name: str) -> NodeExecutionSpan:
        """Start tracking a new node execution span."""
        span = NodeExecutionSpan(
            node_name=node_name,
            start_time=time.perf_counter(),
            status="running",
        )
        self.spans.append(span)
        return span

    def end_span(
        self,
        span: NodeExecutionSpan,
        model_name: str = "mock",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        status: str = "success",
        error: Optional[str] = None,
    ) -> NodeExecutionSpan:
        """Complete a node execution span and update cumulative counters."""
        span.end_time = time.perf_counter()
        span.duration_ms = round((span.end_time - span.start_time) * 1000.0, 2)
        span.prompt_tokens = prompt_tokens
        span.completion_tokens = completion_tokens
        span.status = status
        span.error = error

        cost = ModelPricing.calculate_cost(model_name, prompt_tokens, completion_tokens)
        span.estimated_cost_usd = cost

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost
        self.total_duration_ms += span.duration_ms

        self.record_audit_event(
            node_name=span.node_name,
            event_type=f"node_{status}",
            details={
                "duration_ms": span.duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
                "error": error,
            },
        )
        return span

    def record_audit_event(self, node_name: str, event_type: str, details: Dict[str, Any]):
        """Append an event to the structured audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_name": node_name,
            "event_type": event_type,
            "details": details,
        }
        self.audit_log.append(entry)

    def summary(self) -> Dict[str, Any]:
        """Return a structured telemetry summary for API responses."""
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "spans_count": len(self.spans),
            "spans": [s.model_dump() for s in self.spans],
        }


def append_node_telemetry(
    current_telemetry: Optional[Dict[str, Any]],
    node_name: str,
    duration_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_name: str = "mock",
    status: str = "success",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure dictionary update function for node telemetry in state."""
    tel = dict(current_telemetry or {})
    spans = list(tel.get("spans", []))
    cost = ModelPricing.calculate_cost(model_name, prompt_tokens, completion_tokens)

    span_dict = {
        "node_name": node_name,
        "duration_ms": round(duration_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": cost,
        "status": status,
        "error": error,
    }
    spans.append(span_dict)

    tot_prompt = tel.get("total_prompt_tokens", 0) + prompt_tokens
    tot_comp = tel.get("total_completion_tokens", 0) + completion_tokens
    tot_cost = tel.get("total_cost_usd", 0.0) + cost
    tot_dur = tel.get("total_duration_ms", 0.0) + duration_ms

    return {
        "total_duration_ms": round(tot_dur, 2),
        "total_prompt_tokens": tot_prompt,
        "total_completion_tokens": tot_comp,
        "total_tokens": tot_prompt + tot_comp,
        "total_cost_usd": round(tot_cost, 6),
        "spans_count": len(spans),
        "spans": spans,
    }


def get_telemetry() -> TelemetryTracker:
    return TelemetryTracker()
