"""Agent execution and real-time Server-Sent Events (SSE) streaming endpoints."""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentone.agents.graph import compile_agent_graph
from agentone.core.state import AgentState


router = APIRouter(prefix="/api/agents", tags=["Agents Orchestration"])


class AgentExecuteRequest(BaseModel):
    """Payload to initiate multi-agent workflow."""
    query: str = Field(..., description="User query or operation instruction")
    thread_id: Optional[str] = Field(default=None, description="Conversation / session ID")
    provider: Optional[str] = Field(default="mock", description="LLM provider: mock, gemini, openai, anthropic")
    customer_id: Optional[str] = Field(default=None, description="Optional customer context identifier")


class AgentExecuteResponse(BaseModel):
    """Synchronous execution result."""
    thread_id: str
    original_query: str
    sanitized_query: str
    final_response: str
    intent: Optional[str] = None
    priority: Optional[str] = None
    executed_actions: list = Field(default_factory=list)
    pending_actions: list = Field(default_factory=list)
    requires_human_approval: bool = False
    critic_notes: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: list = Field(default_factory=list)


@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent_pipeline(request: AgentExecuteRequest):
    """Run full LangGraph multi-agent pipeline and return aggregated outcome."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:8]}"

    app = compile_agent_graph()
    initial_state = {
        "thread_id": thread_id,
        "original_query": request.query,
        "sanitized_query": request.query,
        "messages": [{"role": "user", "content": request.query}],
        "current_agent": "supervisor",
        "next_agent": None,
        "iteration_count": 0,
        "is_complete": False,
        "status": "running",
        "intent": None,
        "priority": None,
        "customer_id": request.customer_id,
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

    try:
        final_state = await app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        return AgentExecuteResponse(
            thread_id=thread_id,
            original_query=request.query,
            sanitized_query=final_state.get("sanitized_query", request.query),
            final_response=final_state.get("final_response", ""),
            intent=final_state.get("intent"),
            priority=final_state.get("priority"),
            executed_actions=final_state.get("executed_actions", []),
            pending_actions=final_state.get("pending_actions", []),
            requires_human_approval=final_state.get("requires_human_approval", False),
            critic_notes=final_state.get("critic_notes"),
            telemetry=final_state.get("telemetry", {}),
            audit_trail=final_state.get("audit_trail", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-agent execution error: {str(e)}")


async def stream_agent_events(query: str, thread_id: str) -> AsyncIterator[str]:
    """Stream live LangGraph node transitions and thinking updates over SSE."""
    app = compile_agent_graph()

    initial_state = {
        "thread_id": thread_id,
        "original_query": query,
        "sanitized_query": query,
        "messages": [{"role": "user", "content": query}],
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

    yield f"data: {json.dumps({'event': 'start', 'thread_id': thread_id})}\n\n"
    await asyncio.sleep(0.05)

    last_telemetry = {}
    async for chunk in app.astream(initial_state, config={"configurable": {"thread_id": thread_id}}):
        for node_name, node_output in chunk.items():
            if "telemetry" in node_output:
                last_telemetry = node_output["telemetry"]
            payload = {
                "event": "node_update",
                "node": node_name,
                "data": {
                    "current_agent": node_output.get("current_agent", node_name),
                    "intent": node_output.get("intent"),
                    "priority": str(node_output.get("priority", "")),
                    "docs_count": len(node_output.get("retrieved_documents", [])),
                    "pending_actions": len(node_output.get("pending_actions", [])),
                    "executed_actions": len(node_output.get("executed_actions", [])),
                    "requires_approval": node_output.get("requires_human_approval", False),
                    "final_response": node_output.get("final_response"),
                },
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.05)

    yield f"data: {json.dumps({'event': 'complete', 'telemetry': last_telemetry})}\n\n"


@router.get("/stream")
async def stream_agent_pipeline(query: str = Query(..., min_length=1), thread_id: Optional[str] = None):
    """Server-Sent Events endpoint for real-time visualization."""
    session_id = thread_id or f"stream-{uuid.uuid4().hex[:8]}"
    return StreamingResponse(
        stream_agent_events(query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
