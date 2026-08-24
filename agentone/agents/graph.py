"""Compilation of LangGraph StateGraph multi-agent workflow."""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agentone.core.state import AgentState
from agentone.agents.supervisor import supervisor_node
from agentone.agents.triage import triage_node
from agentone.agents.rag_agent import rag_agent_node
from agentone.agents.tool_executor import tool_executor_node
from agentone.agents.critic import critic_node


def route_after_triage(state: AgentState) -> str:
    """Conditional edge routing after triage analysis."""
    next_step = state.get("next_agent", "rag_retrieval")
    if next_step == "tool_action":
        return "tool_executor"
    return "rag"


def route_after_tool(state: AgentState) -> str:
    """Routing after action execution."""
    return "critic"


def create_agent_graph() -> StateGraph:
    """Build the multi-agent execution StateGraph."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("rag", rag_agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("critic", critic_node)

    # Register Edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "triage")
    workflow.add_conditional_edges("triage", route_after_triage, {"rag": "rag", "tool_executor": "tool_executor"})
    workflow.add_edge("rag", "tool_executor")
    workflow.add_conditional_edges("tool_executor", route_after_tool, {"critic": "critic"})
    workflow.add_edge("critic", END)

    return workflow


def compile_agent_graph(checkpointer: Any = None):
    """Compile graph into a runnable multi-agent pipeline with memory checkpointer."""
    workflow = create_agent_graph()
    memory = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=memory)
