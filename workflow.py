import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from agents.idea_generator_agent import idea_generator_agent_node, idea_selector_node
from agents.research_agent import research_agent_node
from agents.freshness_agent import freshness_check_agent_node
from agents.human_approval_agent import human_approval_node
from agents.image_prompt_agent import image_prompt_agent_node
from agents.image_generation_agent import image_generation_agent_node
from agents.script_agent import script_agent_node

MAX_ITERATIONS = 3


# ── Routing functions ──────────────────────────────────────────────────────────

def route_entry(state: AgentState) -> str:
    return "idea_generator" if state.get("use_idea_generator") else "research"


def route_after_idea_selector(state: AgentState) -> str:
    """Loop back to generator if user wants more ideas, else proceed to research."""
    return "research" if state.get("selected_idea") else "idea_generator"


def route_after_freshness(state: AgentState) -> str:
    if state.get("freshness_approved") and state.get("best_idea"):
        return "ask_human"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "ask_human"
    return "retry"


def route_after_approval(state: AgentState) -> str:
    return "generate_images" if state.get("human_approved") else "restart"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("idea_generator",  idea_generator_agent_node)
    graph.add_node("idea_selector",   idea_selector_node)
    graph.add_node("research",        research_agent_node)
    graph.add_node("freshness_check", freshness_check_agent_node)
    graph.add_node("human_approval",  human_approval_node)
    graph.add_node("image_prompts",   image_prompt_agent_node)
    graph.add_node("image_generation",image_generation_agent_node)
    graph.add_node("script",          script_agent_node)

    # Entry: route node decides whether to use idea generator or skip straight to research
    graph.add_node("entry_router", lambda s: {})
    graph.set_entry_point("entry_router")
    graph.add_conditional_edges(
        "entry_router", route_entry,
        {"idea_generator": "idea_generator", "research": "research"},
    )

    # idea_generator → idea_selector (selector has the interrupt)
    graph.add_edge("idea_generator", "idea_selector")

    # After user picks: more → regenerate, or selected → research
    graph.add_conditional_edges(
        "idea_selector",
        route_after_idea_selector,
        {"research": "research", "idea_generator": "idea_generator"},
    )

    # Research pipeline
    graph.add_edge("research", "freshness_check")
    graph.add_conditional_edges(
        "freshness_check",
        route_after_freshness,
        {"ask_human": "human_approval", "retry": "research"},
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"generate_images": "image_prompts", "restart": "research"},
    )

    # Output pipeline
    graph.add_edge("image_prompts",    "image_generation")
    graph.add_edge("image_generation", "script")
    graph.add_edge("script",           END)

    return graph.compile(checkpointer=checkpointer)
