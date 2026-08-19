# ================================================
# LANGGRAPH ORCHESTRATION
# Builds the agent graph with conditional routing
# ================================================

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    triage_node,
    retrieval_node,
    generation_node,
    verification_node,
    output_node
)
from config import MAX_RETRIES
from rich.console import Console

console = Console()

def should_retrieve(state: AgentState) -> str:
    """
    After triage: route based on classification.
    Out-of-scope and escalation skip retrieval.
    """
    classification = state["classification"]

    if classification in [
        "out_of_scope",
        "requires_escalation"
    ]:
        console.log(
            f"[yellow]ROUTE: triage → "
            f"generation "
            f"(skip retrieval: {classification})"
            f"[/yellow]")
        return "skip_retrieval"

    console.log(
        "[cyan]ROUTE: triage → retrieval[/cyan]")
    return "retrieve"


def should_retry(state: AgentState) -> str:
    """
    After verification: retry, output or safe fail.
    """
    passed = state.get("verification_passed", False)
    retry_count = state.get("retry_count", 0)
    classification = state["classification"]

    # Always pass for non-answerable
    if classification in [
        "out_of_scope",
        "requires_clarification",
        "requires_escalation"
    ]:
        console.log(
            "[cyan]ROUTE: verification → "
            "output[/cyan]")
        return "output"

    if passed:
        console.log(
            "[green]ROUTE: verification → "
            "output (passed)[/green]")
        return "output"

    # Retry if under limit
    if retry_count < MAX_RETRIES:
        console.log(
            f"[yellow]ROUTE: verification → "
            f"generation (retry "
            f"{retry_count + 1}/{MAX_RETRIES})"
            f"[/yellow]")
        return "retry"

    # Safe failure
    console.log(
        "[red]ROUTE: verification → "
        "output (safe_failure)[/red]")
    return "output"


def build_graph(kb, llm):
    """Build and compile the LangGraph agent"""

    # Wrap nodes to inject kb and llm
    def triage(state):
        return triage_node(state, kb, llm)

    def retrieval(state):
        return retrieval_node(state, kb, llm)

    def generation(state):
        return generation_node(state, kb, llm)

    def verification(state):
        return verification_node(state, kb, llm)

    def output(state):
        return output_node(state, kb, llm)

    # Retry wrapper — increments retry counter
    def generation_retry(state):
        new_state = {
            **state,
            "retry_count": state.get(
                "retry_count", 0) + 1
        }
        return generation_node(new_state, kb, llm)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("triage", triage)
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("generation", generation)
    workflow.add_node("generation_retry",
                      generation_retry)
    workflow.add_node("verification", verification)
    workflow.add_node("output", output)

    # Entry point
    workflow.set_entry_point("triage")

    # Triage → conditional routing
    workflow.add_conditional_edges(
        "triage",
        should_retrieve,
        {
            "retrieve": "retrieval",
            "skip_retrieval": "generation"
        }
    )

    # Retrieval → generation
    workflow.add_edge("retrieval", "generation")

    # Generation → verification
    workflow.add_edge("generation", "verification")

    # Verification → conditional routing
    workflow.add_conditional_edges(
        "verification",
        should_retry,
        {
            "output": "output",
            "retry": "generation_retry"
        }
    )

    # Retry generation → verification again
    workflow.add_edge("generation_retry",
                      "verification")

    # Output → END
    workflow.add_edge("output", END)

    return workflow.compile()