"""
Phase 4 | Script 09 — Planner + Executor (Agentic Workflow)
============================================================
Goal: Chain two specialised agents in a LangGraph graph.

What you'll learn:
- The Planner/Executor pattern: one agent plans, another executes
- How to pass state between nodes in a multi-step graph
- Why this is more powerful than a single agent for complex tasks
- How to loop over a list of steps dynamically

Graph:
  START → planner_node → executor_node (loops per step) → END

Run: python scripts/phase4_multiagent/09_planner_executor.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.tools import calculator as _calc, get_date as _get_date, search_docs as _search

SEPARATOR = "\n" + "─" * 60 + "\n"
MODEL = "llama3.2"


# ── Tools (same as script 08) ─────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '(47.50 * 0.15)'"""
    return _calc(expression)

@tool
def get_date(query: str = "") -> str:
    """Get today's date."""
    return _get_date(query)

@tool
def search_docs(query: str) -> str:
    """Search the local knowledge base. Input: a search query."""
    return _search(query)

tools = [calculator, get_date, search_docs]


# ── State ─────────────────────────────────────────────────────────────────
class WorkflowState(TypedDict):
    goal: str
    plan: list[str]          # list of step strings from the planner
    current_step_index: int  # which step we're executing
    step_results: list[str]  # results collected per step
    final_output: str        # assembled final answer


# ── Nodes ─────────────────────────────────────────────────────────────────
llm = ChatOllama(model=MODEL, temperature=0.0)
llm_with_tools = llm.bind_tools(tools)


def planner_node(state: WorkflowState) -> dict:
    """Generate a numbered list of steps to accomplish the goal."""
    print("[planner_node] Planning...")

    system = SystemMessage(content=(
        "You are a planning assistant. Given a goal, break it into 3-5 clear, concrete steps. "
        "Output ONLY a JSON array of step strings. Example:\n"
        '["Step 1: search for X", "Step 2: calculate Y", "Step 3: write summary"]'
    ))
    human = HumanMessage(content=f"Goal: {state['goal']}\n\nOutput the steps as a JSON array.")

    response = llm.invoke([system, human])
    raw = response.content.strip()

    # Extract JSON array from the response
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            steps = json.loads(raw[start:end])
        except json.JSONDecodeError:
            steps = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        steps = [raw]

    print(f"\nPlan ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    return {"plan": steps, "current_step_index": 0, "step_results": []}


def executor_node(state: WorkflowState) -> dict:
    """Execute the current step using tools if needed."""
    idx = state["current_step_index"]
    step = state["plan"][idx]
    results_so_far = state["step_results"]

    print(f"\n[executor_node] Step {idx + 1}/{len(state['plan'])}: {step}")

    context = ""
    if results_so_far:
        context = "\n\nPrevious step results:\n" + "\n".join(
            f"Step {i+1}: {r}" for i, r in enumerate(results_so_far)
        )

    system = SystemMessage(content=(
        "You are an executor. Complete the given step using tools if needed. "
        "Give a concise result (1-3 sentences)."
    ))
    human = HumanMessage(content=f"Complete this step: {step}{context}")

    # Simple single-turn tool use for each step
    response = llm_with_tools.invoke([system, human])

    result_text = response.content or ""

    # Execute any tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_map = {t.name: t for t in tools}
            if tc["name"] in tool_map:
                tool_result = tool_map[tc["name"]].invoke(tc["args"])
                result_text += f" [Tool: {tc['name']} → {tool_result}]"
                print(f"  → Used tool: {tc['name']}({tc['args']}) = {tool_result}")

    print(f"  → Result: {result_text[:200]}")

    new_results = results_so_far + [f"{step}: {result_text}"]
    return {
        "step_results": new_results,
        "current_step_index": idx + 1,
    }


def synthesiser_node(state: WorkflowState) -> dict:
    """Combine all step results into a final coherent output."""
    print("\n[synthesiser_node] Assembling final output...")

    steps_summary = "\n".join(
        f"{i+1}. {r}" for i, r in enumerate(state["step_results"])
    )

    system = SystemMessage(content=(
        "You are a writer. Synthesise the step results into a clear, well-structured final answer."
    ))
    human = HumanMessage(content=(
        f"Goal: {state['goal']}\n\n"
        f"Step results:\n{steps_summary}\n\n"
        "Write the final answer."
    ))

    response = llm.invoke([system, human])
    return {"final_output": response.content}


# ── Routing ───────────────────────────────────────────────────────────────
def route_after_executor(state: WorkflowState) -> str:
    """Loop back to executor if more steps remain, otherwise synthesise."""
    if state["current_step_index"] < len(state["plan"]):
        return "executor_node"
    return "synthesiser_node"


# ── Build graph ───────────────────────────────────────────────────────────
print("=" * 60)
print("Building Planner → Executor → Synthesiser graph")
print("=" * 60)

builder = StateGraph(WorkflowState)
builder.add_node("planner_node", planner_node)
builder.add_node("executor_node", executor_node)
builder.add_node("synthesiser_node", synthesiser_node)

builder.set_entry_point("planner_node")
builder.add_edge("planner_node", "executor_node")
builder.add_conditional_edges("executor_node", route_after_executor)
builder.add_edge("synthesiser_node", END)

workflow = builder.compile()

print("\nGraph:")
print("  START → planner → executor (loops) → synthesiser → END")


# ── Run ───────────────────────────────────────────────────────────────────
def run_workflow(goal: str):
    print(SEPARATOR)
    print(f"GOAL: {goal}")
    print(SEPARATOR)

    initial_state = WorkflowState(
        goal=goal,
        plan=[],
        current_step_index=0,
        step_results=[],
        final_output="",
    )

    final_state = workflow.invoke(initial_state)

    print(SEPARATOR)
    print("FINAL OUTPUT:")
    print(final_state["final_output"])
    return final_state


print("\n")
run_workflow("Explain what RAG is, its key components, and its main failure modes.")

print("\n\nDone! Key insight: specialised agents (planner/executor) outperform a single")
print("all-purpose agent on complex tasks — each node can be tuned for its specific role.")
