"""
Phase 4 | Script 10 — Reflection Loop (Critic + Revision)
==========================================================
Goal: Add a Critic node that reviews the output and can request revisions.
This is how agentic systems improve their own output through self-critique.

What you'll learn:
- The Generator → Critic → (revise or finish) pattern
- How to add a revision loop to any LangGraph workflow
- Using a max_revisions guard to prevent infinite loops
- Why reflection dramatically improves output quality

Graph:
  START → generator → critic → [revise (back to generator) | END]

Run: python scripts/phase4_multiagent/10_reflection_loop.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

SEPARATOR = "\n" + "─" * 60 + "\n"
MODEL = "llama3.2"
MAX_REVISIONS = 2  # safety guard — allow at most this many revision cycles


# ── State ─────────────────────────────────────────────────────────────────
class ReflectionState(TypedDict):
    task: str           # the writing/generation task
    draft: str          # current draft output
    feedback: str       # critic's feedback on the current draft
    revision_count: int # how many times we've revised
    final: str          # the approved final output


# ── LLM ──────────────────────────────────────────────────────────────────
llm = ChatOllama(model=MODEL, temperature=0.3)
critic_llm = ChatOllama(model=MODEL, temperature=0.0)  # critic should be consistent


# ── Nodes ─────────────────────────────────────────────────────────────────
def generator_node(state: ReflectionState) -> dict:
    """Generate or revise a draft based on the task and any prior feedback."""
    revision = state["revision_count"]

    if revision == 0:
        print(f"[generator] Creating initial draft...")
        prompt_text = f"Write a response for this task:\n\n{state['task']}"
    else:
        print(f"[generator] Revising draft (revision {revision})...")
        prompt_text = (
            f"Task: {state['task']}\n\n"
            f"Your previous draft:\n{state['draft']}\n\n"
            f"Critic's feedback:\n{state['feedback']}\n\n"
            "Rewrite the draft addressing all the feedback."
        )

    system = SystemMessage(content="You are a skilled writer. Produce clear, accurate, well-structured content.")
    response = llm.invoke([system, HumanMessage(content=prompt_text)])
    draft = response.content

    print(f"\nDraft preview: {draft[:200]}...")
    return {"draft": draft}


def critic_node(state: ReflectionState) -> dict:
    """Review the draft and decide whether it meets quality standards."""
    print(f"\n[critic] Reviewing draft (revision {state['revision_count']})...")

    system = SystemMessage(content=(
        "You are a critical reviewer. Evaluate the draft against the task. "
        "Respond in this EXACT format:\n\n"
        "VERDICT: APPROVE or REVISE\n"
        "FEEDBACK: <specific, actionable feedback if REVISE — or 'Meets all requirements' if APPROVE>"
    ))
    human = HumanMessage(content=(
        f"Task: {state['task']}\n\n"
        f"Draft:\n{state['draft']}\n\n"
        "Does this draft fully and accurately complete the task?"
    ))

    response = critic_llm.invoke([system, human])
    raw = response.content.strip()
    print(f"\nCritic response:\n{raw}")

    # Parse verdict and feedback
    verdict = "APPROVE"  # default to approve
    feedback = ""

    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = "REVISE" if "REVISE" in line.upper() else "APPROVE"
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    return {"feedback": feedback, "revision_count": state["revision_count"] + 1}


def finalise_node(state: ReflectionState) -> dict:
    """Accept the current draft as the final output."""
    print(f"\n[finalise] Draft approved after {state['revision_count']} revision(s).")
    return {"final": state["draft"]}


# ── Routing ───────────────────────────────────────────────────────────────
def route_after_critic(state: ReflectionState) -> str:
    """Approve if feedback is positive or max revisions reached; otherwise revise."""
    feedback_lower = state["feedback"].lower()
    approved = (
        "meets" in feedback_lower
        or "good" in feedback_lower
        or "well done" in feedback_lower
        or "approve" in feedback_lower
        or state["revision_count"] >= MAX_REVISIONS
    )

    if state["revision_count"] >= MAX_REVISIONS:
        print(f"\n[router] Max revisions ({MAX_REVISIONS}) reached — accepting draft.")

    if approved:
        return "finalise_node"
    else:
        print(f"\n[router] Revisions needed → sending back to generator")
        return "generator_node"


# ── Build graph ───────────────────────────────────────────────────────────
print("=" * 60)
print("Building Generator → Critic → (revise loop or END) graph")
print("=" * 60)
print(f"Max revisions allowed: {MAX_REVISIONS}")
print("\nGraph:")
print("  START → generator → critic → [generator (loop) | finalise → END]")

builder = StateGraph(ReflectionState)
builder.add_node("generator_node", generator_node)
builder.add_node("critic_node", critic_node)
builder.add_node("finalise_node", finalise_node)

builder.set_entry_point("generator_node")
builder.add_edge("generator_node", "critic_node")
builder.add_conditional_edges("critic_node", route_after_critic)
builder.add_edge("finalise_node", END)

reflection_graph = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────
def run_reflection(task: str):
    print(SEPARATOR)
    print(f"TASK: {task}")
    print(SEPARATOR)

    initial_state = ReflectionState(
        task=task,
        draft="",
        feedback="",
        revision_count=0,
        final="",
    )

    final_state = reflection_graph.invoke(initial_state)

    print(SEPARATOR)
    print(f"FINAL OUTPUT (after {final_state['revision_count']} critic review(s)):")
    print(SEPARATOR)
    print(final_state["final"])
    return final_state


run_reflection(
    "Write a concise 3-paragraph explanation of RAG (Retrieval-Augmented Generation) "
    "suitable for a software engineer who is new to AI. "
    "Must include: what problem it solves, how it works, and when to use it."
)

print("\n\nDone! Key insights:")
print("1. Reflection = generating + critiquing + revising in a loop")
print("2. The generator and critic can have different temperature settings")
print("3. Always add a max_revisions guard — without it the loop could run forever")
print("4. This same pattern applies to code generation, report writing, planning, etc.")
