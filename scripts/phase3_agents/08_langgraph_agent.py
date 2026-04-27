"""
Phase 3 | Script 08 — LangGraph Agent
=======================================
Goal: Rebuild the same ReAct agent from script 07 as a LangGraph state graph.

What you'll learn:
- LangGraph core concepts: State, Nodes, Edges, Conditional routing
- How the same agent loop becomes explicit and visual as a graph
- Why LangGraph is better than the hand-rolled loop for real projects
  (debuggable, extensible, supports human-in-the-loop, memory, etc.)

Run: python scripts/phase3_agents/08_langgraph_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import Annotated, Optional, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.tools import calculator as _calc, get_date as _get_date, search_docs as _search

SEPARATOR = "\n" + "─" * 60 + "\n"
MODEL = "llama3.2"


# ── Step 1: Define tools as LangChain @tool decorated functions ───────────
print("=" * 60)
print("STEP 1: Define tools with the @tool decorator")
print("=" * 60)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example input: '15 * 47.50 / 100'"""
    return _calc(expression)

@tool
def get_date(query: Optional[str] = None) -> str:
    """Get today's date, weekday number, and days until next Monday. The query input is ignored."""
    return _get_date(query or "")

@tool
def search_docs(query: str) -> str:
    """Search the local document knowledge base. Input: a search query."""
    return _search(query)

tools = [calculator, get_date, search_docs]
print(f"Registered tools: {[t.name for t in tools]}")


# ── Step 2: Define the agent state ────────────────────────────────────────
print(SEPARATOR)
print("STEP 2: AgentState — the shared state that flows through the graph")
print("=" * 60)

class AgentState(TypedDict):
    # add_messages is a reducer: new messages are appended, not replaced
    messages: Annotated[list, add_messages]

print("AgentState has one field: 'messages' (a list that grows with each step)")
print("Every node reads from and writes to this shared state.")


# ── Step 3: Define graph nodes ────────────────────────────────────────────
print(SEPARATOR)
print("STEP 3: Graph nodes — reason_node and tool_node")
print("=" * 60)

llm = ChatOllama(model=MODEL, temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

def reason_node(state: AgentState) -> dict:
    """Call the LLM. It either picks a tool or produces a final answer."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

print("reason_node: calls the LLM → decides whether to use a tool or answer")
print("tool_node  : executes whichever tool the LLM chose")


# ── Step 4: Define routing logic ──────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Route: if the last message has tool calls → tool_node, else → END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


# ── Step 5: Build and compile the graph ───────────────────────────────────
print(SEPARATOR)
print("STEP 4 & 5: Build the graph and compile it")
print("=" * 60)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("reason_node", reason_node)
graph_builder.add_node("tool_node", tool_node)

graph_builder.set_entry_point("reason_node")
graph_builder.add_conditional_edges("reason_node", should_continue)
graph_builder.add_edge("tool_node", "reason_node")  # always return to reason after tool

agent = graph_builder.compile()
print("Graph compiled successfully.")
print("\nGraph structure:")
print("  START → reason_node")
print("  reason_node → [tool_node | END]  (conditional)")
print("  tool_node → reason_node")


# ── Step 6: Run the agent ─────────────────────────────────────────────────
def run_graph_agent(goal: str):
    print(f"\nGoal: {goal}")
    print(SEPARATOR)

    system = SystemMessage(content=(
        "You are a helpful assistant. Use tools when needed to answer accurately. "
        "Be concise in your final answer."
    ))
    initial_state = {"messages": [system, HumanMessage(content=goal)]}

    step = 0
    for event in agent.stream(initial_state):
        for node_name, node_output in event.items():
            step += 1
            print(f"[Step {step} — {node_name}]")
            for msg in node_output.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  → Tool call: {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  → Tool result: {msg.content[:150]}")
                elif hasattr(msg, "content") and msg.content:
                    print(f"  → {msg.content[:300]}")
            print()

print(SEPARATOR)
print("RUNNING TASKS")
print(SEPARATOR)

print("TASK 1: Math")
run_graph_agent("What is 15% tip on a $47.50 dinner? What is the total?")

print("\n\n")
print("TASK 2: Date")
run_graph_agent("What is today's date?")

print("\n\nDone! Compare this to script 07.")
print("Same agent behaviour — but now the loop is an explicit graph:")
print("  - Each step is a named node you can inspect")
print("  - Routing is a separate function, not buried in a while-loop")
print("  - You can add checkpointing, human-in-the-loop, or memory with a few lines")
