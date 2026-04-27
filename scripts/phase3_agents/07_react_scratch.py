"""
Phase 3 | Script 07 — ReAct Agent from Scratch
===============================================
Goal: Build an agent loop manually to understand what "agent" really means.

What you'll learn:
- The ReAct pattern: Thought → Action → Observation → repeat
- An agent is just a while-loop around an LLM with string parsing
- How tools are defined and called
- How to handle the loop termination (Final Answer)

Run: python scripts/phase3_agents/07_react_scratch.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import re
from src.ollama_client import generate
from src.tools import calculator, get_date, search_docs

SEPARATOR = "\n" + "─" * 60 + "\n"

# ── Tool registry ─────────────────────────────────────────────────────────
TOOLS = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a mathematical expression. Input: a math expression like '15 * 47.50 / 100'.",
    },
    "get_date": {
        "fn": get_date,
        "description": "Get today's date, weekday number (1=Monday…7=Sunday), and days until next Monday. Input: anything (it's ignored). Use this for any date or calendar question — it gives you everything you need to answer without extra calculation.",
    },
    "search_docs": {
        "fn": search_docs,
        "description": "Search an AI/ML educational knowledge base about topics like LLMs, RAG, tokens, embeddings, and agents. Do NOT use for dates, math, or general knowledge outside AI/ML.",
    },
}


def build_system_prompt() -> str:
    tool_descriptions = "\n".join(
        f"- {name}: {info['description']}" for name, info in TOOLS.items()
    )
    return f"""You are a helpful assistant that solves tasks step by step using tools.

You have access to these tools:
{tool_descriptions}

Use this EXACT format for every step:

Thought: <what you're thinking>
Action: <tool name — must be one of: {', '.join(TOOLS.keys())}>
Action Input: <the input to the tool>

When you have enough information to give the final answer, use:

Thought: <final reasoning>
Final Answer: <your complete answer>

IMPORTANT: Only output one Thought/Action/Action Input block at a time. Wait for the Observation before continuing.
IMPORTANT: Never repeat the exact same Action + Action Input you have already tried. If a tool returned a result, use that result to reason toward a Final Answer.
IMPORTANT: When you are ready to answer, you MUST use the exact "Final Answer:" format. Never use "Action: None" or "Action: N/A" — those are invalid."""


def parse_llm_output(text: str) -> dict:
    """Extract the action or final answer from LLM output."""
    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    # Check for Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

    if action_match and input_match:
        return {
            "type": "action",
            "tool": action_match.group(1).strip(),
            "input": input_match.group(1).strip(),
        }

    return {"type": "unknown", "content": text}


FORMAT_CORRECTION = (
    "Your response did not follow the required format. You must use exactly one of:\n"
    "  Thought: ...\n  Action: <tool>\n  Action Input: <input>\n"
    "OR:\n"
    "  Thought: ...\n  Final Answer: <answer>\n"
    "Never use 'Action: None'. If you have enough information, write a Final Answer now."
)


def _force_conclusion(goal: str, scratchpad: str, system: str) -> str:
    """Make a targeted LLM call to extract a Final Answer from what's already in the scratchpad."""
    prompt = (
        f"{scratchpad}\n"
        f"You have enough information to answer the task. Do NOT call any tools.\n"
        f"Task: {goal}\n\n"
        f"Respond with exactly this format:\n"
        f"Thought: <brief reasoning from the observations above>\n"
        f"Final Answer: <your complete answer>"
    )
    output = generate(prompt, system=system, temperature=0.0)
    parsed = parse_llm_output(output)
    if parsed["type"] == "final":
        return parsed["content"]
    # Fallback: return whatever the LLM said
    return output.strip()


def run_agent(goal: str, max_steps: int = 10) -> str:
    """Run the ReAct agent loop."""
    system = build_system_prompt()
    scratchpad = f"Task: {goal}\n\n"
    seen_actions: set = set()
    tool_call_counts: dict = {}
    format_errors: int = 0
    blocked_streak: int = 0

    print(f"Goal: {goal}")
    print(SEPARATOR)

    for step in range(1, max_steps + 1):
        print(f"[Step {step}]")

        # Ask the LLM what to do next
        prompt = scratchpad + "What is your next step?"
        llm_output = generate(prompt, system=system, temperature=0.0)

        print(llm_output.strip())

        parsed = parse_llm_output(llm_output)

        if parsed["type"] == "final":
            print(SEPARATOR)
            print(f"FINAL ANSWER: {parsed['content']}")
            return parsed["content"]

        elif parsed["type"] == "action":
            tool_name = parsed["tool"]
            tool_input = parsed["input"]

            action_key = (tool_name, tool_input.strip("'\""))
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            blocked = False

            if tool_name.lower() in ("none", "n/a", "na"):
                observation = (
                    "Invalid action. When you are ready to answer, use the Final Answer: format — "
                    "do not use 'Action: None'. Write your complete answer after 'Final Answer:'."
                )
                blocked = True
            elif tool_call_counts[tool_name] > 3:
                observation = (
                    f"You have called '{tool_name}' more than 3 times. "
                    "Stop using tools and write a Final Answer based on what you already found."
                )
                blocked = True
            elif action_key in seen_actions:
                observation = (
                    "You already called this tool with this exact input. "
                    "Use what you already found to write a Final Answer."
                )
                blocked = True
            elif tool_name not in TOOLS:
                observation = f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
            else:
                seen_actions.add(action_key)
                try:
                    observation = TOOLS[tool_name]["fn"](tool_input)
                except Exception as e:
                    observation = f"Tool error: {e}"

            if blocked:
                blocked_streak += 1
            else:
                blocked_streak = 0

            print(f"Observation: {observation}")
            print()

            # Add this step to the scratchpad so the LLM has context next turn
            scratchpad += (
                f"{llm_output.strip()}\n"
                f"Observation: {observation}\n\n"
            )

            # Force a conclusion if the LLM keeps ignoring stop signals
            if blocked_streak >= 2:
                print("[Forcing conclusion — LLM ignored stop signals]")
                answer = _force_conclusion(goal, scratchpad, system)
                print(SEPARATOR)
                print(f"FINAL ANSWER: {answer}")
                return answer

        else:
            format_errors += 1
            if format_errors >= 3:
                print(f"[Too many format errors, stopping]")
                break
            print(f"[Format error #{format_errors} — injecting correction]")
            scratchpad += f"{llm_output.strip()}\nObservation: {FORMAT_CORRECTION}\n\n"

    return "Max steps reached without a final answer."


# ── Run example tasks ─────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Math with the calculator tool")
print("=" * 60)
run_agent("What is 15% tip on a $47.50 dinner? How much is the total including tip?")

print("\n\n" + "=" * 60)
print("TASK 2: Date reasoning")
print("=" * 60)
run_agent("What day of the week is today, and how many days until the next Monday?")

print("\n\n" + "=" * 60)
print("TASK 3: Document search (requires running script 06 first to build the index)")
print("=" * 60)
run_agent("What does the knowledge base say about RAG failure modes?")

print("\n\nDone! Key insight: an agent is just a while-loop.")
print("The 'intelligence' is the prompt format + the LLM's ability to follow it.")
