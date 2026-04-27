"""
Phase 1 | Script 02 — Prompting Techniques
===========================================
Goal: Understand how prompt structure changes model output.

What you'll learn:
- System prompts: set the model's persona/rules
- Zero-shot vs few-shot prompting
- Temperature: how randomness affects output
- Chain-of-thought: ask the model to reason step by step

Run: python scripts/phase1_llm_basics/02_prompting.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ollama_client import generate

SEPARATOR = "\n" + "─" * 60 + "\n"


# ── Step 1: System prompt ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: System Prompt — controlling the model's persona")
print("=" * 60)

# Without system prompt
print("\n[No system prompt]")
answer = generate("What is recursion?", temperature=0.0)
print(answer)

# With a system prompt
print(SEPARATOR)
print("[With system prompt: explain like I'm 10]")
answer = generate(
    prompt="What is recursion?",
    system="You are a friendly teacher explaining concepts to a 10-year-old. Use simple words and a fun analogy.",
    temperature=0.0,
)
print(answer)


# ── Step 2: Zero-shot vs Few-shot ──────────────────────────────────────────
print(SEPARATOR)
print("STEP 2: Zero-shot vs Few-shot Prompting")
print("=" * 60)

# Zero-shot: no examples
print("\n[Zero-shot] Classify sentiment:")
zero_shot = generate(
    "Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n'The battery life is great but the screen is dim.'",
    temperature=0.0,
)
print(zero_shot)

# Few-shot: show examples before the real question
print(SEPARATOR)
print("[Few-shot] Classify sentiment with examples:")
few_shot_prompt = """Classify the sentiment as Positive, Negative, or Neutral.

Review: "I love this product, it works perfectly!"
Sentiment: Positive

Review: "Terrible quality, broke after one day."
Sentiment: Negative

Review: "It arrived on time."
Sentiment: Neutral

Review: "The battery life is great but the screen is dim."
Sentiment:"""

answer = generate(few_shot_prompt, temperature=0.0)
print(answer)


# ── Step 3: Temperature effect ────────────────────────────────────────────
print(SEPARATOR)
print("STEP 3: Temperature — controlling randomness")
print("=" * 60)
print("Same prompt, three temperatures. Watch how output varies.\n")

prompt = "Give me a creative name for a coffee shop. Just the name, nothing else."

for temp in [0.0, 0.7, 1.5]:
    answer = generate(prompt, temperature=temp)
    print(f"  temperature={temp}: {answer.strip()}")


# ── Step 4: Chain-of-thought ──────────────────────────────────────────────
print(SEPARATOR)
print("STEP 4: Chain-of-Thought — making the model reason step by step")
print("=" * 60)

problem = "A store sells apples for $0.50 each and oranges for $0.75 each. If Sarah buys 4 apples and 3 oranges, and pays with a $10 bill, how much change does she get?"

print("[Without chain-of-thought]")
answer = generate(problem, temperature=0.0)
print(answer)

print(SEPARATOR)
print("[With chain-of-thought: 'Let's think step by step']")
answer = generate(problem + "\n\nLet's think step by step.", temperature=0.0)
print(answer)

print("\nDone! Key insight: the prompt IS the program. Small changes → big output differences.")
