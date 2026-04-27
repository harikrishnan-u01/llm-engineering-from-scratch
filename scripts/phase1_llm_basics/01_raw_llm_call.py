"""
Phase 1 | Script 01 — Raw LLM Call
===================================
Goal: Talk to Ollama directly using the requests library.
No frameworks. Just HTTP POST and JSON.

What you'll learn:
- An LLM is a stateless HTTP endpoint
- What a raw API response looks like (tokens, timing, model info)
- The difference between stream=False (wait for full answer) and stream=True (get tokens live)

Run: python scripts/phase1_llm_basics/01_raw_llm_call.py
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"


def call_llm(prompt: str, stream: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    response.raise_for_status()
    return response.json()


def call_llm_streaming(prompt: str):
    """Streams tokens and prints them as they arrive."""
    payload = {"model": MODEL, "prompt": prompt, "stream": True}
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)
    response.raise_for_status()

    print("Streaming response: ", end="")
    full_text = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_text += token
            if chunk.get("done"):
                print()  # newline when done
                return full_text, chunk  # last chunk has timing stats


# ── Step 1: Simple call, inspect the raw response ──────────────────────────
print("=" * 60)
print("STEP 1: Raw response object")
print("=" * 60)

result = call_llm("What is 2 + 2? Answer in one sentence.")
print(json.dumps(result, indent=2))

print("\nKey fields:")
print(f"  model        : {result['model']}")
print(f"  response     : {result['response'].strip()}")
print(f"  done         : {result['done']}")
print(f"  prompt_tokens: {result.get('prompt_eval_count', 'n/a')}")
print(f"  output_tokens: {result.get('eval_count', 'n/a')}")
total_ns = result.get("total_duration", 0)
print(f"  total_time   : {total_ns / 1e9:.2f}s")


# ── Step 2: Streaming response ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Streaming tokens live")
print("=" * 60)

full_text, final_chunk = call_llm_streaming(
    "Name three benefits of exercise. Be brief."
)
print(f"\nFull answer captured: {len(full_text)} characters")


# ── Step 3: List available models ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Models available in Ollama")
print("=" * 60)

resp = requests.get(f"{OLLAMA_URL}/api/tags")
models = resp.json().get("models", [])
for m in models:
    size_gb = m.get("size", 0) / 1e9
    print(f"  {m['name']:<30} {size_gb:.1f} GB")

print("\nDone! Key insight: an LLM is just an HTTP endpoint that takes text and returns text.")
