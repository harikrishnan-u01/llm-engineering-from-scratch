# Quick Reference — Learning Sequence

## How to Run Any Script

```bash
source venv/bin/activate
python scripts/phase1_llm_basics/01_raw_llm_call.py
```

Or open the script in VS Code and press `▶` (make sure interpreter is set to `venv/bin/python`).

---

## Learning Sequence — One Script Per Session

| # | Script | What you learn |
|---|--------|----------------|
| 01 | `scripts/phase1_llm_basics/01_raw_llm_call.py` | LLM = HTTP endpoint, tokens, streaming |
| 02 | `scripts/phase1_llm_basics/02_prompting.py` | System prompts, few-shot, temperature, chain-of-thought |
| 03 | `scripts/phase1_llm_basics/03_langchain_llm.py` | LangChain wraps what you did in 01/02 |
| 04 | `scripts/phase2_rag/04_embeddings.py` | Vectors, cosine similarity, semantic search |
| 05 | `scripts/phase2_rag/05_rag_scratch.py` | RAG from scratch — no framework |
| 06 | `scripts/phase2_rag/06_rag_langchain.py` | Same RAG via LangChain + ChromaDB |
| 07 | `scripts/phase3_agents/07_react_scratch.py` | Agent loop from scratch — Thought/Action/Observation |
| 08 | `scripts/phase3_agents/08_langgraph_agent.py` | Same agent as a LangGraph graph |
| 09 | `scripts/phase4_multiagent/09_planner_executor.py` | Two-agent workflow: planner + executor |
| 10 | `scripts/phase4_multiagent/10_reflection_loop.py` | Self-critique loop: generator + critic |

---

## Notes

> **Run script 06 before 07** — script 06 builds the ChromaDB index that the `search_docs` tool in scripts 07–10 uses.

> **Ollama must be running** before executing any script. Open Ollama.app from Applications (look for the icon in the menu bar), or run `ollama serve` in a terminal.

> **Models needed:** `llama3.2` and `nomic-embed-text`. Verify with `ollama list`.
