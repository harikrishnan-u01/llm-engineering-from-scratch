# AI Learning App — Plan

## Context

Learn LLMs, RAG, Agents, and Agentic workflows by writing Python scripts in VS Code.
No Jupyter, no Streamlit, no CrewAI, no LlamaIndex — keep it simple.
Stack: Ollama (local LLM) + LangChain + LangGraph + ChromaDB.
Each phase = a folder of runnable `.py` scripts. Run them from VS Code or terminal.

---

## Step 1 — Install Packages

```bash
source /Users/hari/Documents/Projects/AI/ai-learning-app/venv/bin/activate

pip install \
  requests \
  langchain langchain-community langchain-ollama \
  langgraph \
  chromadb \
  pypdf \
  python-dotenv
```

VS Code setup:
- Open the `ai-learning-app` folder in VS Code
- Select the venv interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `venv/bin/python`
- Run any script: open it, press `▶` or `Ctrl+F5`, or use the terminal: `python scripts/phase1_rag/01_raw_llm_call.py`

---

## Step 2 — Folder Structure

```
ai-learning-app/
├── venv/
├── data/
│   ├── sample_docs/
│   │   ├── intro_to_llms.txt       # sample text for RAG experiments
│   │   └── rag_overview.txt
│   └── chroma_db/                  # persisted vector store (created at runtime)
├── scripts/
│   ├── phase1_llm_basics/
│   │   ├── 01_raw_llm_call.py      # call Ollama REST API directly with requests
│   │   ├── 02_prompting.py         # system prompts, few-shot, temperature
│   │   └── 03_langchain_llm.py     # same calls via ChatOllama
│   ├── phase2_rag/
│   │   ├── 04_embeddings.py        # embed text, compute cosine similarity manually
│   │   ├── 05_rag_scratch.py       # chunk → embed → retrieve → generate, no framework
│   │   └── 06_rag_langchain.py     # same pipeline via LangChain + ChromaDB
│   ├── phase3_agents/
│   │   ├── 07_react_scratch.py     # hand-rolled ReAct loop: Thought/Action/Observation
│   │   └── 08_langgraph_agent.py   # same agent as a LangGraph graph with state
│   └── phase4_multiagent/
│       ├── 09_planner_executor.py  # Planner node + Executor node in LangGraph
│       └── 10_reflection_loop.py   # add a Critic/Reflection node that can revise output
├── src/
│   ├── __init__.py
│   ├── ollama_client.py            # thin wrapper: generate(), embed(), list_models()
│   └── tools.py                    # reusable tool functions agents can call
├── requirements.txt
└── .env                            # OLLAMA_BASE_URL=http://localhost:11434
```

---

## Phase 1 — LLM Basics (scripts 01–03)

**Goal:** Understand what an LLM actually does before any framework.

### `01_raw_llm_call.py`
- `POST http://localhost:11434/api/generate` with `requests`
- Print the raw JSON response — see tokens, timing, model info
- Run it: understand the LLM is just a stateless HTTP endpoint

### `02_prompting.py`
- System prompt vs user prompt
- Zero-shot vs few-shot examples in the prompt
- Temperature: run the same prompt at 0.0, 0.7, 1.5 — print all three outputs
- Streaming: use `stream=True`, print tokens as they arrive

### `03_langchain_llm.py`
- `ChatOllama` from `langchain_ollama`
- `ChatPromptTemplate` — define reusable prompt templates
- `StrOutputParser` — get plain string from response
- Chain them with LCEL: `prompt | llm | parser`
- Key insight: LangChain is just structured wrappers around what you did in 01/02

---

## Phase 2 — RAG (scripts 04–06)

**Goal:** Build retrieval-augmented generation from scratch, then see how LangChain simplifies it.

### `04_embeddings.py`
- `POST http://localhost:11434/api/embed` with `nomic-embed-text`
- Print the vector dimension (768 floats)
- Compute cosine similarity between two sentences manually with `numpy`
- Try similar sentences vs unrelated ones — see similarity scores differ

### `05_rag_scratch.py`
- Load `data/sample_docs/intro_to_llms.txt`
- Split into fixed-size chunks (e.g. 500 chars, 50 overlap) — plain Python
- Embed every chunk with `src/ollama_client.embed()`
- Store chunks + vectors in a Python list
- At query time: embed the question, find top-3 chunks by cosine similarity
- Build a prompt: "Use this context: {chunks}\n\nAnswer: {question}"
- Send to `llama3.2`, print the answer
- Key insight: RAG is just search + context injection. No magic.

### `06_rag_langchain.py`
- `RecursiveCharacterTextSplitter` — smarter splitting than fixed-size
- `OllamaEmbeddings` + `Chroma` — persistent vector store on disk
- `RetrievalQA.from_chain_type()` — full pipeline in ~5 lines
- Print retrieved source chunks alongside the answer
- Experiment: ask the same question with/without retrieval, compare answers

---

## Phase 3 — Agents (scripts 07–08)

**Goal:** Understand what an agent is — an LLM in a loop with access to tools.

### `07_react_scratch.py`
- Define 2–3 tools as plain Python functions in `src/tools.py`:
  - `calculator(expression: str) -> str`
  - `get_date() -> str`
  - `search_docs(query: str) -> str` (wraps the ChromaDB retriever from phase 2)
- Implement the ReAct loop manually:
  1. Send goal to LLM with a prompt that says "you can use tools"
  2. Parse LLM output for `Thought:`, `Action:`, `Action Input:`
  3. Execute the tool, format result as `Observation:`
  4. Feed back into LLM, repeat until `Final Answer:`
- Key insight: an agent is just a while-loop around an LLM with string parsing

### `08_langgraph_agent.py`
- Rebuild the same agent as a LangGraph state graph:
  - `AgentState` TypedDict: messages, next step
  - `reason_node`: calls the LLM
  - `tool_node`: executes whichever tool was chosen
  - Conditional edge: if LLM output has a tool call → tool_node, else → END
- Run the same goal as 07, compare the two approaches
- Key insight: LangGraph makes the loop explicit, debuggable, and extensible

---

## Phase 4 — Agentic Workflows (scripts 09–10)

**Goal:** Chain multiple specialised agents to solve problems that need planning + execution.

### `09_planner_executor.py`
- Two LangGraph nodes:
  - `planner_node`: takes a goal, returns a numbered list of steps
  - `executor_node`: takes one step at a time, has tools, returns result
- Graph: `planner → executor (loop over steps) → done`
- Demo goal: "Research what RAG is and write a 3-paragraph summary"
  - Planner breaks it into: search, draft paragraph 1, draft paragraph 2, draft paragraph 3
  - Executor runs each with the doc search tool

### `10_reflection_loop.py`
- Extend script 09 with a `critic_node`:
  - Reads the executor's output
  - Decides: "good enough" → END, or "needs revision" → back to executor
- Add a `max_revisions` guard to prevent infinite loops
- Key insight: reflection/self-critique is how agentic systems improve their own output

---

## `src/` Helpers

### `src/ollama_client.py`
```python
# generate(prompt, model, system, temperature, stream) -> str
# embed(text, model) -> list[float]
# list_models() -> list[str]
# Base URL from .env, defaults to http://localhost:11434
```

### `src/tools.py`
```python
# calculator(expression: str) -> str   — uses eval() safely
# get_date() -> str                    — returns today's date
# search_docs(query: str) -> str       — queries ChromaDB, returns top chunks as text
```

---

## Optional Later — Streamlit UI

Once all 10 scripts work, build `app/app.py` with 3 tabs:
- **RAG Explorer** — upload a doc, ask questions, see retrieved chunks
- **Agent** — give a task, watch step-by-step Thought/Action/Observation trace
- **Agentic Workflow** — give a goal, watch planner + executor + critic work

---

## Verification Checklist

1. `ollama list` — shows `llama3.2` and `nomic-embed-text`
2. `python scripts/phase1_llm_basics/01_raw_llm_call.py` — prints LLM response
3. `python scripts/phase2_rag/05_rag_scratch.py` — answers a question from the doc
4. `python scripts/phase3_agents/07_react_scratch.py` — completes a multi-step task
5. `python scripts/phase4_multiagent/09_planner_executor.py` — multi-node graph runs end to end

---

## Learning Sequence

| Phase | Scripts | Core Concept | Framework Used |
|-------|---------|-------------|----------------|
| 1 | 01–03 | LLM = HTTP endpoint, prompting, templates | `requests` → LangChain |
| 2 | 04–06 | RAG = search + context injection | numpy → LangChain + ChromaDB |
| 3 | 07–08 | Agent = LLM in a loop with tools | scratch → LangGraph |
| 4 | 09–10 | Agentic = planner + executor + reflection | LangGraph |

One script per session. Finish phases 1–2 in a weekend. Full agentic system in a week.
