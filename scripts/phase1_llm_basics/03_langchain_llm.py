"""
Phase 1 | Script 03 — LangChain LLM
=====================================
Goal: See how LangChain wraps the same raw API calls you did in 01 and 02.

What you'll learn:
- ChatOllama: LangChain's wrapper for Ollama chat models
- ChatPromptTemplate: reusable, parameterised prompt templates
- StrOutputParser: extract plain text from LLM response objects
- LCEL (LangChain Expression Language): chain components with the | operator
- The pattern: prompt | llm | parser

Run: python scripts/phase1_llm_basics/03_langchain_llm.py
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SEPARATOR = "\n" + "─" * 60 + "\n"
MODEL = "llama3.2"

# ── Step 1: Basic ChatOllama call ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: ChatOllama — LangChain wrapper for Ollama")
print("=" * 60)

llm = ChatOllama(model=MODEL, temperature=0.0)

# .invoke() takes a string or list of messages
response = llm.invoke("What is the capital of France? One word answer.")
print(f"Response type : {type(response)}")
print(f"Response      : {response}")
print(f"Content only  : {response.content}")


# ── Step 2: StrOutputParser ────────────────────────────────────────────────
print(SEPARATOR)
print("STEP 2: StrOutputParser — get a plain string back")
print("=" * 60)

parser = StrOutputParser()
text = parser.invoke(response)
print(f"After parser  : {repr(text)}")


# ── Step 3: ChatPromptTemplate ─────────────────────────────────────────────
print(SEPARATOR)
print("STEP 3: ChatPromptTemplate — reusable parameterised prompts")
print("=" * 60)

# Define a template with placeholders
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer concisely in 2-3 sentences."),
    ("human", "{question}"),
])

# Fill in the placeholders
filled_prompt = template.invoke({"domain": "astronomy", "question": "Why is the sky blue at day but red at sunset?"})
print("Filled prompt messages:")
for msg in filled_prompt.messages:
    print(f"  [{msg.type}] {msg.content}")


# ── Step 4: LCEL chain with | operator ────────────────────────────────────
print(SEPARATOR)
print("STEP 4: LCEL — chain components with the pipe | operator")
print("=" * 60)
print("Pattern: prompt | llm | parser\n")

chain = template | llm | StrOutputParser()

# Run the chain with different inputs
topics = [
    {"domain": "cooking", "question": "Why do onions make you cry?"},
    {"domain": "history", "question": "Why did the Roman Empire fall?"},
]

for inputs in topics:
    print(f"Q [{inputs['domain']}]: {inputs['question']}")
    answer = chain.invoke(inputs)
    print(f"A: {answer}")
    print()


# ── Step 5: Batch processing ──────────────────────────────────────────────
print(SEPARATOR)
print("STEP 5: Batch — run a chain on multiple inputs at once")
print("=" * 60)

summarise_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarise the following text in exactly one sentence."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

texts = [
    {"text": "The sun is a star at the center of our solar system. It provides the energy that drives life on Earth through photosynthesis and warmth."},
    {"text": "Python is a high-level programming language known for its readable syntax. It is widely used in data science, web development, and automation."},
]

results = summarise_chain.batch(texts)
for i, summary in enumerate(results):
    print(f"  Summary {i+1}: {summary}")

print("\nDone! Key insight: LangChain is organised wrappers around the same HTTP calls you made in script 01.")
print("The | operator chains: input → prompt formatting → LLM → output parsing.")
