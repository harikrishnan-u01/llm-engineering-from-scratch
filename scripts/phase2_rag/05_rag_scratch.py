"""
Phase 2 | Script 05 — RAG from Scratch
=======================================
Goal: Build every part of a RAG pipeline manually — no frameworks.

What you'll learn:
- Document chunking (fixed-size with overlap)
- Embedding every chunk
- In-memory vector search
- Injecting retrieved context into an LLM prompt
- The difference between answering with vs without retrieval

Run: python scripts/phase2_rag/05_rag_scratch.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from src.ollama_client import embed, generate

SEPARATOR = "\n" + "─" * 60 + "\n"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/sample_docs")


# ── Step 1: Load and chunk a document ────────────────────────────────────
def load_text(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # skip tiny tail chunks


print("=" * 60)
print("STEP 1: Load and chunk documents")
print("=" * 60)

docs = []
for filename in ["intro_to_llms.txt", "rag_overview.txt"]:
    text = load_text(os.path.join(DATA_DIR, filename))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    docs.extend(chunks)
    print(f"  {filename}: {len(text)} chars → {len(chunks)} chunks")

print(f"\nTotal chunks: {len(docs)}")
print(f"\nSample chunk:\n{docs[0][:300]}...")


# ── Step 2: Embed all chunks ──────────────────────────────────────────────
print(SEPARATOR)
print("STEP 2: Embed every chunk (this takes a moment...)")
print("=" * 60)

chunk_embeddings = []
for i, chunk in enumerate(docs):
    vec = embed(chunk)
    chunk_embeddings.append(np.array(vec))
    if (i + 1) % 5 == 0:
        print(f"  Embedded {i+1}/{len(docs)} chunks...")

print(f"\nEmbedding shape: {len(chunk_embeddings)} vectors × {len(chunk_embeddings[0])} dims")


# ── Step 3: Similarity search ─────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Find the top_k most relevant chunks for a query."""
    q_vec = np.array(embed(query))
    scores = [cosine_similarity(q_vec, c_vec) for c_vec in chunk_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in top_indices], [scores[i] for i in top_indices]


# ── Step 4: Full RAG pipeline ─────────────────────────────────────────────
def rag_answer(question: str, top_k: int = 3) -> str:
    retrieved_chunks, scores = retrieve(question, top_k)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Use ONLY the following context to answer the question.
If the context does not contain enough information, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    return generate(prompt, temperature=0.0), retrieved_chunks, scores


print(SEPARATOR)
print("STEP 3 & 4: Retrieve + Generate (full RAG pipeline)")
print("=" * 60)

questions = [
    "What is temperature in the context of LLMs?",
    "What is the difference between RAG and fine-tuning?",
    "What is cosine similarity used for in RAG?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    answer, chunks, scores = rag_answer(question)
    print(f"\nAnswer:\n{answer}")
    print(f"\nRetrieved chunks (top 3 similarity scores):")
    for chunk, score in zip(chunks, scores):
        print(f"  [{score:.3f}] {chunk[:100]}...")
    print(SEPARATOR)


# ── Step 5: Compare with/without RAG ─────────────────────────────────────
print("STEP 5: Compare answers with vs without RAG")
print("=" * 60)

specific_question = "According to the documents, what are the four RAG failure modes?"

print(f"Question: {specific_question}\n")

print("[WITHOUT RAG — LLM answers from memory]")
raw_answer = generate(specific_question, temperature=0.0)
print(raw_answer)

print(SEPARATOR)
print("[WITH RAG — LLM answers from retrieved context]")
rag_result, _, _ = rag_answer(specific_question)
print(rag_result)

print("\nDone! Key insight: RAG is just search + context injection.")
print("The LLM did not change — only the prompt did.")
