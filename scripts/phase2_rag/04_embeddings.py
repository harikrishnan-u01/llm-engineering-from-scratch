"""
Phase 2 | Script 04 — Embeddings and Cosine Similarity
=======================================================
Goal: Understand what embeddings are and why they are the foundation of RAG.

What you'll learn:
- What an embedding is (a list of floats representing meaning)
- How to get embeddings from Ollama's nomic-embed-text model
- How to compute cosine similarity manually with numpy
- Why semantically similar sentences have high similarity scores

Run: python scripts/phase2_rag/04_embeddings.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from src.ollama_client import embed

SEPARATOR = "\n" + "─" * 60 + "\n"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors. Range: -1 (opposite) to 1 (identical)."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Step 1: Get a single embedding ────────────────────────────────────────
print("=" * 60)
print("STEP 1: What does an embedding look like?")
print("=" * 60)

sentence = "The cat sat on the mat."
vector = embed(sentence)

print(f"Sentence  : {sentence}")
print(f"Dimensions: {len(vector)}")
print(f"First 8   : {[round(v, 4) for v in vector[:8]]}")
print(f"Min value : {min(vector):.4f}")
print(f"Max value : {max(vector):.4f}")
print("\nThis 768-dimensional vector represents the MEANING of the sentence.")


# ── Step 2: Cosine similarity on similar sentences ─────────────────────────
print(SEPARATOR)
print("STEP 2: Similar sentences → high cosine similarity")
print("=" * 60)

sentence_a = "The cat sat on the mat."
sentence_b = "A feline rested on a rug."  # same meaning, different words
sentence_c = "The stock market crashed yesterday."  # unrelated

print(f"A: {sentence_a}")
print(f"B: {sentence_b}")
print(f"C: {sentence_c}")
print()

vec_a = embed(sentence_a)
vec_b = embed(sentence_b)
vec_c = embed(sentence_c)

sim_ab = cosine_similarity(vec_a, vec_b)
sim_ac = cosine_similarity(vec_a, vec_c)
sim_bc = cosine_similarity(vec_b, vec_c)

print(f"  Similarity(A, B) = {sim_ab:.4f}  ← same meaning, should be HIGH")
print(f"  Similarity(A, C) = {sim_ac:.4f}  ← unrelated, should be LOW")
print(f"  Similarity(B, C) = {sim_bc:.4f}  ← unrelated, should be LOW")


# ── Step 3: Semantic search over a mini dataset ───────────────────────────
print(SEPARATOR)
print("STEP 3: Semantic search — find the most relevant sentence for a query")
print("=" * 60)

facts = [
    "Python is a high-level programming language known for its simple syntax.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are inspired by the human brain.",
    "Water boils at 100 degrees Celsius at sea level.",
    "LLMs learn by predicting the next token in a sequence.",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Transformers use self-attention to process sequences in parallel.",
    "Mount Everest is the tallest mountain on Earth.",
]

print("Indexing facts...")
fact_embeddings = [embed(f) for f in facts]
print(f"Indexed {len(facts)} facts.\n")

queries = [
    "How do language models work?",
    "What is special about the Eiffel Tower?",
    "Tell me about coding languages.",
]

for query in queries:
    query_vec = embed(query)
    similarities = [cosine_similarity(query_vec, fv) for fv in fact_embeddings]
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    print(f"Query : {query}")
    print(f"Match : {facts[best_idx]}")
    print(f"Score : {best_score:.4f}")
    print()


# ── Step 4: Keyword search vs semantic search comparison ──────────────────
print(SEPARATOR)
print("STEP 4: Why keyword search fails where semantic search succeeds")
print("=" * 60)

corpus = [
    "The patient was prescribed antibiotics for the infection.",
    "She takes ibuprofen for her headaches.",
    "The mechanic fixed the car engine.",
    "Python's garbage collector manages memory automatically.",
]

query = "What medication was given to the sick person?"

print(f"Query: {query}")
print("\nKeyword search: looking for 'medication' or 'sick'...")
keyword_matches = [s for s in corpus if "medication" in s.lower() or "sick" in s.lower()]
print(f"  Keyword results: {keyword_matches or 'NO MATCHES'}")

print("\nSemantic search: using embeddings...")
q_vec = embed(query)
corpus_vecs = [embed(s) for s in corpus]
sims = [(cosine_similarity(q_vec, cv), s) for cv, s in zip(corpus_vecs, corpus)]
sims.sort(reverse=True)
print(f"  Top match: {sims[0][1]}")
print(f"  Score    : {sims[0][0]:.4f}")

print("\nDone! Key insight: embeddings capture MEANING, not just words.")
print("Cosine similarity is the measure that makes semantic search possible.")
print("This is the foundation of RAG.")
