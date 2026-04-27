"""
Phase 2 | Script 06 — RAG with LangChain + ChromaDB
=====================================================
Goal: Rebuild the same RAG pipeline from script 05 using LangChain abstractions.
See how the framework simplifies the code while doing the same thing underneath.

What you'll learn:
- RecursiveCharacterTextSplitter: smarter splitting than fixed-size
- OllamaEmbeddings + Chroma: persistent vector store (survives restarts)
- RetrievalQA chain: full RAG in ~5 lines
- ConversationalRetrievalChain: RAG with follow-up question memory
- How to inspect what was retrieved

Run: python scripts/phase2_rag/06_rag_langchain.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SEPARATOR = "\n" + "─" * 60 + "\n"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/sample_docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../../data/chroma_db")
MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


# ── Step 1: Load documents ────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load documents with DirectoryLoader")
print("=" * 60)

loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} documents")
for doc in raw_docs:
    print(f"  {doc.metadata['source']} — {len(doc.page_content)} chars")


# ── Step 2: Smart chunking ────────────────────────────────────────────────
print(SEPARATOR)
print("STEP 2: RecursiveCharacterTextSplitter — smarter than fixed-size")
print("=" * 60)
print("Splits by paragraph → sentence → character, preserving natural boundaries.\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"  {len(raw_docs)} documents → {len(chunks)} chunks")
print(f"\nSample chunk (metadata + content):")
print(f"  Source: {chunks[0].metadata.get('source', 'unknown')}")
print(f"  Content: {chunks[0].page_content[:200]}...")


# ── Step 3: Build persistent vector store ────────────────────────────────
print(SEPARATOR)
print("STEP 3: Embed and store in ChromaDB (persists to disk)")
print("=" * 60)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB at {CHROMA_DIR}")
print("This index persists — you won't need to re-embed next time.")


# ── Step 4: LCEL retrieval chain (replaces deprecated RetrievalQA) ────────
print(SEPARATOR)
print("STEP 4: LCEL retrieval chain — full RAG pipeline in a few lines")
print("=" * 60)

llm = ChatOllama(model=MODEL, temperature=0.0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template(
    "Use only the provided context to answer. Be concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# LCEL pipe: retrieve → format → prompt → LLM → parse
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "What is temperature in the context of LLMs?",
    "What are the four RAG failure modes mentioned in the documents?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    answer = qa_chain.invoke(question)
    print(f"\nAnswer:\n{answer}")
    source_docs = retriever.invoke(question)
    print(f"\nSources retrieved:")
    for doc in source_docs:
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  [{src}] {doc.page_content[:120]}...")
    print(SEPARATOR)


# ── Step 5: Conversational RAG with manual history ────────────────────────
print("STEP 5: Conversational RAG — follow-up questions with message history")
print("=" * 60)
print("Manually track HumanMessage/AIMessage history; no memory object needed.\n")

chat_history: list = []

convo_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use only the provided context to answer. Be concise.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# chat_history is captured by reference — list grows each turn
convo_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | convo_prompt
    | llm
    | StrOutputParser()
)

conversation = [
    "What is RAG?",
    "What are its failure modes?",
    "Which one is most common according to the document?",
]

for turn, question in enumerate(conversation, 1):
    print(f"Turn {turn} — User: {question}")
    answer = convo_chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    print(f"Answer: {answer}\n")

print("Done! Key insight: LangChain 1.x is LCEL-only — chains are |  pipes, memory is a plain list.")
print("But script 05 showed you exactly what LangChain is doing under the hood.")
