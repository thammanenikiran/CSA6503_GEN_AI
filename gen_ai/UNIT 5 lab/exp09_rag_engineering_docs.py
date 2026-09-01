"""
Unit 5 - Experiment 9: Local Retrieval-Augmented Generation (RAG) over Engineering Docs
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Implement a local RAG system using engineering documents, a vector database
    (ChromaDB) and Ollama to answer technical questions. Instead of relying on
    the model's memory, we retrieve the most relevant document chunks and let the
    LLM answer strictly from them - this grounds the answer and reduces
    hallucination.

PIPELINE:
    load docs -> split into chunks -> embed (Ollama nomic-embed-text)
    -> store in ChromaDB -> retrieve top-k for a question -> answer with llama3.2

SETUP:
    pip install ollama chromadb
    ollama serve
    ollama pull llama3.2
    ollama pull nomic-embed-text        # embedding model used here

RUN:
    python exp09_rag_engineering_docs.py
"""

import os
import sys
import glob

import ollama
import chromadb
from chromadb.config import Settings

MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def chunk_text(text: str, max_chars: int = 500) -> list:
    """Split text into chunks of a few paragraphs, up to max_chars each."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks, cur = [], ""
    for b in blocks:
        if len(cur) + len(b) + 2 <= max_chars:
            cur = (cur + "\n\n" + b).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = b
    if cur:
        chunks.append(cur)
    return chunks


def embed(text: str) -> list:
    """Get an embedding vector for a piece of text from Ollama."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def build_index():
    """Read every doc, chunk it, embed it and store it in ChromaDB."""
    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")) +
                   glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not files:
        sys.exit(f"No documents found in {DOCS_DIR}. Add some .md/.txt files.")

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("engineering_docs")

    ids, docs, metas, embs = [], [], [], []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        source = os.path.basename(path)
        for j, ch in enumerate(chunk_text(text)):
            ids.append(f"{source}#{j}")
            docs.append(ch)
            metas.append({"source": source})
            embs.append(embed(ch))

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"Indexed {len(ids)} chunks from {len(files)} documents.\n")
    return col


def answer(col, question: str, k: int = 3):
    """Retrieve the top-k chunks and answer the question from them only."""
    q_emb = embed(question)
    res = col.query(query_embeddings=[q_emb], n_results=k)
    contexts = res["documents"][0]
    sources = [m["source"] for m in res["metadatas"][0]]

    context_block = "\n\n".join(f"[{s}]\n{c}" for s, c in zip(sources, contexts))
    messages = [
        {"role": "system", "content": (
            "You are a technical assistant. Answer the question using ONLY the "
            "context below. If the context does not contain the answer, say you "
            "do not have that information. Cite the source file name in square "
            "brackets."
        )},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.2})
    return resp["message"]["content"].strip(), sources


def main():
    print("=" * 72)
    print(" Local RAG over engineering documents (ChromaDB + Ollama)")
    print("=" * 72)
    col = build_index()

    demo_questions = [
        "What is the recommended hydraulic oil grade for the HP-50 press?",
        "How often should the centrifugal pump mechanical seal be replaced?",
        "What are the common causes of an induction motor overheating?",
    ]
    for q in demo_questions:
        ans, sources = answer(col, q)
        print(f"Q: {q}")
        print(f"A: {ans}")
        print(f"   (retrieved from: {', '.join(sources)})\n")

    print("-" * 72)
    print("Ask your own technical questions ('exit' to quit).")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            ans, sources = answer(col, q)
            print("A:", ans)
            print("   (retrieved from:", ", ".join(sources) + ")")
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"\n[Error] {e}\n"
                 f"Make sure 'ollama serve' is running and both models are pulled:\n"
                 f"  ollama pull {MODEL}\n  ollama pull {EMBED_MODEL}")
