"""
Unit 5 - Experiment 10: Local RAG-based Engineering Troubleshooting System
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Implement a local RAG-based troubleshooting system that retrieves relevant
    information from local technical documents and generates step-by-step repair
    recommendations using Ollama. Here embeddings are produced locally by a
    sentence-transformers model (no embedding download from Ollama needed),
    stored in ChromaDB, and the LLM turns the retrieved context into an ordered
    diagnostic procedure.

PIPELINE:
    load docs -> chunk -> embed (sentence-transformers all-MiniLM-L6-v2)
    -> store in ChromaDB (cosine) -> retrieve for a symptom
    -> llama3.2 writes numbered troubleshooting steps grounded in the context

SETUP:
    pip install ollama chromadb sentence-transformers
    ollama serve
    ollama pull llama3.2
    # first run downloads the small MiniLM embedding model (~90 MB) once

RUN:
    python exp10_rag_troubleshooting.py
"""

import os
import sys
import glob

import ollama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2"
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

print("Loading the local embedding model (first run downloads it)...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


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


def embed_batch(texts: list) -> list:
    """Embed a list of strings locally with sentence-transformers."""
    return EMBEDDER.encode(texts, normalize_embeddings=True).tolist()


def build_index():
    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")) +
                   glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not files:
        sys.exit(f"No documents found in {DOCS_DIR}. Add some .md/.txt files.")

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(
        "troubleshooting", metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        source = os.path.basename(path)
        for j, ch in enumerate(chunk_text(text)):
            ids.append(f"{source}#{j}")
            docs.append(ch)
            metas.append({"source": source})

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed_batch(docs))
    print(f"Indexed {len(ids)} chunks from {len(files)} documents.\n")
    return col


def troubleshoot(col, problem: str, k: int = 3):
    """Retrieve relevant context and generate step-by-step recommendations."""
    q_emb = embed_batch([problem])[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    contexts = res["documents"][0]
    sources = [m["source"] for m in res["metadatas"][0]]

    context_block = "\n\n".join(f"[{s}]\n{c}" for s, c in zip(sources, contexts))
    messages = [
        {"role": "system", "content": (
            "You are an experienced maintenance engineer. Using ONLY the technical "
            "context provided, give a clear, NUMBERED, step-by-step troubleshooting "
            "procedure for the reported problem. Put the safest and most likely "
            "checks first. If the context does not cover the problem, say so. Cite "
            "the source file name in square brackets where relevant."
        )},
        {"role": "user", "content": (
            f"Technical context:\n{context_block}\n\n"
            f"Reported problem: {problem}\n\n"
            f"Provide step-by-step troubleshooting recommendations."
        )},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.2})
    return resp["message"]["content"].strip(), sources


def main():
    print("=" * 72)
    print(" Local RAG troubleshooting assistant (sentence-transformers + Ollama)")
    print("=" * 72)
    col = build_index()

    demo_problems = [
        "The centrifugal pump is running but discharge pressure is very low.",
        "The hydraulic press is slow and the oil is overheating.",
        "The induction motor trips the overload relay a few minutes after starting.",
    ]
    for p in demo_problems:
        steps, sources = troubleshoot(col, p)
        print("#" * 72)
        print("PROBLEM:", p)
        print(f"(retrieved from: {', '.join(sources)})\n")
        print(steps, "\n")

    print("-" * 72)
    print("Describe your own equipment problem ('exit' to quit).")
    while True:
        try:
            p = input("\nProblem> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if p.lower() in {"exit", "quit"}:
            break
        if p:
            steps, sources = troubleshoot(col, p)
            print(f"\n(retrieved from: {', '.join(sources)})\n")
            print(steps)
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"\n[Error] {e}\n"
                 f"Make sure 'ollama serve' is running and '{MODEL}' is pulled.")
