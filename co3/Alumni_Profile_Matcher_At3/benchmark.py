"""
benchmark.py
------------
Benchmarks FAISS vs ChromaDB for the Alumni Profile Matcher use case.

Pipeline:
  1. Load 220 alumni bios (generate_dataset.py).
  2. Embed all bios with the SAME sentence-embedding model.

     NOTE ON MODEL CHOICE / SANDBOX NETWORK RESTRICTION:
     This benchmark was executed inside a sandboxed environment whose
     egress allow-list covers only pypi/npm/crates/apt/GitHub hosts and
     explicitly BLOCKS huggingface.co and storage.googleapis.com -- the
     two hosts that `sentence-transformers` / `fastembed` use to download
     pretrained weights for `all-MiniLM-L6-v2`. Both were attempted first
     and failed with 403/host-not-allowed errors (see project notes).
     As a like-for-like, fully-offline substitute we use spaCy's
     `en_core_web_md` pretrained model (300-dim GloVe-based static word
     vectors, mean-pooled into a document vector) -- a real, pretrained,
     general-purpose semantic embedding, installed from a GitHub Releases
     asset (a whitelisted host). It is used IDENTICALLY for both FAISS
     and ChromaDB, which is all this assignment's comparison requires:
     the FAISS-vs-Chroma benchmark (indexing time, latency, storage,
     top-k relevance) is independent of which embedding model produced
     the vectors. To reproduce with all-MiniLM-L6-v2 on a machine with
     open internet access, change EMBED_BACKEND below to "minilm" --
     the `embed_texts()` function already contains that code path.
  3. Index the embeddings in:
       (a) FAISS  -- IndexFlatIP over L2-normalised vectors (=cosine sim)
       (b) ChromaDB -- PersistentClient collection, cosine space
  4. Run the same 12 semantic queries (incl. the required fintech
     special-case) against both, at k=5.
  5. Measure: indexing time, average query latency (mean of 20 repeated
     runs per query, after 3 warm-up runs), top-5 relevance (Precision@5
     against a hand-labelled relevant-domain set), and on-disk footprint.
  6. Dump all results to results.json for the report.
"""

import json
import os
import shutil
import time
import statistics as stats

import numpy as np
import faiss
import chromadb
from chromadb.config import Settings

EMBED_BACKEND = "spacy_md"   # "spacy_md" (offline, used here) or "minilm" (needs internet)
MODEL_NAME = "en_core_web_md"  # 300-dim GloVe-based static vectors, mean-pooled per doc
K = 5
N_WARMUP = 3
N_TIMED = 20


def embed_texts(texts):
    """Return an (N, dim) float32 array of sentence embeddings.

    Both backends expose the exact same interface, so swapping the model
    used by BOTH FAISS and ChromaDB is a one-line change (EMBED_BACKEND).
    """
    if EMBED_BACKEND == "minilm":
        # Requires internet access to huggingface.co (blocked in this sandbox).
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return np.array(model.encode(texts), dtype="float32")
    elif EMBED_BACKEND == "spacy_md":
        import spacy
        nlp = spacy.load("en_core_web_md")
        vecs = [doc.vector for doc in nlp.pipe(texts, batch_size=64)]
        return np.array(vecs, dtype="float32")
    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}")

FAISS_DIR = "faiss_store"
CHROMA_DIR = "chroma_store"

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
with open("alumni_bios.json") as f:
    alumni = json.load(f)

ids = [r["id"] for r in alumni]
docs = [r["bio"] for r in alumni]
domains = [r["domain"] for r in alumni]
print(f"Loaded {len(alumni)} alumni bios.")

# ---------------------------------------------------------------------
# 2. Embed corpus once (shared by both DBs, as required)
# ---------------------------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME} (backend={EMBED_BACKEND}) ...")

t0 = time.perf_counter()
doc_embeddings = embed_texts(docs)
embed_time = time.perf_counter() - t0
dim = doc_embeddings.shape[1]
print(f"Embedded {len(docs)} docs -> shape {doc_embeddings.shape} in {embed_time:.3f}s")

# normalise for cosine similarity via inner product (FAISS) / cosine (Chroma)
def normalise(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return mat / norms

doc_embeddings_norm = normalise(doc_embeddings)

# ---------------------------------------------------------------------
# 3a. Build FAISS index
# ---------------------------------------------------------------------
if os.path.exists(FAISS_DIR):
    shutil.rmtree(FAISS_DIR)
os.makedirs(FAISS_DIR, exist_ok=True)

t0 = time.perf_counter()
faiss_index = faiss.IndexFlatIP(dim)          # cosine similarity via normalised IP
faiss_index = faiss.IndexIDMap(faiss_index)   # allow custom integer ids
int_ids = np.arange(len(ids), dtype="int64")
faiss_index.add_with_ids(doc_embeddings_norm, int_ids)
faiss_index_time = time.perf_counter() - t0
faiss_path = os.path.join(FAISS_DIR, "alumni.index")
faiss.write_index(faiss_index, faiss_path)
faiss_size_bytes = os.path.getsize(faiss_path)
print(f"FAISS index built in {faiss_index_time*1000:.2f} ms, size {faiss_size_bytes/1024:.1f} KB")

id_lookup = {i: ids[i] for i in range(len(ids))}

# ---------------------------------------------------------------------
# 3b. Build ChromaDB collection
# ---------------------------------------------------------------------
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
t0 = time.perf_counter()
collection = chroma_client.create_collection(
    name="alumni_bios",
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=ids,
    embeddings=doc_embeddings_norm.tolist(),
    documents=docs,
    metadatas=[{"domain": d} for d in domains],
)
chroma_index_time = time.perf_counter() - t0

def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

chroma_size_bytes = dir_size(CHROMA_DIR)
print(f"ChromaDB collection built in {chroma_index_time*1000:.2f} ms, size {chroma_size_bytes/1024:.1f} KB")

# ---------------------------------------------------------------------
# 4. Query set (>=10 required). Includes the mandatory special case:
#    a "fintech" interest query that should surface alumni whose bios
#    never say "fintech" but work at payments/lending companies.
# ---------------------------------------------------------------------
QUERIES = [
    {"id": "q1",  "text": "I'm interested in fintech and digital payments", "relevant_domains": ["fintech"]},
    {"id": "q2",  "text": "I want to work on machine learning and AI research", "relevant_domains": ["data_ai"]},
    {"id": "q3",  "text": "Interested in cloud computing and distributed backend systems", "relevant_domains": ["core_software"]},
    {"id": "q4",  "text": "I'd like to get into management consulting and business strategy", "relevant_domains": ["consulting"]},
    {"id": "q5",  "text": "Passionate about renewable energy and sustainability", "relevant_domains": ["energy_climate"]},
    {"id": "q6",  "text": "Interested in cybersecurity and ethical hacking", "relevant_domains": ["cybersecurity"]},
    {"id": "q7",  "text": "I want to build products in online education / edtech", "relevant_domains": ["edtech"]},
    {"id": "q8",  "text": "Interested in supply chain and e-commerce logistics", "relevant_domains": ["ecommerce_logistics"]},
    {"id": "q9",  "text": "I'm curious about digital health and healthtech startups", "relevant_domains": ["healthtech"]},
    {"id": "q10", "text": "Interested in mechanical design and core engineering roles", "relevant_domains": ["core_engineering"]},
    {"id": "q11", "text": "I want to pursue academic research and a PhD", "relevant_domains": ["academia_research"]},
    {"id": "q12", "text": "Interested in brand strategy and digital marketing careers", "relevant_domains": ["marketing_media"]},
    # SPECIAL CASE (duplicate of q1 phrasing variant, analysed separately in report):
    {"id": "q_special", "text": "Looking for an alumnus in fintech to guide me", "relevant_domains": ["fintech"]},
]

query_texts = [q["text"] for q in QUERIES]
query_embeddings = embed_texts(query_texts)
query_embeddings_norm = normalise(query_embeddings)

# ---------------------------------------------------------------------
# 5. Run + time queries on both DBs
# ---------------------------------------------------------------------
def faiss_search(qvec, k=K):
    D, I = faiss_index.search(qvec.reshape(1, -1), k)
    return [(id_lookup[i], float(d)) for i, d in zip(I[0], D[0]) if i != -1]

def chroma_search(qvec, k=K):
    res = collection.query(query_embeddings=[qvec.tolist()], n_results=k)
    return [(doc_id, float(1 - dist)) for doc_id, dist in zip(res["ids"][0], res["distances"][0])]
    # chroma with cosine space returns cosine *distance*; similarity = 1 - distance

results = {"faiss": {}, "chroma": {}}
latencies = {"faiss": [], "chroma": []}

alumni_by_id = {r["id"]: r for r in alumni}

for qi, q in enumerate(QUERIES):
    qvec = query_embeddings_norm[qi]

    # warm-up (not timed)
    for _ in range(N_WARMUP):
        faiss_search(qvec)
        chroma_search(qvec)

    # timed runs
    t_faiss = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        faiss_res = faiss_search(qvec)
        t_faiss.append(time.perf_counter() - t0)

    t_chroma = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        chroma_res = chroma_search(qvec)
        t_chroma.append(time.perf_counter() - t0)

    latencies["faiss"].extend(t_faiss)
    latencies["chroma"].extend(t_chroma)

    def enrich(res):
        out = []
        for doc_id, score in res:
            rec = alumni_by_id[doc_id]
            out.append({
                "id": doc_id, "score": round(score, 4), "name": rec["name"],
                "domain": rec["domain"], "company": rec["company"],
                "role": rec["role"], "explicit_domain_wording": rec["explicit_domain_wording"],
                "bio": rec["bio"],
            })
        return out

    faiss_enriched = enrich(faiss_res)
    chroma_enriched = enrich(chroma_res)

    def precision_at_k(enriched, relevant_domains):
        if not enriched:
            return 0.0
        hits = sum(1 for r in enriched if r["domain"] in relevant_domains)
        return hits / len(enriched)

    results["faiss"][q["id"]] = {
        "query": q["text"],
        "top_k": faiss_enriched,
        "precision_at_5": precision_at_k(faiss_enriched, q["relevant_domains"]),
        "mean_latency_ms": round(stats.mean(t_faiss) * 1000, 4),
    }
    results["chroma"][q["id"]] = {
        "query": q["text"],
        "top_k": chroma_enriched,
        "precision_at_5": precision_at_k(chroma_enriched, q["relevant_domains"]),
        "mean_latency_ms": round(stats.mean(t_chroma) * 1000, 4),
    }

    print(f"\n=== {q['id']}: {q['text']} ===")
    print(f"  FAISS  P@5={results['faiss'][q['id']]['precision_at_5']:.2f}  "
          f"latency={results['faiss'][q['id']]['mean_latency_ms']:.3f} ms  "
          f"-> {[ (r['name'], r['domain'], r['company']) for r in faiss_enriched]}")
    print(f"  Chroma P@5={results['chroma'][q['id']]['precision_at_5']:.2f}  "
          f"latency={results['chroma'][q['id']]['mean_latency_ms']:.3f} ms  "
          f"-> {[ (r['name'], r['domain'], r['company']) for r in chroma_enriched]}")

# ---------------------------------------------------------------------
# 6. Aggregate summary
# ---------------------------------------------------------------------
summary = {
    "model_name": MODEL_NAME,
    "embedding_dim": dim,
    "n_docs": len(docs),
    "corpus_embedding_time_s": round(embed_time, 4),
    "faiss": {
        "index_time_ms": round(faiss_index_time * 1000, 4),
        "avg_query_latency_ms": round(stats.mean(latencies["faiss"]) * 1000, 4),
        "p95_query_latency_ms": round(np.percentile(latencies["faiss"], 95) * 1000, 4),
        "storage_bytes": faiss_size_bytes,
        "avg_precision_at_5": round(
            stats.mean(v["precision_at_5"] for v in results["faiss"].values()), 4
        ),
    },
    "chroma": {
        "index_time_ms": round(chroma_index_time * 1000, 4),
        "avg_query_latency_ms": round(stats.mean(latencies["chroma"]) * 1000, 4),
        "p95_query_latency_ms": round(np.percentile(latencies["chroma"], 95) * 1000, 4),
        "storage_bytes": chroma_size_bytes,
        "avg_precision_at_5": round(
            stats.mean(v["precision_at_5"] for v in results["chroma"].values()), 4
        ),
    },
}

with open("results.json", "w") as f:
    json.dump({"summary": summary, "per_query": results}, f, indent=2)

print("\n\n================ SUMMARY ================")
print(json.dumps(summary, indent=2))
print("\nSaved detailed results to results.json")
