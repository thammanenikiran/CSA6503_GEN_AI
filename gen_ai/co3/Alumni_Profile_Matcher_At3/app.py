from flask import Flask, render_template, request
import json
import numpy as np
import spacy
import faiss

app = Flask(__name__)

# Load the embedding model
nlp = spacy.load("en_core_web_md")

# Load alumni data
with open("alumni_bios.json", "r", encoding="utf-8") as f:
    alumni = json.load(f)

# Create embeddings
texts = []

for person in alumni:
    # Handles common JSON field names
    text = " ".join([
        str(person.get("name", "")),
        str(person.get("domain", "")),
        str(person.get("company", "")),
        str(person.get("bio", "")),
        str(person.get("skills", ""))
    ])
    texts.append(text)

vectors = np.array(
    [nlp(text).vector for text in texts],
    dtype="float32"
)

# Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)

# Create FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors)


def search_alumni(query, k=5):
    """Find the top matching alumni."""
    query_vector = np.array(
        [nlp(query).vector],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        person = alumni[int(idx)]

        results.append({
            "name": person.get("name", "Unknown"),
            "domain": person.get("domain", "Unknown"),
            "company": person.get("company", "Unknown"),
            "score": round(float(score), 4)
        })

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if query:
            results = search_alumni(query)

    return render_template(
        "index.html",
        results=results,
        query=query
    )


if __name__ == "__main__":
    app.run(debug=True)