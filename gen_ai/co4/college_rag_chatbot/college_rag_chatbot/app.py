"""
app.py
======
Flask web application exposing the College RAG Chatbot:
  - GET  /              -> chat UI (templates/index.html)
  - POST /api/chat      -> {"query": "..."} -> JSON ChatResponse
  - GET  /api/health     -> simple health check
  - GET  /api/sample-questions -> a few example questions for the UI
"""

import os
from flask import Flask, request, jsonify, render_template

from rag_engine import RAGChatbot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")

app = Flask(__name__)

# Instantiate once at startup (indexing the small corpus is fast; for a
# large document collection this would be precomputed / cached to disk).
bot = RAGChatbot(DOCS_DIR, top_k=4, min_score=0.06, use_llm=False)

SAMPLE_QUESTIONS = [
    "What is the last date to apply for admission?",
    "How much is the B.Tech tuition fee?",
    "What CGPA is needed to sit for placements?",
    "What is the hostel in-time on weekdays?",
    "How many books can a PG student borrow from the library?",
    "What is the capital of France?",  # intentionally out-of-domain
]


@app.route("/")
def index():
    return render_template("index.html", sample_questions=SAMPLE_QUESTIONS)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    response = bot.chat(query)
    return jsonify({
        "answer": response.answer,
        "sources": response.sources,
        "status": response.status,
        "num_chunks_retrieved": len(response.retrieved),
    })


@app.route("/api/sample-questions")
def api_samples():
    return jsonify(SAMPLE_QUESTIONS)


@app.route("/demo")
def demo():
    """
    Renders a static page pre-populated with real chatbot responses.
    Used only to generate documentation screenshots (not part of the
    normal user-facing app).
    """
    demo_queries = [
        "What is the last date to apply for admission?",
        "How much is the B.Tech tuition fee per year?",
        "What CGPA is needed to sit for placements?",
        "What is the capital of France?",
    ]
    conversation = []
    for q in demo_queries:
        resp = bot.chat(q)
        conversation.append({
            "query": q,
            "answer": resp.answer,
            "sources": resp.sources,
            "status": resp.status,
        })
    return render_template("demo.html", conversation=conversation)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "chunks_indexed": len(bot.chunks)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
