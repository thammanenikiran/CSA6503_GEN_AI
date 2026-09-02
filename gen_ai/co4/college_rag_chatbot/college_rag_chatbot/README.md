# RAG-Based College Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions
using a collection of college documents (admissions, fees, courses, exams,
hostel life, and placements). Built entirely in Python with a Flask web UI.

---

## 1. Problem Statement

Build a chatbot that:
1. Retrieves relevant information from a collection of college documents.
2. Generates an answer grounded in the retrieved content (not from the
   model's own unconstrained memory).
3. Is exposed through a working user interface.
4. Handles invalid / empty / unexpected input gracefully.

---

## 2. Application Architecture

```
                     ┌─────────────────────────┐
                     │   documents/*.txt        │  (6 college documents)
                     └────────────┬──────────────┘
                                  │  INGESTION
                                  ▼
                     ┌─────────────────────────┐
                     │  Section-aware chunker    │  chunk_document()
                     │  (paragraph + heading      │
                     │   aware, ~90 words/chunk)  │
                     └────────────┬──────────────┘
                                  │  INDEXING
                                  ▼
                     ┌─────────────────────────┐
                     │   TfidfRetriever          │  TF-IDF (1-2 grams)
                     │   (scikit-learn)           │  + cosine similarity
                     └────────────┬──────────────┘
     User question ──────────────►│  RETRIEVAL (top-k, score-gated)
                                  ▼
                     ┌─────────────────────────┐
                     │  Top-k relevant chunks     │  AUGMENTATION
                     │  (with source + heading)   │
                     └────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ ExtractiveSynthesis        │  GENERATION
                     │ Generator (default,        │  (or LLMGenerator,
                     │ fully offline)             │   pluggable)
                     └────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  Answer + cited sources    │  →  Flask JSON API
                     └────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │   Chat UI (HTML/CSS/JS)    │
                     └─────────────────────────┘
```

### Request Workflow (`RAGChatbot.chat()`)

1. **Validate** the raw input (empty / whitespace / too long / pure
   punctuation are rejected before any retrieval work happens).
2. **Expand** the query with a small synonym table (e.g. "last date" →
   also search for "deadline").
3. **Retrieve** the top-k chunks by TF-IDF cosine similarity, subject to
   a minimum-score threshold.
4. **Domain gate**: if the best match is weak *and* the query contains no
   recognisable college-domain keyword, treat it as no-match rather than
   returning a low-confidence guess.
5. **Generate**: the default `ExtractiveSynthesisGenerator` re-ranks
   sentences inside the retrieved chunks by query-term overlap, removes
   duplicate/overlapping sentences, and stitches the best ones into an
   answer — every sentence is copied verbatim from a source document, so
   the answer cannot contain a hallucinated fact.
6. **Respond** with the answer, the list of `document (section)` sources
   used, and a status code (`ok`, `no_match`, `empty_input`,
   `invalid_input`, `too_long`, `error`) that the UI uses to show a
   subtle badge.

### Project Structure

```
college_rag_chatbot/
├── documents/                  # Knowledge base (6 .txt files)
│   ├── admissions.txt
│   ├── fees_and_scholarships.txt
│   ├── courses_and_departments.txt
│   ├── exams_and_library.txt
│   ├── hostel_and_campus_life.txt
│   └── placements.txt
├── rag_engine.py                # Core RAG pipeline (chunking, retrieval, generation)
├── app.py                       # Flask app (UI + REST API)
├── templates/
│   ├── index.html               # Chat UI
│   └── demo.html                # Static conversation view (for screenshots)
├── static/
│   ├── style.css
│   └── script.js
├── test_chatbot.py              # Automated test suite (19 cases)
├── test_results.md              # Generated test report
├── screenshots/                 # UI screenshots
├── requirements.txt
└── README.md                    # This file
```

---

## 3. Model / Technique Justification

| Stage | Technique used | Why |
|---|---|---|
| Chunking | Paragraph/heading-aware splitting (custom) | Keeps a section's facts (e.g. a bullet list of dates) attached to the heading that gives them meaning, instead of an arbitrary fixed-size window that can cut a list in half. |
| Retrieval | **TF-IDF + cosine similarity** (`scikit-learn`) with unigrams+bigrams | Chosen deliberately over a dense/embedding retriever (e.g. `sentence-transformers`) because it needs **no model download and no internet access at run time or index time** — it is 100% reproducible in any offline grading environment, has zero GPU/large-dependency requirement, and is fast enough (sub-2 ms/query) for a small college corpus. For a corpus of this size (a few dozen chunks), TF-IDF's precision is close to a dense retriever's; the gap widens mainly on paraphrases (see Limitations). |
| Generation (default) | **Extractive sentence-synthesis** (custom, rule-based) | Since a hosted LLM call also requires internet + an API key, the default generator instead re-ranks and stitches sentences *from the retrieved text itself*. This guarantees the RAG property that matters most for a factual college-helpdesk use case: **zero hallucination** — every fact in the answer is traceable to a specific document and section, which is shown to the user as a citation. |
| Generation (pluggable) | `LLMGenerator` (OpenAI-compatible chat endpoint) | Included and fully wired in `rag_engine.py` for architectural completeness. If `OPENAI_API_KEY` is set and `use_llm=True` is passed to `RAGChatbot`, the same retrieved context is sent to a real LLM for a more fluent, paraphrased answer, with automatic fallback to the offline generator if the API call fails. This demonstrates the retrieval/generation separation that makes RAG systems swappable in production. |
| Web framework | Flask | Lightweight, no build step, ships with the environment, good fit for a small REST + server-rendered UI. |

**In one sentence:** the system implements the full RAG pattern (retrieve → augment → generate) using a lightweight sparse retriever and a grounded extractive generator so it is deterministic, fast, dependency-light, and runs completely offline — with a clearly pluggable path to a real LLM for production use.

---

## 4. Running the Application

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Also runnable as a terminal chatbot (no Flask needed):
```bash
python rag_engine.py
```

### REST API

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/chat` | POST | `{"query": "..."}` | `{answer, sources, status, num_chunks_retrieved}` |
| `/api/health` | GET | — | `{status, chunks_indexed}` |
| `/api/sample-questions` | GET | — | list of example questions |

---

## 5. Testing

`test_chatbot.py` runs 19 automated cases across 5 categories and writes a
Markdown report (`test_results.md`).

**Result: 19/19 passed (100%)** — see `test_results.md` for the full
question-by-question transcript with sources and latency.

| Category | Example | What it checks |
|---|---|---|
| Normal, one per document | "What is the intake for Computer Science and Engineering?" | Basic retrieval + grounded answer for every source document |
| Paraphrased | "Can I get my money back if I cancel my seat?" (doc says "refund policy... withdraws admission") | Retriever generalises beyond exact keyword match |
| Ambiguous / multi-topic | "Tell me about fees and hostel rules" | Answer draws from more than one document sensibly |
| Out-of-domain | "What is the capital of France?", "Write a python program to sort a list", "Who won the cricket world cup in 2023?" | System says "no match" instead of forcing an unrelated document into an answer |
| Empty / whitespace / `None` | `""`, `"   "`, `None` | Rejected with a friendly prompt, not a crash |
| Pure punctuation | `"!!!@@@###???"` | Rejected as invalid input |
| Extremely long input | 2000+ word string | Rejected with a "please shorten" message instead of being silently truncated |
| Single word | `"fees"` | Very short queries still retrieve something reasonable |

All latencies were **under 2 ms** per query (offline TF-IDF retrieval on a
39-chunk corpus), confirming the pipeline is fast enough for real-time chat.

### Screenshots

- `screenshots/01_initial_ui.png` — chat UI on load, with clickable sample
  questions.
- `screenshots/02_conversation_demo.png` — a real multi-turn conversation
  (captured from live `bot.chat()` calls, not mocked), including a
  correctly rejected out-of-domain question.

---

## 6. Limitations of the Selected Technique

Being transparent about where this specific design breaks is as important
as the happy path. Three limitations were **found empirically** while
building and testing this project, not just listed in theory:

1. **TF-IDF has no semantic understanding — it only matches surface
   word forms.** Concretely: the query *"What is the **last date** to
   apply for admission?"* initially scored **0.0** cosine similarity
   against the chunk containing *"Application **deadline**: 20 June
   2026"* — the two phrases mean the same thing but share no words. A
   small hand-built synonym table (`QUERY_SYNONYMS` in `rag_engine.py`)
   was added to patch this specific case, but it only covers the
   paraphrases someone thought to anticipate. A dense/embedding
   retriever (e.g. `sentence-transformers` + FAISS/Chroma) would close
   this gap generally, because it compares meaning, not word overlap —
   at the cost of needing a model download and more compute.

2. **Retrieval score alone cannot distinguish "shares a rare word" from
   "is actually relevant."** Out-of-domain queries like *"Who won the
   cricket world cup in 2023?"* originally still retrieved a chunk (the
   word "cricket" appears once, in a sentence about the campus cricket
   ground) with a non-trivial similarity score. A two-tier acceptance
   gate — requiring either a high score, or a lower score *plus* an
   explicit domain keyword in the query — was added to filter these out,
   but it is a heuristic patch, not a principled fix. A cross-encoder
   re-ranker or an LLM-based "is this context sufficient to answer?"
   check would be more robust.

3. **The extractive generator ranks sentences by keyword overlap, not by
   which sentence best answers the question.** For the deadline example
   above, once the correct chunk *was* retrieved, the generator's
   sentence-selection step still sometimes preferred a sentence with more
   raw keyword overlap (e.g. "apply" + "admission") over the more
   specific one-line fact ("Application deadline: 20 June 2026") because
   both scored similarly under a simple word-overlap heuristic. It never
   hallucinates (every sentence is copied verbatim from a source), but it
   can pick a *less precise* true sentence over a *more precise* true
   one. A real LLM generator (the included `LLMGenerator` path) reads the
   whole context and would phrase this correctly and concisely.

4. **No conversational memory.** Each query is handled independently;
   the system cannot resolve "What about for the M.Tech program?" as a
   follow-up to a previous question about B.Tech fees.

5. **Static, manually-curated document set.** The 6 sample `.txt` files
   were written for this project rather than scraped from a real college
   website, and the corpus is small (39 chunks). Retrieval quality and
   the tradeoffs above would look somewhat different at the scale of a
   real college's full document set (handbooks, PDFs, circulars, FAQs).

---

## 7. Suggested Improvements

1. **Swap in a dense retriever** (`sentence-transformers` embeddings +
   FAISS/Chroma vector store) to fix the synonym/paraphrase gap
   (Limitation 1) and improve recall on more casually-phrased questions.
   Even a hybrid retriever (TF-IDF + embeddings, combined by
   Reciprocal Rank Fusion) would keep TF-IDF's precision on exact terms
   (course codes, dates, INR amounts) while adding semantic recall.
2. **Use a real LLM for generation** by enabling the already-implemented
   `LLMGenerator` path — this alone would fix Limitation 3, since a
   language model can synthesise the single correct sentence into a
   direct, well-phrased answer instead of relying on a keyword-overlap
   heuristic.
3. **Add a cross-encoder re-ranker** on top of the initial TF-IDF
   retrieval to more precisely judge chunk relevance (Limitation 2)
   before generation.
4. **Add conversational memory** (a rolling window of the last few
   turns) so follow-up questions can be resolved.
5. **Automate document ingestion** from PDFs/Word docs/the college
   website via a scheduled crawler, with versioning so answers can cite
   "as of &lt;date&gt;" and stay current without manual re-indexing.
6. **Add feedback capture** (a thumbs up/down per answer) to build a set
   of real failure cases for continual improvement, and to eventually
   fine-tune retrieval thresholds instead of hand-tuning them as was done
   here.
7. **Add authentication/personalisation** so the bot could answer
   student-specific questions (e.g. "How many books have I borrowed?")
   by connecting to the college's actual student information system.

---

## 8. Submission Checklist

- [x] Source code (`rag_engine.py`, `app.py`, templates, static assets)
- [x] Working UI (Flask chat interface, screenshots included)
- [x] Test results (`test_chatbot.py` → `test_results.md`, 19/19 passing)
- [x] Architecture and workflow explanation (this document, §2)
- [x] Model/technique justification (§3)
- [x] Limitations + suggested improvements, backed by concrete examples
      found while testing (§6, §7)
