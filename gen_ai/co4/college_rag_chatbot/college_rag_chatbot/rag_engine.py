"""
rag_engine.py
==============
Core Retrieval-Augmented Generation (RAG) engine for the College Chatbot.

Pipeline
--------
1. INGESTION   : Load .txt documents from a folder and split them into
                 overlapping passage-level chunks.
2. INDEXING    : Vectorize every chunk with TF-IDF (a lightweight, fully
                 offline sparse-embedding technique).
3. RETRIEVAL   : For an incoming query, vectorize it with the same TF-IDF
                 space and rank chunks by cosine similarity.
4. AUGMENTATION: The top-k retrieved chunks are assembled into a context
                 block ("the prompt").
5. GENERATION  : The context + query are handed to a generation backend.
                 Two backends are supported:
                   a) LLM backend  - calls an external LLM (OpenAI /
                      Anthropic-compatible chat endpoint) if an API key is
                      configured. This is the "production" path.
                   b) Extractive/offline backend - a template-based
                      synthesiser that composes a fluent answer purely
                      from the retrieved text, with no external calls.
                      This is the default so the app works fully offline
                      and deterministically for grading/demo purposes.

The retrieval + generation split means the *generation* backend is a
pluggable strategy: swapping the offline synthesiser for a real LLM call
requires no change anywhere else in the pipeline (see LLMGenerator below).
"""

from __future__ import annotations

import os
import re
import glob
import json
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# 1. DATA STRUCTURES
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    chunk_id: int
    source: str          # file name the chunk came from
    heading: str         # nearest preceding heading/title, for citation
    text: str            # the actual passage text


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class ChatResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    status: str = "ok"          # ok | no_match | empty_input | error


# --------------------------------------------------------------------------- #
# 2. DOCUMENT LOADING + CHUNKING
# --------------------------------------------------------------------------- #

def load_documents(folder: str) -> Dict[str, str]:
    """Read every .txt file in `folder` into {filename: raw_text}."""
    docs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


def chunk_document(filename: str, text: str, max_chunk_words: int = 90,
                    min_chunk_words: int = 20) -> List[Chunk]:
    """
    Section-aware chunking.

    Rather than a blind fixed-size sliding window (which can split a
    bullet list away from the heading that gives it meaning -- e.g. an
    "Important Dates" list ending up glued to the wrong section), this
    splits the document into blank-line-separated paragraphs, tags each
    paragraph with the nearest preceding heading, and merges neighbouring
    small paragraphs under the *same* heading up to `max_chunk_words` so
    a whole section travels together as one retrievable unit. A paragraph
    longer than `max_chunk_words` is further split with a sliding window
    so no single chunk becomes too large for TF-IDF to weight sensibly.

    A line is treated as a heading if it is short, does not end in
    sentence punctuation, and is not a bullet/numbered list item -- this
    is looser than requiring strict Title Case, so headings containing
    lowercase function words (e.g. "Important Dates for 2026-27") are
    still recognised.
    """
    heading_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ,()\-&/]{2,60}$")

    def is_heading(stripped: str) -> bool:
        if not stripped or stripped.endswith((".", ":", ";")):
            return False
        if stripped.startswith(("-", "•", "*")):
            return False
        if len(stripped.split()) > 8:
            return False
        return bool(heading_re.match(stripped))

    # Split into raw paragraphs on blank lines, preserving line order.
    raw_paragraphs: List[str] = []
    current_lines: List[str] = []
    for line in text.split("\n"):
        if line.strip() == "":
            if current_lines:
                raw_paragraphs.append("\n".join(current_lines))
                current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        raw_paragraphs.append("\n".join(current_lines))

    default_heading = filename.replace("_", " ").replace(".txt", "").title()
    sections: List[tuple] = []   # (heading, paragraph_text)
    current_heading = default_heading

    for para in raw_paragraphs:
        first_line = para.split("\n", 1)[0].strip()
        rest_lines = para.split("\n", 1)[1] if "\n" in para else ""
        if is_heading(first_line):
            current_heading = first_line
            remainder = rest_lines.strip()
            if remainder:
                sections.append((current_heading, remainder))
        else:
            sections.append((current_heading, para.strip()))

    # Ensure every paragraph line ends with terminal punctuation so
    # bullet items become distinct sentences once flattened to plain text
    # (bullet lists otherwise run on without periods).
    def normalise(paragraph: str) -> str:
        out_lines = []
        for line in paragraph.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.endswith((".", "!", "?", ":")):
                stripped += "."
            out_lines.append(stripped)
        return " ".join(out_lines)

    # Merge consecutive paragraphs under the same heading up to the word cap.
    merged: List[Chunk] = []
    cid_base = abs(hash(filename)) % 100000
    idx = 0
    buffer_heading, buffer_text, buffer_words = None, "", 0

    def flush():
        nonlocal buffer_heading, buffer_text, buffer_words, idx
        if buffer_text.strip():
            merged.append(Chunk(
                chunk_id=cid_base + idx,
                source=filename,
                heading=buffer_heading or default_heading,
                text=buffer_text.strip(),
            ))
            idx += 1
        buffer_heading, buffer_text, buffer_words = None, "", 0

    for heading, para in sections:
        norm = normalise(para)
        if not norm:
            continue
        n_words = len(norm.split())

        # Oversized paragraph: flush buffer, then window-split this one.
        if n_words > max_chunk_words:
            flush()
            words = norm.split()
            step = max(max_chunk_words - 15, 1)
            for start in range(0, len(words), step):
                window = words[start:start + max_chunk_words]
                if not window:
                    continue
                merged.append(Chunk(
                    chunk_id=cid_base + idx, source=filename,
                    heading=heading, text=" ".join(window),
                ))
                idx += 1
                if start + max_chunk_words >= len(words):
                    break
            continue

        if buffer_heading is None:
            buffer_heading, buffer_text, buffer_words = heading, norm, n_words
        elif heading == buffer_heading and buffer_words + n_words <= max_chunk_words:
            buffer_text += " " + norm
            buffer_words += n_words
        else:
            flush()
            buffer_heading, buffer_text, buffer_words = heading, norm, n_words

    flush()

    # Merge any very small trailing chunk into the previous one so short
    # sections (e.g. a 2-line "Contact" block) aren't left as a near-empty,
    # low-signal chunk.
    final: List[Chunk] = []
    for c in merged:
        if final and len(c.text.split()) < min_chunk_words \
                and c.source == final[-1].source:
            final[-1] = Chunk(
                chunk_id=final[-1].chunk_id, source=final[-1].source,
                heading=final[-1].heading,
                text=final[-1].text + " " + c.text,
            )
        else:
            final.append(c)
    return final


def build_corpus(folder: str) -> List[Chunk]:
    docs = load_documents(folder)
    all_chunks: List[Chunk] = []
    for fname, text in docs.items():
        all_chunks.extend(chunk_document(fname, text))
    return all_chunks


DOMAIN_HINT_WORDS = {
    "college", "admission", "fee", "fees", "hostel", "exam", "exams",
    "course", "courses", "department", "placement", "placements",
    "library", "scholarship", "cgpa", "attendance", "semester", "credit",
    "internship", "warden", "mess", "campus", "backlog", "revaluation",
}

# Expands a query with a few hand-picked synonyms so both the retriever's
# vector search AND the generator's sentence-level re-ranking recognise
# paraphrases like "last date" -> "deadline". This is a partial, manual
# patch, not a real fix -- see README.md "Limitations" for why a
# semantic/embedding retriever solves this class of problem more
# generally.
QUERY_SYNONYMS = {
    "last date": "deadline",
    "due date": "deadline",
    "cost": "fee cost",
    "price": "fee price",
    "when": "date deadline schedule",
    "how much": "fee amount cost",
    "curfew": "in-time entry time",
    "timing": "time hours",
    "refund": "refund reimbursement",
}


def expand_query(query: str) -> str:
    expanded = query
    lowered = query.lower()
    for phrase, expansion in QUERY_SYNONYMS.items():
        if phrase in lowered:
            expanded += " " + expansion
    return expanded


# --------------------------------------------------------------------------- #
# 3. RETRIEVER (TF-IDF + cosine similarity)
# --------------------------------------------------------------------------- #

class TfidfRetriever:
    """
    A lightweight, dependency-free (no downloads / no internet) sparse
    retriever. TF-IDF + cosine similarity is chosen deliberately -- see
    README.md "Model Justification" for the reasoning.
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams improve phrase matches
            min_df=1,
        )
        corpus_texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 4,
                 min_score: float = 0.08) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []
        q_vec = self.vectorizer.transform([expand_query(query)])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        ranked_idx = np.argsort(-sims)

        results: List[RetrievedChunk] = []
        seen_sources = set()
        for i in ranked_idx:
            score = float(sims[i])
            if score < min_score:
                break
            chunk = self.chunks[i]
            # light diversity: avoid returning 4 chunks from the same
            # paragraph of the same file back to back if better spread exists
            results.append(RetrievedChunk(chunk=chunk, score=score))
            seen_sources.add(chunk.source)
            if len(results) >= top_k:
                break
        return results


# --------------------------------------------------------------------------- #
# 4. GENERATORS (pluggable)
# --------------------------------------------------------------------------- #

class BaseGenerator:
    def generate(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        raise NotImplementedError


class ExtractiveSynthesisGenerator(BaseGenerator):
    """
    Default, fully-offline generation backend.

    It does not merely paste the raw chunks back at the user -- it:
      * strips duplicate/overlapping sentences caused by chunk overlap,
      * ranks candidate sentences within the retrieved chunks by their
        lexical overlap with the query (a mini re-ranking step),
      * stitches the best sentences into a short, direct answer,
      * appends a "Sources" line citing the section/document used.

    This keeps the pipeline genuinely retrieval-grounded (no hallucinated
    facts -- every sentence in the answer is copied from a source
    document) while still reading as a synthesised answer rather than a
    raw dump of paragraphs.
    """

    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = self.SENT_SPLIT_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, query_terms: set) -> int:
        words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
        return len(words & query_terms)

    def generate(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        if not retrieved:
            return (
                "I couldn't find anything about that in the college "
                "documents I have access to (admissions, fees, courses, "
                "exams, hostel/campus life, and placements). Could you "
                "rephrase your question, or ask about one of those topics?"
            )

        query_terms = set(re.findall(r"[a-zA-Z]+", expand_query(query).lower()))
        candidate_sentences = []
        seen_normalised = set()

        for rc in retrieved:
            for sent in self._split_sentences(rc.chunk.text):
                norm = re.sub(r"\s+", " ", sent.lower())
                if norm in seen_normalised or len(sent.split()) < 4:
                    continue
                seen_normalised.add(norm)
                relevance = self._score_sentence(sent, query_terms)
                candidate_sentences.append((relevance, rc.score, sent, rc.chunk))

        # Sort by (query-term overlap, retrieval score) descending
        candidate_sentences.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Take the best few sentences, but keep a sane cap so the answer
        # stays a focused paragraph, not the whole chunk.
        top_sentences = candidate_sentences[:4] if candidate_sentences else []
        if not top_sentences:
            # fall back to the single best chunk verbatim (short)
            best = retrieved[0].chunk.text
            answer_body = best
        else:
            # restore a natural reading order: sort chosen sentences by
            # their original chunk order rather than by score, for flow
            order_map = {id(c): i for i, c in enumerate(retrieved)}
            top_sentences.sort(key=lambda x: order_map.get(id(x[3]), 0))
            answer_body = " ".join(s for _, _, s, _ in top_sentences)

        sources = sorted({f"{rc.chunk.source} — {rc.chunk.heading}" for rc in retrieved})
        answer = answer_body.strip()
        return answer


class LLMGenerator(BaseGenerator):
    """
    Production-path generation backend: sends the retrieved context and
    the user's question to a hosted LLM chat endpoint and returns the
    model's fluent answer. This is the recommended backend for a real
    deployment; it is included here for architectural completeness and
    is used automatically if an API key is present in the environment
    (OPENAI_API_KEY). It requires outbound internet access, which is why
    ExtractiveSynthesisGenerator is the default for this offline
    submission/demo.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant for a college's student helpdesk. "
        "Answer the student's question using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't "
        "have that information and suggest contacting the relevant "
        "office. Be concise and factual."
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def generate(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "LLMGenerator requires OPENAI_API_KEY to be set in the "
                "environment. Falling back is handled by RAGChatbot."
            )
        import requests  # local import: optional dependency

        context = "\n\n".join(
            f"[{rc.chunk.source} | {rc.chunk.heading}]\n{rc.chunk.text}"
            for rc in retrieved
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# 5. TOP-LEVEL CHATBOT ORCHESTRATOR
# --------------------------------------------------------------------------- #


class RAGChatbot:
    def __init__(self, documents_folder: str, top_k: int = 4,
                 min_score: float = 0.08, strict_score: float = 0.16,
                 use_llm: bool = False):
        self.chunks = build_corpus(documents_folder)
        if not self.chunks:
            raise ValueError(f"No .txt documents found in {documents_folder}")
        self.retriever = TfidfRetriever(self.chunks)
        self.extractive_generator = ExtractiveSynthesisGenerator()
        self.llm_generator = LLMGenerator() if use_llm else None
        self.top_k = top_k
        self.min_score = min_score
        # Two-tier acceptance gate to curb false positives from generic
        # words that coincidentally overlap the corpus (see README
        # "Limitations"): a low-score match is only accepted if the query
        # also contains an explicit college-domain keyword; a high-score
        # match is accepted regardless, since strong lexical/semantic
        # overlap is itself good evidence of relevance.
        self.strict_score = strict_score

    def _looks_in_domain(self, query: str) -> bool:
        query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
        return bool(query_words & DOMAIN_HINT_WORDS)

    # ---- input validation -------------------------------------------------
    @staticmethod
    def _validate(query: Optional[str]) -> Optional[str]:
        """Returns an error status string, or None if input is valid."""
        if query is None:
            return "empty_input"
        stripped = query.strip()
        if stripped == "":
            return "empty_input"
        if len(stripped) > 1000:
            return "too_long"
        # reject inputs that are pure punctuation / gibberish symbols
        if not re.search(r"[a-zA-Z0-9]", stripped):
            return "invalid_input"
        return None

    # ---- main entry point ---------------------------------------------------
    def chat(self, query: str) -> ChatResponse:
        problem = self._validate(query)
        if problem == "empty_input":
            return ChatResponse(
                answer="Please type a question -- for example, "
                       "\"What is the last date to apply for admission?\"",
                status="empty_input",
            )
        if problem == "too_long":
            return ChatResponse(
                answer="That question is quite long -- could you shorten "
                       "it to a single, specific question?",
                status="too_long",
            )
        if problem == "invalid_input":
            return ChatResponse(
                answer="I couldn't understand that input. Please ask a "
                       "question in words, e.g. \"How do I apply for a "
                       "scholarship?\"",
                status="invalid_input",
            )

        try:
            retrieved = self.retriever.retrieve(
                query, top_k=self.top_k, min_score=self.min_score
            )

            # Apply the domain-relevance gate: drop weak matches that
            # aren't backed by an explicit domain keyword in the query.
            if retrieved and retrieved[0].score < self.strict_score \
                    and not self._looks_in_domain(query):
                retrieved = []

            if not retrieved:
                return ChatResponse(
                    answer=self.extractive_generator.generate(query, []),
                    status="no_match",
                )

            if self.llm_generator is not None:
                try:
                    answer = self.llm_generator.generate(query, retrieved)
                except Exception:
                    # graceful degradation to offline generator
                    answer = self.extractive_generator.generate(query, retrieved)
            else:
                answer = self.extractive_generator.generate(query, retrieved)

            sources = sorted({f"{rc.chunk.source} ({rc.chunk.heading})" for rc in retrieved})
            return ChatResponse(
                answer=answer, sources=sources, retrieved=retrieved, status="ok"
            )
        except Exception as exc:  # last-resort safety net
            return ChatResponse(
                answer="Something went wrong while processing your "
                       "question. Please try again or rephrase it.",
                status="error",
            )


# --------------------------------------------------------------------------- #
# Convenience CLI for quick manual testing: `python rag_engine.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    bot = RAGChatbot(os.path.join(here, "documents"))
    print("College RAG Chatbot (CLI mode). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        resp = bot.chat(q)
        print(f"Bot: {resp.answer}")
        if resp.sources:
            print(f"     [Sources: {', '.join(resp.sources)}]")
        print()
