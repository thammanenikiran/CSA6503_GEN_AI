"""
Unit 4 - Experiment 7: Summarising a Lengthy Engineering Document
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop an AI application that summarizes a lengthy engineering document
    into a short and meaningful summary using a pre-trained language model.

MODEL:
    facebook/bart-large-cnn - an encoder-decoder model fine-tuned for abstractive
    summarisation (it writes new sentences, it does not just copy lines).

METHOD:
    Long documents exceed the model's 1024-token limit, so the text is split into
    chunks, each chunk is summarised, and the chunk summaries are summarised again
    (map-reduce / hierarchical summarisation).

RUN:
    python exp07_document_summarizer.py                  # uses docs/smart_grid.txt
    python exp07_document_summarizer.py docs/my_doc.txt
"""

import sys
from pathlib import Path

from transformers import pipeline

MODEL = "facebook/bart-large-cnn"
BASE = Path(__file__).parent
DEFAULT_DOC = BASE / "docs" / "smart_grid.txt"
CHUNK_WORDS = 700          # ~ 900 tokens, safely under the 1024 limit


def chunk_text(text, size=CHUNK_WORDS):
    """Split the document into word-count chunks the model can accept."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize(summarizer, text, max_len=150, min_len=50):
    return summarizer(text, max_length=max_len, min_length=min_len,
                      do_sample=False)[0]["summary_text"].strip()


def main():
    doc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    if not doc_path.exists():
        sys.exit(f"Document not found: {doc_path}")

    text = doc_path.read_text(encoding="utf-8")
    words = len(text.split())
    print(f"Document : {doc_path.name}")
    print(f"Length   : {words} words, {len(text)} characters\n")

    print("Loading the pre-trained summarisation model...")
    summarizer = pipeline("summarization", model=MODEL)

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunk(s).\n")

    partial = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarising chunk {i}/{len(chunks)}...")
        partial.append(summarize(summarizer, chunk))

    combined = " ".join(partial)
    if len(chunks) > 1:
        print("Combining the chunk summaries into the final summary...")
        final = summarize(summarizer, combined, max_len=180, min_len=70)
    else:
        final = combined

    print("\n=== FINAL SUMMARY ===")
    print(final)

    reduction = 100 * (1 - len(final.split()) / words)
    print(f"\nOriginal: {words} words -> Summary: {len(final.split())} words "
          f"({reduction:.1f}% shorter)")

    out = BASE / "outputs"
    out.mkdir(exist_ok=True)
    (out / f"exp07_summary_{doc_path.stem}.txt").write_text(final, encoding="utf-8")
    print(f"Summary saved to: outputs/exp07_summary_{doc_path.stem}.txt")


if __name__ == "__main__":
    main()
