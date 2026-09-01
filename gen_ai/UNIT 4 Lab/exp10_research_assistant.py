"""
Unit 4 - Experiment 10: AI-Based Research Assistance Application
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop an AI-based Research Assistance application that accepts a research
    topic and generates relevant information, keywords, and a concise summary.

PIPELINE:
    topic -> flan-t5-base generates an overview and sub-topics (information)
          -> TF-IDF extracts the most important keywords
          -> bart-large-cnn compresses everything into a concise summary
          -> a markdown report is saved in outputs/

RUN:
    python exp10_research_assistant.py
    python exp10_research_assistant.py "solid state batteries for electric vehicles"
"""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

GEN_MODEL = "google/flan-t5-base"
SUM_MODEL = "facebook/bart-large-cnn"
BASE = Path(__file__).parent

DEFAULT_TOPIC = "5G network slicing for industrial automation"

# The assistant answers these five research questions about any topic.
QUESTIONS = [
    ("Definition", "Explain in 4 sentences what {topic} means in engineering."),
    ("Working principle", "Describe the working principle and key components of {topic} in 4 sentences."),
    ("Applications", "List and explain four real world engineering applications of {topic}."),
    ("Advantages", "Explain four advantages of {topic} in engineering practice."),
    ("Challenges", "Explain four current technical challenges and limitations of {topic}."),
]


def extract_keywords(text, top_n=12):
    """Rank words and 2-word phrases by TF-IDF weight."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 max_features=400)
    matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), matrix.toarray()[0])
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in ranked[:top_n]]


def main():
    topic = " ".join(sys.argv[1:]) or DEFAULT_TOPIC

    print("Loading the pre-trained models...")
    generator = pipeline("text2text-generation", model=GEN_MODEL)
    summarizer = pipeline("summarization", model=SUM_MODEL)

    print(f"\n=== Research Assistant ===\nTopic: {topic}\n")

    sections = {}
    for heading, template in QUESTIONS:
        print(f"Researching: {heading}...")
        prompt = template.format(topic=topic)
        answer = generator(prompt, max_new_tokens=200, do_sample=True,
                           temperature=0.7, top_p=0.9)[0]["generated_text"].strip()
        sections[heading] = answer

    full_text = " ".join(sections.values())

    print("Extracting keywords...")
    keywords = extract_keywords(f"{topic}. {full_text}")

    print("Writing the concise summary...")
    summary = summarizer(full_text[:4000], max_length=150, min_length=60,
                         do_sample=False)[0]["summary_text"].strip()

    # ---- Display -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESEARCH REPORT: {topic.upper()}")
    print("=" * 70)
    for heading, body in sections.items():
        print(f"\n## {heading}\n{body}")
    print(f"\n## Keywords\n{', '.join(keywords)}")
    print(f"\n## Concise summary\n{summary}")
    print("=" * 70)

    # ---- Save as a markdown report -----------------------------------------
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in topic.lower())[:50].strip("_")
    report = [f"# Research Report: {topic}\n"]
    report += [f"## {h}\n\n{b}\n" for h, b in sections.items()]
    report.append("## Keywords\n\n" + ", ".join(keywords) + "\n")
    report.append("## Concise Summary\n\n" + summary + "\n")
    path = out_dir / f"exp10_{slug}.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()
