"""
Unit 5 - Experiment 3: Question-Answering System with a Local LLM
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Integrate a locally running Large Language Model (Ollama) with a Python
    application to implement a question-answering system. Answers are grounded
    ONLY in a supplied context document, so the system stays factual and admits
    when it does not know.

SETUP:
    pip install ollama
    ollama serve
    ollama pull llama3.2

RUN:
    python exp03_question_answering.py                 # uses the built-in sample context
    python exp03_question_answering.py mynotes.txt     # answer questions about your own file
"""

import sys
import ollama

MODEL = "llama3.2"

SAMPLE_CONTEXT = """\
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris,
France. It is named after the engineer Gustave Eiffel, whose company designed
and built the tower for the 1889 World's Fair. The tower is 330 metres tall and
was the tallest man-made structure in the world until the Chrysler Building in
New York was completed in 1930. It has three visitor levels; the third level is
276 metres above the ground. About seven million people visit the tower each year.
"""


def answer(question: str, context: str) -> str:
    """Ask the local model to answer using ONLY the given context."""
    messages = [
        {"role": "system", "content": (
            "You are a question-answering assistant. Answer the user's question "
            "using ONLY the information in the provided context. If the answer is "
            "not contained in the context, reply exactly: "
            "'I cannot find that in the provided context.' Keep answers short and factual."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.1})
    return resp["message"]["content"].strip()


def load_context() -> str:
    """Use a file passed on the command line, else the built-in sample."""
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return SAMPLE_CONTEXT


def main():
    context = load_context()
    print("=" * 68)
    print(" Question-Answering System (answers grounded on a context document)")
    print("=" * 68)
    print(f"Context loaded ({len(context)} characters). "
          f"Type a question, or 'exit' to quit.\n")

    # Demo questions so the experiment produces output immediately.
    # The third one is NOT in the context, to show the honest refusal.
    for q in ["How tall is the Eiffel Tower?",
              "Who designed the Eiffel Tower?",
              "How many floors does the Empire State Building have?"]:
        print(f"Q: {q}")
        print(f"A: {answer(q, context)}\n")

    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print("A:", answer(q, context), "\n")
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"\n[Error] Could not reach Ollama. Is `ollama serve` running "
                 f"and is '{MODEL}' pulled?\nDetails: {e}")
