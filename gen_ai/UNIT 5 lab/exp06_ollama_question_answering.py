"""
Unit 5 - Experiment 6: Question Answering using a Local LLM through Ollama and Python
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Demonstrate question answering using a local Large Language Model through
    Ollama and Python. This is an open-domain, CONVERSATIONAL assistant that
    remembers previous turns, so follow-up questions work.

SETUP:
    pip install ollama
    ollama serve
    ollama pull llama3.2

RUN:
    python exp06_ollama_question_answering.py
"""

import sys
import ollama

MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are a helpful and concise question-answering assistant. "
    "Give correct, to-the-point answers. If you are unsure, say so honestly."
)


def ask(history: list, question: str) -> str:
    """Send the running conversation plus the new question to the model."""
    history.append({"role": "user", "content": question})
    resp = ollama.chat(model=MODEL, messages=history, options={"temperature": 0.3})
    reply = resp["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    print("=" * 68)
    print(f" Conversational Question Answering via Ollama  (model: {MODEL})")
    print("=" * 68)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Sample questions. The 2nd/3rd rely on memory of the 1st answer (Tokyo).
    for q in ["What is the capital of Japan?",
              "What is its approximate population?",
              "Name two famous landmarks there."]:
        print(f"\nQ: {q}")
        print(f"A: {ask(history, q)}")

    print("\n" + "-" * 68)
    print("Ask your own questions ('exit' to quit, 'reset' to clear memory).")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(conversation memory cleared)")
            continue
        if q:
            print("A:", ask(history, q))
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"\n[Error] Could not reach Ollama. Is `ollama serve` running "
                 f"and is '{MODEL}' pulled?\nDetails: {e}")
