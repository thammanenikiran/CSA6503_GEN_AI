"""
Unit 5 - Experiment 4: Translation and Paraphrasing with a Local LLM
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Integrate a locally running Large Language Model (Ollama) with a Python
    application to perform text translation and text paraphrasing.

SETUP:
    pip install ollama
    ollama serve
    ollama pull llama3.2

RUN:
    python exp04_translation_paraphrasing.py
"""

import sys
import ollama

MODEL = "llama3.2"


def translate(text: str, target_language: str) -> str:
    """Translate text into the requested language."""
    messages = [
        {"role": "system", "content": (
            "You are a professional translator. Translate the user's text "
            "accurately and naturally. Output ONLY the translation, with no "
            "explanations or notes."
        )},
        {"role": "user",
         "content": f"Translate the following text into {target_language}:\n\n{text}"},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.2})
    return resp["message"]["content"].strip()


def paraphrase(text: str, style: str = "clear and simple") -> str:
    """Rewrite text in a different wording while keeping the meaning."""
    messages = [
        {"role": "system", "content": (
            "You are a writing assistant that rephrases text while preserving its "
            "meaning. Output ONLY the rewritten text."
        )},
        {"role": "user",
         "content": f"Rewrite the following text in a {style} style:\n\n{text}"},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.7})
    return resp["message"]["content"].strip()


def demo():
    print("--- DEMO ---")
    sample = "The results of the experiment were highly encouraging and exceeded our expectations."
    print("Original   :", sample)
    print("-> French  :", translate(sample, "French"))
    print("-> Hindi   :", translate(sample, "Hindi"))
    print("-> Formal  :", paraphrase(sample, "formal"))
    print()


def menu():
    while True:
        print("=" * 60)
        print(" 1) Translate text")
        print(" 2) Paraphrase text")
        print(" 3) Run demo again")
        print(" 4) Exit")
        choice = input("Choose an option> ").strip()
        if choice == "1":
            text = input("Text to translate> ").strip()
            lang = input("Target language (e.g. French, Tamil, German)> ").strip() or "French"
            print("\nTranslation:\n" + translate(text, lang) + "\n")
        elif choice == "2":
            text = input("Text to paraphrase> ").strip()
            style = input("Style (formal / simple / creative)> ").strip() or "clear and simple"
            print("\nParaphrase:\n" + paraphrase(text, style) + "\n")
        elif choice == "3":
            demo()
        elif choice in {"4", "exit", "quit"}:
            break
        else:
            print("Invalid choice.\n")
    print("Goodbye.")


if __name__ == "__main__":
    try:
        demo()
        menu()
    except (EOFError, KeyboardInterrupt):
        print()
    except Exception as e:
        sys.exit(f"\n[Error] Could not reach Ollama. Is `ollama serve` running "
                 f"and is '{MODEL}' pulled?\nDetails: {e}")
