"""
Unit 5 - Experiment 5: Text Generation using a Local LLM through Ollama and Python
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Demonstrate text generation using a local Large Language Model through
    Ollama and the Python client, showing both a one-shot call and
    token-by-token streaming.

SETUP:
    pip install ollama
    ollama serve
    ollama pull llama3.2

RUN:
    python exp05_ollama_text_generation.py
"""

import sys
import ollama

MODEL = "llama3.2"


def generate_once(prompt: str) -> str:
    """Simple, non-streaming generation: wait for the full response."""
    resp = ollama.generate(model=MODEL, prompt=prompt)
    return resp["response"].strip()


def generate_stream(prompt: str) -> None:
    """Streaming generation: print each token the moment the model produces it."""
    for chunk in ollama.generate(model=MODEL, prompt=prompt, stream=True):
        print(chunk.get("response", ""), end="", flush=True)
    print()


def main():
    print("=" * 68)
    print(f" Text Generation via Ollama + Python  (model: {MODEL})")
    print("=" * 68)

    prompts = [
        "Explain the concept of recursion in one short paragraph.",
        "Write a haiku about the monsoon season.",
        "List three practical tips for writing clean Python code.",
    ]

    print("\n[1] One-shot (non-streaming) generation")
    print("-" * 68)
    print("Prompt:", prompts[0], "\n")
    print(generate_once(prompts[0]))

    print("\n[2] Streaming generation (token by token)")
    print("-" * 68)
    for p in prompts[1:]:
        print(f"\nPrompt: {p}\n")
        generate_stream(p)

    print("\n[3] Your turn — type a prompt (or 'exit')")
    print("-" * 68)
    while True:
        try:
            p = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if p.lower() in {"exit", "quit"}:
            break
        if p:
            generate_stream(p)
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"\n[Error] Could not reach Ollama. Is `ollama serve` running "
                 f"and is '{MODEL}' pulled?\nDetails: {e}")
