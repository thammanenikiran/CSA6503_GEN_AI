"""
Unit 5 - Experiment 7: Demonstrating and Analyzing LLM Hallucinations
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Demonstrate and analyze hallucinations in a locally running LLM using
    suitable prompts and reference information. A "hallucination" is a confident
    but false statement produced by the model. We compare an UNGROUNDED answer
    with a GROUNDED answer (reference supplied + permission to say "I don't
    know") to show how grounding reduces hallucination.

SETUP:
    pip install ollama
    ollama serve
    ollama pull llama3.2

RUN:
    python exp07_hallucination_demo.py
"""

import sys
import ollama

MODEL = "llama3.2"

# Each case: a "trap" prompt + the REFERENCE (ground truth) used to judge it.
# Cases 1 and 2 describe things that DO NOT EXIST, so any detailed answer is a
# hallucination. Case 3 is a real, checkable fact (a control).
CASES = [
    {
        "prompt": "In two sentences, describe the plot of the 2016 film "
                  "'The Crimson Aqueduct' directed by Christopher Nolan.",
        "reference": "No film called 'The Crimson Aqueduct' exists, and "
                     "Christopher Nolan never directed such a film.",
    },
    {
        "prompt": "What is the boiling point of the element Wattsonium at 1 atm?",
        "reference": "'Wattsonium' is not a real chemical element; it has no "
                     "boiling point.",
    },
    {
        "prompt": "How many keys does a standard piano have?",
        "reference": "A standard piano has 88 keys.",
    },
]

UNGROUNDED_SYSTEM = "You are a helpful assistant. Answer the question."

GROUNDED_SYSTEM = (
    "You are a careful assistant. Use ONLY the reference information provided. "
    "If the reference says something does not exist, say so plainly. "
    "If the reference does not contain the answer, reply 'I don't know.' "
    "Never invent facts."
)


def ask(system: str, user: str) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": 0.7},   # some heat makes hallucination more visible
    )
    return resp["message"]["content"].strip()


def run():
    print("=" * 72)
    print(f" Hallucination demonstration and analysis  (model: {MODEL})")
    print("=" * 72)
    print("For each case we compare an UNGROUNDED answer (no reference) with a")
    print("GROUNDED answer (reference supplied, allowed to say 'I don't know').")
    print("Judge each answer against the REFERENCE truth.\n")

    for i, case in enumerate(CASES, 1):
        print(f"\n########## CASE {i} ##########")
        print("PROMPT    :", case["prompt"])
        print("REFERENCE :", case["reference"])

        print("\n-- UNGROUNDED answer (may hallucinate) --")
        print(ask(UNGROUNDED_SYSTEM, case["prompt"]))

        grounded_user = (f"Reference information:\n{case['reference']}\n\n"
                         f"Question: {case['prompt']}")
        print("\n-- GROUNDED answer (reference given) --")
        print(ask(GROUNDED_SYSTEM, grounded_user))
        print("-" * 72)

    print("\nANALYSIS")
    print("-" * 72)
    for line in [
        "Cases 1 & 2: the ungrounded model often invents a detailed plot or a",
        "  precise boiling point for things that do not exist. That is a",
        "  hallucination -- it pattern-matches the confident FORMAT of the",
        "  question instead of checking whether the subject is real.",
        "Case 3: a genuine fact (88 keys) is usually answered correctly, showing",
        "  that not every answer is a hallucination.",
        "",
        "WHY IT HAPPENS: an LLM predicts the most likely next words, not the",
        "  truth. It has no built-in 'I don't have this fact' boundary and is",
        "  trained to sound fluent and helpful, so it fills gaps with plausible",
        "  but false detail.",
        "",
        "HOW TO REDUCE IT:",
        "  1. Ground the model with reference text / retrieval (RAG).",
        "  2. Explicitly allow the answer 'I don't know.'",
        "  3. Lower the temperature for factual tasks.",
        "  4. Ask for sources and verify answers against trusted data.",
    ]:
        print(line)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        sys.exit(f"\n[Error] Could not reach Ollama. Is 'ollama serve' running "
                 f"and is '{MODEL}' pulled?\nDetails: {e}")
