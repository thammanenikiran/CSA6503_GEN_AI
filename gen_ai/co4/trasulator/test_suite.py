"""
test_suite.py
===============================================================================
Test cases for the English -> Tamil translation application.

Run with:  python3 test_suite.py

Produces a human-readable results log (test_results.txt) covering:
  A. Normal / representative sentences (statements, questions, negatives,
     greetings, different sentence lengths, numbers)
  B. Invalid / empty / unexpected input handling (the assignment's explicit
     requirement)
===============================================================================
"""

from translator import Translator

# We force prefer_neural=False here because this sandbox has no network access
# to a model hub. In a normal deployment, prefer_neural=True is the default
# and the app will automatically use the neural backend when available.
translator = Translator(prefer_neural=False)

NORMAL_CASES = [
    ("Simple greeting", "Hello, how are you?"),
    ("Statement - present tense", "I am going to school today."),
    ("Statement - with object", "I like books."),
    ("Question", "What is your name?"),
    ("Question 2", "Where are you going?"),
    ("Negative sentence", "I am not going to the market."),
    ("Politeness phrase", "Thank you very much."),
    ("Longer sentence", "My friend and I are studying English and Tamil at school."),
    ("Sentence with numbers", "I have two books and three pens."),
    ("Sentence with unknown/rare word", "The astrophysicist calibrated the telescope."),
    ("Short exclamation", "Good morning!"),
    ("Multi-clause sentence", "I am happy because the weather is very good today."),
]

EDGE_CASES = [
    ("Completely empty string", ""),
    ("Whitespace only", "     "),
    ("None value (non-string input)", None),
    ("Integer input (wrong type)", 12345),
    ("Digits only, no letters", "42 100 7"),
    ("Punctuation only", "!!! ??? ..."),
    ("Non-English input (French)", "Bonjour, comment ça va?"),
    ("Already-Tamil input", "வணக்கம், எப்படி இருக்கிறீர்கள்?"),
    ("HTML/script injection attempt", "<script>alert('hi')</script> Hello there"),
    ("Very long repeated input", "hello " * 300),
    ("Mixed language input", "Hello நண்பா, how are you today?"),
    ("Single unknown word", "Xylophonic"),
]


def run(cases, label, f):
    f.write(f"\n{'=' * 90}\n{label}\n{'=' * 90}\n")
    for name, inp in cases:
        result = translator.translate(inp)
        display_in = repr(inp) if not isinstance(inp, str) or len(str(inp)) < 80 else repr(inp[:77] + "...")
        f.write(f"\n[{name}]\n")
        f.write(f"  Input        : {display_in}\n")
        f.write(f"  Success      : {result.success}\n")
        f.write(f"  Backend      : {result.backend_used}\n")
        f.write(f"  Output       : {result.translated_text!r}\n")
        if result.oov_words:
            f.write(f"  OOV words    : {result.oov_words}\n")
        if result.warnings:
            f.write(f"  Warnings     : {result.warnings}\n")
        # also echo to console
        print(f"[{label[:6]}] {name:38} -> success={result.success!s:5} backend={result.backend_used:12} out={result.translated_text!r}")


def main():
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("ENGLISH -> TAMIL TRANSLATION APPLICATION - TEST RESULTS\n")
        f.write("Backend under test: rule_based (neural backend requires internet access to a model hub)\n")
        run(NORMAL_CASES, "A. NORMAL / REPRESENTATIVE SENTENCE TESTS", f)
        run(EDGE_CASES, "B. INVALID / EMPTY / UNEXPECTED INPUT TESTS", f)
    print("\nDone. Full results written to test_results.txt")


if __name__ == "__main__":
    main()
