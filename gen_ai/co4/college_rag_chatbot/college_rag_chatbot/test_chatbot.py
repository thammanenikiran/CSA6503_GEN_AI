"""
test_chatbot.py
================
Automated test suite for the College RAG Chatbot.

Run with:  python test_chatbot.py
Produces:
  - console output (pass/fail per case)
  - test_results.md  (a Markdown report suitable for submission)

Test categories
----------------
1. Normal / in-domain questions covering every source document.
2. Paraphrased questions (different wording than the source text) to
   check the retriever generalises beyond exact keyword overlap.
3. Edge cases: empty string, whitespace-only, extremely long input,
   pure punctuation/gibberish, very short input.
4. Out-of-domain questions (general knowledge, unrelated topics).
5. Ambiguous / multi-topic questions.
"""

import os
import sys
import time
from rag_engine import RAGChatbot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")


TEST_CASES = [
    # ---- category, query, expectation ----
    ("Normal - Admissions", "What is the last date to apply for admission?", "ok"),
    ("Normal - Fees", "How much is the B.Tech tuition fee per year?", "ok"),
    ("Normal - Courses", "What is the intake for Computer Science and Engineering?", "ok"),
    ("Normal - Exams", "What percentage of attendance is required for the semester exam?", "ok"),
    ("Normal - Hostel", "What time do I need to be back in the hostel on weekdays?", "ok"),
    ("Normal - Placements", "What was the highest placement package last year?", "ok"),
    ("Paraphrased", "When do I need to submit original documents after seat allotment?", "ok"),
    ("Paraphrased", "Is there a fee waiver for good sports players?", "ok"),
    ("Paraphrased", "Can I get my money back if I cancel my seat?", "ok"),
    ("Ambiguous / multi-topic", "Tell me about fees and hostel rules", "ok"),
    ("Out-of-domain", "What is the capital of France?", "no_match"),
    ("Out-of-domain", "Write a python program to sort a list", "no_match"),
    ("Out-of-domain", "Who won the cricket world cup in 2023?", "no_match"),
    ("Edge - empty string", "", "empty_input"),
    ("Edge - whitespace only", "     ", "empty_input"),
    ("Edge - gibberish/punctuation", "!!!@@@###???", "invalid_input"),
    ("Edge - single word", "fees", "ok"),
    ("Edge - very long input", "fees " * 400, "too_long"),
    ("Edge - None (simulated bad client)", None, "empty_input"),
]


def run_tests():
    print("Loading RAG chatbot and indexing documents...\n")
    bot = RAGChatbot(DOCS_DIR, top_k=4, min_score=0.06, use_llm=False)
    print(f"Indexed {len(bot.chunks)} chunks from "
          f"{len(set(c.source for c in bot.chunks))} documents.\n")

    results = []
    passed = 0
    for category, query, expected_status in TEST_CASES:
        start = time.time()
        response = bot.chat(query)
        elapsed_ms = (time.time() - start) * 1000

        ok = response.status == expected_status
        passed += int(ok)
        results.append({
            "category": category,
            "query": query,
            "expected_status": expected_status,
            "actual_status": response.status,
            "pass": ok,
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": round(elapsed_ms, 2),
        })

        display_query = query if query not in (None, "") else repr(query)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] ({category}) Query: {display_query!r}")
        print(f"       expected={expected_status} actual={response.status} "
              f"latency={elapsed_ms:.1f}ms")
        print(f"       Answer: {response.answer[:160]}"
              f"{'...' if len(response.answer) > 160 else ''}")
        if response.sources:
            print(f"       Sources: {response.sources}")
        print()

    total = len(TEST_CASES)
    print(f"\n{passed}/{total} test cases passed "
          f"({(passed/total)*100:.1f}%).")

    write_markdown_report(results, passed, total)
    return passed, total, results


def write_markdown_report(results, passed, total):
    lines = []
    lines.append("# College RAG Chatbot - Test Results\n")
    lines.append(f"**Summary:** {passed}/{total} test cases passed "
                  f"({(passed/total)*100:.1f}%)\n")
    lines.append("| # | Category | Query | Expected | Actual | Pass | Latency (ms) |")
    lines.append("|---|----------|-------|----------|--------|------|---------------|")
    for i, r in enumerate(results, 1):
        q = r["query"] if r["query"] not in (None, "") else repr(r["query"])
        q = str(q).replace("|", "\\|")
        if len(q) > 60:
            q = q[:57] + "..."
        lines.append(
            f"| {i} | {r['category']} | {q} | {r['expected_status']} | "
            f"{r['actual_status']} | {'✅' if r['pass'] else '❌'} | "
            f"{r['latency_ms']} |"
        )

    lines.append("\n## Full answers\n")
    for i, r in enumerate(results, 1):
        q = r["query"] if r["query"] not in (None, "") else repr(r["query"])
        lines.append(f"### {i}. {r['category']} — `{q}`")
        lines.append(f"- **Status:** {r['actual_status']} "
                      f"(expected: {r['expected_status']}, "
                      f"{'PASS' if r['pass'] else 'FAIL'})")
        lines.append(f"- **Answer:** {r['answer']}")
        if r["sources"]:
            lines.append(f"- **Sources:** {', '.join(r['sources'])}")
        lines.append("")

    out_path = os.path.join(BASE_DIR, "test_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDetailed report written to: {out_path}")


if __name__ == "__main__":
    passed, total, _ = run_tests()
    sys.exit(0 if passed == total else 1)
