import os
import json
import csv
import time
from openai import OpenAI


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY is not set.")
    print("Run:")
    print("set OPENROUTER_API_KEY=YOUR_KEY")
    raise SystemExit

print("OpenRouter API key detected.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

MODEL = "openrouter/free"


# ============================================================
# TEST DATA - 15 CUSTOMER SUPPORT MESSAGES
# ============================================================

TEST_DATA = [
    {
        "message": "My order was supposed to arrive yesterday but the tracking still says it is in transit.",
        "category": "DELIVERY_DELAY",
        "urgency": "MEDIUM",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "I was charged twice for the same order. Please refund the extra payment.",
        "category": "PAYMENT_REFUND",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "The headphones I received only work on one side.",
        "category": "PRODUCT_DEFECT",
        "urgency": "MEDIUM",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "I forgot my password and the reset link is not reaching my email.",
        "category": "ACCOUNT_ACCESS",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "Great service, my package arrived earlier than expected!",
        "category": "FEEDBACK_OTHER",
        "urgency": "LOW",
        "sentiment": "POSITIVE"
    },
    {
        "message": "Where???",
        "category": "DELIVERY_DELAY",
        "urgency": "MEDIUM",
        "sentiment": "NEUTRAL"
    },
    {
        "message": "Order 784521 is still not shipped and I need it before tomorrow's wedding.",
        "category": "DELIVERY_DELAY",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "Refund says completed nine days ago but the money is still missing from my bank account.",
        "category": "PAYMENT_REFUND",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "Superb job guys, because obviously waiting three weeks for a phone is exactly what I wanted.",
        "category": "DELIVERY_DELAY",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "Bhai payment ho gaya but order confirm nahi hua.",
        "category": "PAYMENT_REFUND",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "The product looks nice, but the screen has a crack straight out of the box.",
        "category": "PRODUCT_DEFECT",
        "urgency": "HIGH",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "Can you tell me whether this product is available in blue?",
        "category": "FEEDBACK_OTHER",
        "urgency": "LOW",
        "sentiment": "NEUTRAL"
    },
    {
        "message": "I cannot log into my account after changing my phone number.",
        "category": "ACCOUNT_ACCESS",
        "urgency": "MEDIUM",
        "sentiment": "NEGATIVE"
    },
    {
        "message": "The refund was processed and I can now see the money in my account. Thanks!",
        "category": "PAYMENT_REFUND",
        "urgency": "LOW",
        "sentiment": "POSITIVE"
    },
    {
        "message": "My delivery is late and the box arrived damaged. The actual product seems okay.",
        "category": "DELIVERY_DELAY",
        "urgency": "MEDIUM",
        "sentiment": "NEGATIVE"
    }
]


# ============================================================
# COMMON INSTRUCTION
# ============================================================

BASE_INSTRUCTION = """
You are a support-ticket triage engine for an e-commerce company.

Classify the customer message into exactly ONE CATEGORY.

Allowed categories:

DELIVERY_DELAY
PAYMENT_REFUND
PRODUCT_DEFECT
ACCOUNT_ACCESS
FEEDBACK_OTHER

CATEGORY DEFINITIONS:

DELIVERY_DELAY:
Problems involving late, delayed, missing, or not-yet-shipped deliveries.

PAYMENT_REFUND:
Problems involving payments, duplicate charges, refunds, or missing refund money.

PRODUCT_DEFECT:
Problems involving damaged, broken, faulty, or defective products.

ACCOUNT_ACCESS:
Problems involving login, password, account access, or account recovery.

FEEDBACK_OTHER:
General feedback, questions, compliments, or issues that do not belong
to the other four categories.

Also assign URGENCY:

HIGH
MEDIUM
LOW

Also assign SENTIMENT:

POSITIVE
NEUTRAL
NEGATIVE

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{
    "category": "...",
    "urgency": "...",
    "sentiment": "..."
}

Do not provide explanations.
Do not use markdown.
Do not echo any order ID from the message.
"""


# ============================================================
# ONE-SHOT EXAMPLE
# ============================================================

ONE_SHOT_EXAMPLE = """
EXAMPLE:

Message:
"Ordered on the 3rd, still not shipped, I need it for a wedding."

Output:
{"category":"DELIVERY_DELAY","urgency":"HIGH","sentiment":"NEGATIVE"}
"""


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:

Message:
"Ordered on the 3rd, still not shipped, I need it for a wedding."

Output:
{"category":"DELIVERY_DELAY","urgency":"HIGH","sentiment":"NEGATIVE"}


EXAMPLE 2:

Message:
"Refund shows credited but nothing in my bank account since 9 days."

Output:
{"category":"PAYMENT_REFUND","urgency":"HIGH","sentiment":"NEGATIVE"}


EXAMPLE 3:

Message:
"The laptop screen is cracked and the keyboard does not work."

Output:
{"category":"PRODUCT_DEFECT","urgency":"HIGH","sentiment":"NEGATIVE"}


EXAMPLE 4:

Message:
"I forgot my password and cannot access my account."

Output:
{"category":"ACCOUNT_ACCESS","urgency":"HIGH","sentiment":"NEGATIVE"}


EXAMPLE 5:

Message:
"Thank you! My order arrived early and everything is perfect."

Output:
{"category":"FEEDBACK_OTHER","urgency":"LOW","sentiment":"POSITIVE"}
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(strategy, message):

    if strategy == "zero-shot":
        examples = ""

    elif strategy == "one-shot":
        examples = ONE_SHOT_EXAMPLE

    elif strategy == "few-shot":
        examples = FEW_SHOT_EXAMPLES

    else:
        examples = ""

    prompt = (
        BASE_INSTRUCTION
        + "\n"
        + examples
        + "\n"
        + "CUSTOMER MESSAGE:\n"
        + message
        + "\n\nOUTPUT:"
    )

    return prompt


# ============================================================
# VALIDATE JSON
# ============================================================

def parse_json(text):

    try:
        data = json.loads(text.strip())

        required_fields = {
            "category",
            "urgency",
            "sentiment"
        }

        if set(data.keys()) != required_fields:
            return None

        valid_categories = {
            "DELIVERY_DELAY",
            "PAYMENT_REFUND",
            "PRODUCT_DEFECT",
            "ACCOUNT_ACCESS",
            "FEEDBACK_OTHER"
        }

        valid_urgency = {
            "HIGH",
            "MEDIUM",
            "LOW"
        }

        valid_sentiment = {
            "POSITIVE",
            "NEUTRAL",
            "NEGATIVE"
        }

        if data["category"] not in valid_categories:
            return None

        if data["urgency"] not in valid_urgency:
            return None

        if data["sentiment"] not in valid_sentiment:
            return None

        return data

    except Exception:
        return None


# ============================================================
# RUN ONE LLM CALL
# ============================================================

def run_call(strategy, message):

    prompt = build_prompt(
        strategy,
        message
    )

    start_time = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            top_p=1.0,
            max_tokens=150
        )

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        output = response.choices[0].message.content

        if output is None:
            output = ""

        prompt_tokens = 0
        completion_tokens = 0

        if response.usage:

            if response.usage.prompt_tokens:
                prompt_tokens = response.usage.prompt_tokens

            if response.usage.completion_tokens:
                completion_tokens = response.usage.completion_tokens

        return {
            "output": output.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency, 2),
            "error": ""
        }

    except Exception as error:

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "output": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": round(latency, 2),
            "error": str(error)
        }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    strategies = [
        "zero-shot",
        "one-shot",
        "few-shot"
    ]

    all_results = []

    print()
    print("=" * 70)
    print("EXERCISE 6")
    print("ZERO-SHOT vs ONE-SHOT vs FEW-SHOT")
    print("=" * 70)

    print()
    print("Test messages       : 15")
    print("Strategies           : 3")
    print("Total API calls      : 45")
    print("Temperature          : 0.2")
    print("Top-p                : 1.0")
    print("Maximum tokens       : 150")
    print()

    for strategy in strategies:

        print()
        print("-" * 70)
        print("CURRENT STRATEGY:", strategy.upper())
        print("-" * 70)

        for number, item in enumerate(
            TEST_DATA,
            start=1
        ):

            print(
                f"Running {strategy}: "
                f"{number}/15"
            )

            result = run_call(
                strategy,
                item["message"]
            )

            parsed = parse_json(
                result["output"]
            )

            if parsed:

                category_correct = int(
                    parsed["category"]
                    == item["category"]
                )

                urgency_correct = int(
                    parsed["urgency"]
                    == item["urgency"]
                )

                sentiment_correct = int(
                    parsed["sentiment"]
                    == item["sentiment"]
                )

            else:

                category_correct = 0
                urgency_correct = 0
                sentiment_correct = 0

            row = {
                "strategy": strategy,
                "test_number": number,
                "message": item["message"],
                "gold_category": item["category"],
                "gold_urgency": item["urgency"],
                "gold_sentiment": item["sentiment"],
                "model_output": result["output"],
                "valid_json": int(
                    parsed is not None
                ),
                "category_correct": category_correct,
                "urgency_correct": urgency_correct,
                "sentiment_correct": sentiment_correct,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
                "error": result["error"]
            }

            all_results.append(row)

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_filename = "exp6_raw_results.csv"

    with open(
        raw_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    summary = []

    for strategy in strategies:

        rows = [
            row
            for row in all_results
            if row["strategy"] == strategy
        ]

        category_score = sum(
            row["category_correct"]
            for row in rows
        )

        urgency_score = sum(
            row["urgency_correct"]
            for row in rows
        )

        sentiment_score = sum(
            row["sentiment_correct"]
            for row in rows
        )

        valid_json_count = sum(
            row["valid_json"]
            for row in rows
        )

        prompt_tokens = sum(
            row["prompt_tokens"]
            for row in rows
        )

        completion_tokens = sum(
            row["completion_tokens"]
            for row in rows
        )

        latency = sum(
            row["latency_ms"]
            for row in rows
        )

        mean_prompt_tokens = (
            prompt_tokens / len(rows)
        )

        mean_completion_tokens = (
            completion_tokens / len(rows)
        )

        mean_latency = (
            latency / len(rows)
        )

        summary.append(
            {
                "strategy": strategy,
                "category_accuracy": (
                    f"{category_score}/15"
                ),
                "urgency_accuracy": (
                    f"{urgency_score}/15"
                ),
                "sentiment_accuracy": (
                    f"{sentiment_score}/15"
                ),
                "valid_json_rate": round(
                    valid_json_count / 15 * 100,
                    2
                ),
                "mean_prompt_tokens": round(
                    mean_prompt_tokens,
                    2
                ),
                "mean_completion_tokens": round(
                    mean_completion_tokens,
                    2
                ),
                "mean_latency_ms": round(
                    mean_latency,
                    2
                )
            }
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_filename = "exp6_summary.csv"

    with open(
        summary_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("EXERCISE 6 FINAL OBSERVATION TABLE")
    print("=" * 100)

    print(
        f"{'Strategy':<15}"
        f"{'Category':<12}"
        f"{'Urgency':<12}"
        f"{'Sentiment':<12}"
        f"{'JSON %':<10}"
        f"{'Prompt Tok':<13}"
        f"{'Completion':<13}"
        f"{'Latency ms':<12}"
    )

    print("-" * 100)

    for row in summary:

        print(
            f"{row['strategy']:<15}"
            f"{row['category_accuracy']:<12}"
            f"{row['urgency_accuracy']:<12}"
            f"{row['sentiment_accuracy']:<12}"
            f"{row['valid_json_rate']:<10}"
            f"{row['mean_prompt_tokens']:<13}"
            f"{row['mean_completion_tokens']:<13}"
            f"{row['mean_latency_ms']:<12}"
        )

    print()
    print("Files created:")
    print("  exp6_raw_results.csv")
    print("  exp6_summary.csv")
    print()
    print("Exercise 6 completed successfully.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()