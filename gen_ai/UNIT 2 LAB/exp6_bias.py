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
# SAME 15 TEST MESSAGES FROM EXERCISE 6
# ============================================================

TEST_DATA = [
    {
        "message": "My order was supposed to arrive yesterday but the tracking still says it is in transit.",
        "category": "DELIVERY_DELAY"
    },
    {
        "message": "I was charged twice for the same order. Please refund the extra payment.",
        "category": "PAYMENT_REFUND"
    },
    {
        "message": "The headphones I received only work on one side.",
        "category": "PRODUCT_DEFECT"
    },
    {
        "message": "I forgot my password and the reset link is not reaching my email.",
        "category": "ACCOUNT_ACCESS"
    },
    {
        "message": "Great service, my package arrived earlier than expected!",
        "category": "FEEDBACK_OTHER"
    },
    {
        "message": "Where???",
        "category": "DELIVERY_DELAY"
    },
    {
        "message": "Order 784521 is still not shipped and I need it before tomorrow's wedding.",
        "category": "DELIVERY_DELAY"
    },
    {
        "message": "Refund says completed nine days ago but the money is still missing from my bank account.",
        "category": "PAYMENT_REFUND"
    },
    {
        "message": "Superb job guys, because obviously waiting three weeks for a phone is exactly what I wanted.",
        "category": "DELIVERY_DELAY"
    },
    {
        "message": "Bhai payment ho gaya but order confirm nahi hua.",
        "category": "PAYMENT_REFUND"
    },
    {
        "message": "The product looks nice, but the screen has a crack straight out of the box.",
        "category": "PRODUCT_DEFECT"
    },
    {
        "message": "Can you tell me whether this product is available in blue?",
        "category": "FEEDBACK_OTHER"
    },
    {
        "message": "I cannot log into my account after changing my phone number.",
        "category": "ACCOUNT_ACCESS"
    },
    {
        "message": "The refund was processed and I can now see the money in my account. Thanks!",
        "category": "PAYMENT_REFUND"
    },
    {
        "message": "My delivery is late and the box arrived damaged. The actual product seems okay.",
        "category": "DELIVERY_DELAY"
    }
]


# ============================================================
# CATEGORY DEFINITIONS
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

Return ONLY valid JSON.

The JSON must contain exactly:

{
    "category": "CATEGORY_NAME"
}

Do not provide explanations.
Do not use markdown.
"""


# ============================================================
# MAJORITY-LABEL FEW-SHOT EXAMPLES
#
# ALL SIX EXAMPLES ARE PAYMENT_REFUND
# ============================================================

BIAS_EXAMPLES = """
EXAMPLE 1:

Message:
"I was charged twice for my order."

Output:
{"category":"PAYMENT_REFUND"}


EXAMPLE 2:

Message:
"My refund has not reached my bank account."

Output:
{"category":"PAYMENT_REFUND"}


EXAMPLE 3:

Message:
"The payment was deducted but my order was not confirmed."

Output:
{"category":"PAYMENT_REFUND"}


EXAMPLE 4:

Message:
"I need a refund because I was charged for the same item twice."

Output:
{"category":"PAYMENT_REFUND"}


EXAMPLE 5:

Message:
"My refund was approved but I still haven't received the money."

Output:
{"category":"PAYMENT_REFUND"}


EXAMPLE 6:

Message:
"I paid for my order but there is a problem with the payment."

Output:
{"category":"PAYMENT_REFUND"}
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(message):

    return (
        BASE_INSTRUCTION
        + "\n"
        + BIAS_EXAMPLES
        + "\n"
        + "CUSTOMER MESSAGE:\n"
        + message
        + "\n\nOUTPUT:"
    )


# ============================================================
# PARSE CATEGORY
#
# Returns:
#     category
#     valid_json
# ============================================================

def parse_category(text):

    valid_categories = [
        "DELIVERY_DELAY",
        "PAYMENT_REFUND",
        "PRODUCT_DEFECT",
        "ACCOUNT_ACCESS",
        "FEEDBACK_OTHER"
    ]

    # --------------------------------------------------------
    # First try strict JSON
    # --------------------------------------------------------

    try:

        data = json.loads(text.strip())

        category = data.get("category")

        if category in valid_categories:
            return category, True

    except Exception:
        pass

    # --------------------------------------------------------
    # If JSON is malformed, extract category from text.
    #
    # This lets us measure:
    # 1. JSON validity separately
    # 2. Classification separately
    # --------------------------------------------------------

    text_upper = text.upper()

    for category in valid_categories:

        if category in text_upper:
            return category, False

    # --------------------------------------------------------
    # Could not identify any category
    # --------------------------------------------------------

    return None, False


# ============================================================
# CALL OPENROUTER
# ============================================================

def classify(message):

    prompt = build_prompt(message)

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
            max_tokens=100
        )

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        output = response.choices[0].message.content

        if output is None:
            output = ""

        predicted_category, valid_json = parse_category(
            output
        )

        return {
            "output": output.strip(),
            "predicted": predicted_category,
            "valid_json": valid_json,
            "latency_ms": round(latency, 2),
            "error": ""
        }

    except Exception as error:

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "output": "",
            "predicted": None,
            "valid_json": False,
            "latency_ms": round(latency, 2),
            "error": str(error)
        }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()
    print("=" * 75)
    print("EXERCISE 6 - MAJORITY-LABEL BIAS EXPERIMENT")
    print("=" * 75)

    print()
    print("Prompt strategy     : Biased Few-shot")
    print("Number of examples  : 6")
    print("Example category    : PAYMENT_REFUND")
    print("Test messages       : 15")
    print("Temperature         : 0.2")
    print("Top-p               : 1.0")
    print("Maximum tokens      : 100")
    print()

    results = []

    # --------------------------------------------------------
    # RUN ALL 15 TEST MESSAGES
    # --------------------------------------------------------

    for number, item in enumerate(
        TEST_DATA,
        start=1
    ):

        print(
            f"Running biased few-shot: "
            f"{number}/15"
        )

        result = classify(
            item["message"]
        )

        results.append(
            {
                "test_number": number,
                "message": item["message"],
                "gold_category": item["category"],
                "predicted_category": (
                    result["predicted"]
                    if result["predicted"]
                    else "UNIDENTIFIED"
                ),
                "valid_json": int(
                    result["valid_json"]
                ),
                "latency_ms": result["latency_ms"],
                "raw_output": result["output"],
                "error": result["error"]
            }
        )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    with open(
        "exp6_bias_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # CATEGORY LIST
    # ========================================================

    categories = [
        "DELIVERY_DELAY",
        "PAYMENT_REFUND",
        "PRODUCT_DEFECT",
        "ACCOUNT_ACCESS",
        "FEEDBACK_OTHER"
    ]

    # ========================================================
    # CREATE CONFUSION MATRIX
    #
    # Rows    = actual category
    # Columns = predicted category
    # ========================================================

    matrix = {}

    for actual in categories:

        matrix[actual] = {}

        for predicted in categories:

            matrix[actual][predicted] = 0

    for row in results:

        actual = row["gold_category"]
        predicted = row["predicted_category"]

        if predicted in categories:

            matrix[actual][predicted] += 1

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    with open(
        "exp6_confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Actual",
            "DELIVERY_DELAY",
            "PAYMENT_REFUND",
            "PRODUCT_DEFECT",
            "ACCOUNT_ACCESS",
            "FEEDBACK_OTHER"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for actual in categories:

            row = {
                "Actual": actual
            }

            for predicted in categories:

                row[predicted] = matrix[
                    actual
                ][predicted]

            writer.writerow(row)

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    prediction_counts = {}

    for category in categories:
        prediction_counts[category] = 0

    unidentified_count = 0

    for row in results:

        predicted = row["predicted_category"]

        if predicted in categories:

            prediction_counts[predicted] += 1

        else:

            unidentified_count += 1

    # ========================================================
    # VALID JSON RATE
    # ========================================================

    valid_json_count = sum(
        row["valid_json"]
        for row in results
    )

    valid_json_rate = (
        valid_json_count / 15
    ) * 100

    # ========================================================
    # PRINT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 75)
    print("PREDICTED CATEGORY DISTRIBUTION")
    print("=" * 75)

    for category in categories:

        count = prediction_counts[category]

        percentage = (
            count / 15
        ) * 100

        print(
            f"{category:<20}"
            f"{count:>3}/15"
            f"  ({percentage:.2f}%)"
        )

    print(
        f"{'UNIDENTIFIED':<20}"
        f"{unidentified_count:>3}/15"
    )

    print()
    print(
        f"Valid JSON responses: "
        f"{valid_json_count}/15 "
        f"({valid_json_rate:.2f}%)"
    )

    # ========================================================
    # PRINT CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 100)
    print("CONFUSION MATRIX")
    print("=" * 100)

    print(
        f"{'Actual':<20}"
        f"{'DELIVERY':<12}"
        f"{'PAYMENT':<12}"
        f"{'DEFECT':<12}"
        f"{'ACCOUNT':<12}"
        f"{'OTHER':<12}"
    )

    print("-" * 100)

    for actual in categories:

        print(
            f"{actual:<20}"
            f"{matrix[actual]['DELIVERY_DELAY']:<12}"
            f"{matrix[actual]['PAYMENT_REFUND']:<12}"
            f"{matrix[actual]['PRODUCT_DEFECT']:<12}"
            f"{matrix[actual]['ACCOUNT_ACCESS']:<12}"
            f"{matrix[actual]['FEEDBACK_OTHER']:<12}"
        )

    # ========================================================
    # MAJORITY-LABEL BIAS
    # ========================================================

    payment_predictions = prediction_counts[
        "PAYMENT_REFUND"
    ]

    payment_prediction_percentage = (
        payment_predictions / 15
    ) * 100

    actual_payment = sum(
        1
        for item in TEST_DATA
        if item["category"] == "PAYMENT_REFUND"
    )

    actual_payment_percentage = (
        actual_payment / 15
    ) * 100

    shift = (
        payment_prediction_percentage
        - actual_payment_percentage
    )

    # ========================================================
    # CATEGORY ACCURACY
    # ========================================================

    correct = sum(
        1
        for row in results
        if (
            row["predicted_category"]
            == row["gold_category"]
        )
    )

    accuracy = (
        correct / 15
    ) * 100

    # ========================================================
    # PRINT BIAS ANALYSIS
    # ========================================================

    print()
    print("=" * 75)
    print("MAJORITY-LABEL BIAS ANALYSIS")
    print("=" * 75)

    print(
        f"Actual PAYMENT_REFUND messages : "
        f"{actual_payment}/15 "
        f"({actual_payment_percentage:.2f}%)"
    )

    print(
        f"Predicted PAYMENT_REFUND        : "
        f"{payment_predictions}/15 "
        f"({payment_prediction_percentage:.2f}%)"
    )

    print(
        f"Shift toward PAYMENT_REFUND     : "
        f"{shift:+.2f} percentage points"
    )

    print()
    print(
        f"Category accuracy               : "
        f"{correct}/15 "
        f"({accuracy:.2f}%)"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        "exp6_bias_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Metric",
                "Value"
            ]
        )

        writer.writerow(
            [
                "Actual PAYMENT_REFUND",
                f"{actual_payment}/15"
            ]
        )

        writer.writerow(
            [
                "Actual PAYMENT_REFUND %",
                f"{actual_payment_percentage:.2f}%"
            ]
        )

        writer.writerow(
            [
                "Predicted PAYMENT_REFUND",
                f"{payment_predictions}/15"
            ]
        )

        writer.writerow(
            [
                "Predicted PAYMENT_REFUND %",
                f"{payment_prediction_percentage:.2f}%"
            ]
        )

        writer.writerow(
            [
                "Shift",
                f"{shift:+.2f} percentage points"
            ]
        )

        writer.writerow(
            [
                "Category accuracy",
                f"{correct}/15"
            ]
        )

        writer.writerow(
            [
                "Category accuracy %",
                f"{accuracy:.2f}%"
            ]
        )

        writer.writerow(
            [
                "Valid JSON",
                f"{valid_json_count}/15"
            ]
        )

        writer.writerow(
            [
                "Valid JSON rate",
                f"{valid_json_rate:.2f}%"
            ]
        )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print("exp6_bias_results.csv")
    print("exp6_confusion_matrix.csv")
    print("exp6_bias_summary.csv")

    print()
    print("Majority-label bias experiment completed.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()