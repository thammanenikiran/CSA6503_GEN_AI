import os
import csv
import time
from openai import OpenAI


# ============================================================
# OPENROUTER
# ============================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY is not set.")
    raise SystemExit

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

MODEL = "openrouter/free"


# ============================================================
# SOURCE - 7A
# ============================================================

TRANSCRIPT = """
PRODUCTION REVIEW MEETING TRANSCRIPT

Meeting date: 18 September

Participants:
Arun - Production Manager
Meera - Quality Manager
Vikram - Maintenance Manager
Priya - Supply Chain Manager
Daniel - Operations Director

Plant A completed 18,450 units against a monthly target of 19,000 units,
a shortfall of 550 units or approximately 2.9 percent.

The main production constraint was unplanned downtime on Press Line 3.
The line experienced 17.5 hours of unplanned downtime compared with
9 hours in the previous month.

Press Line 3 had three significant stoppages:
4.5 hours due to hydraulic pressure failure,
6 hours due to a worn drive belt and damaged tensioner,
and 7 hours because a temperature sensor failed.

The hydraulic pump had been in service for 31 months and exceeded the
manufacturer's recommended inspection interval of 24 months.

The plant recorded 126 defective units compared with 91 defects previously.
The overall defect rate increased from 0.48 percent to 0.68 percent.

Surface scratching was the largest defect category with 47 units.
Dimensional variation accounted for 31 units, incorrect hole positioning
22 units, and other defects 26 units.

29 of the 47 surface-scratch defects originated from Machine M-14.
A new cutting tool was installed on 5 September, but its alignment was
not verified after installation.

Northern Metals delivered aluminum sheet between two and four days late
on three occasions. Current inventory was sufficient for approximately
eight production days.

Priya proposed increasing safety stock from six days to ten days.
The estimated additional working capital requirement was approximately
14,000 dollars.

The plant used 284,000 kWh of electricity during the month, 6 percent
higher than the previous month.

On-time delivery was 94.2 percent compared with an internal target of
97 percent.

18 delayed orders were linked to production downtime, 11 to late
material deliveries, and the remaining delays to transport capacity.

Arun proposed a daily production recovery meeting at 4:30 PM from
23 September.

Meera recommended first-piece inspection for Machine M-14 after every
tool change and a weekly review of the top three defect categories.

Vikram committed to inspect the hydraulic system by 20 September,
replace the drive belt and tensioner by 22 September, and replace the
temperature sensor by 21 September.

Priya agreed to obtain a confirmed Northern Metals delivery schedule
by 19 September and prepare a safety-stock proposal.

Arun would distribute a production dashboard covering output, downtime,
defects, material shortages and delivery status.

Vikram and Priya would prepare a business case for replacing two old
motors by 30 September.

The business case must include purchase cost, expected electricity
savings, payback period and maintenance benefits.

The next production meeting would be on 2 October.
"""


# ============================================================
# SOURCE - 7C
# ============================================================

PRODUCT_DATASHEET = """
Product name: EcoTorque E200 Industrial Motor

Product type: Three-phase industrial motor

Rated power: 30 kW

Efficiency: 94.5%

Applications:
Industrial pumps, conveyors and material-handling systems

Voltage: 415 V

Frequency: 50 Hz

Protection rating: IP55

Primary benefit:
Reduced electricity consumption compared with standard-efficiency
motors operating under comparable conditions.

Warranty: 3 years

Target customers:
Manufacturing plants and industrial facilities seeking improved motor
efficiency and lower operating costs.
"""


# ============================================================
# LLM FUNCTION
# ============================================================

def call_llm(prompt, max_tokens=500):

    start = time.perf_counter()

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
            max_tokens=max_tokens
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        text = ""

        if response.choices:
            text = (
                response.choices[0]
                .message.content
                or ""
            )

        input_tokens = 0
        output_tokens = 0

        if response.usage:

            input_tokens = (
                response.usage.prompt_tokens or 0
            )

            output_tokens = (
                response.usage.completion_tokens or 0
            )

        return {
            "text": text.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency, 2),
            "error": ""
        }

    except Exception as e:

        return {
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "error": str(e)
        }


# ============================================================
# RETRY
# ============================================================

def generate(prompt, max_tokens=500):

    for attempt in range(1, 3):

        print(
            "LLM request:",
            attempt,
            "/ 2"
        )

        result = call_llm(
            prompt,
            max_tokens
        )

        if result["text"]:
            return result

        if result["error"]:
            print(
                "API error:",
                result["error"]
            )

            # Do not repeatedly hammer a rate-limited API.
            if "429" in result["error"]:
                break

    return result


# ============================================================
# SAVE
# ============================================================

def save_text(filename, text):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)


# ============================================================
# 7A
# ============================================================

def run_7a():

    print()
    print("=" * 70)
    print("7A - SUMMARIZATION")
    print("=" * 70)

    results = []

    prompts = [

        (
            "Executive abstract",
            f"""
Using ONLY the source below, write an executive abstract.

Requirements:
- Maximum 80 words.
- Audience: Managing Director.
- Business-focused.
- Preserve important numerical facts.
- Do not invent facts.
- Return ONLY the abstract.

SOURCE:
{TRANSCRIPT}
""",
            "exp7_7A_executive_abstract.txt",
            150
        ),

        (
            "Action-item list",
            f"""
Using ONLY the source below, create an action-item list.

Every item must contain:

Owner:
Task:
Deadline:

Preserve the actual owners, tasks and deadlines.
Do not invent information.
Return ONLY the action-item list.

SOURCE:
{TRANSCRIPT}
""",
            "exp7_7A_action_items.txt",
            300
        ),

        (
            "Technical summary",
            f"""
Using ONLY the source below, write a technical summary.

Preserve:
- machine names
- defect counts
- production quantities
- downtime
- dates
- energy figures
- percentages
- equipment details

Do not invent facts.
Return ONLY the technical summary.

SOURCE:
{TRANSCRIPT}
""",
            "exp7_7A_technical_summary.txt",
            500
        )
    ]

    for task, prompt, filename, tokens in prompts:

        print("Generating:", task)

        result = generate(
            prompt,
            tokens
        )

        save_text(
            filename,
            result["text"]
        )

        results.append({
            "task": task,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "compression_ratio": round(
                result["output_tokens"] /
                result["input_tokens"],
                4
            ) if result["input_tokens"] else 0,
            "latency_ms": result["latency_ms"],
            "error": result["error"]
        })

    with open(
        "exp7_7A_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# 7B
# ============================================================

def run_7b():

    print()
    print("=" * 70)
    print("7B - PROFESSIONAL EMAIL")
    print("=" * 70)

    base = """
You are a senior account manager at a precision-components manufacturer.

Client: Meridian Auto
Account age: 7 years
PO: PO-4471
Quantity: 400 units
Original delivery date: 12 Sep
Revised delivery date: 21 Sep
Cause: sub-supplier casting failure

Requirements:
- Maximum 150 words.
- State 21 Sep explicitly.
- Offer exactly ONE remedy: expedited freight.
- Do not admit legal liability.
- Do not mention penalty clauses.
- Do not apologise more than twice.
- Do not invent names, phone numbers, emails or companies.
- Do not use placeholders.
- Include Subject line.
- End with:

Regards,
Senior Account Manager

Return ONLY the email.
"""

    tones = {
        "formal": "Use a formal and professional tone.",
        "empathetic": "Use a professional and empathetic tone.",
        "assertive": "Use a professional, confident and direct tone."
    }

    results = []

    for name, tone in tones.items():

        print("Generating:", name)

        prompt = (
            base
            + "\n\n"
            + tone
        )

        result = generate(
            prompt,
            300
        )

        filename = (
            "exp7_7B_email_"
            + name
            + ".txt"
        )

        save_text(
            filename,
            result["text"]
        )

        results.append({
            "task": name,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
            "error": result["error"]
        })

    with open(
        "exp7_7B_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# 7C
# ============================================================

def run_7c():

    print()
    print("=" * 70)
    print("7C - PRODUCT CAMPAIGN")
    print("=" * 70)

    results = []

    # --------------------------------------------------------
    # LINKEDIN
    # --------------------------------------------------------

    linkedin_prompt = f"""
You are a B2B industrial marketing specialist.

PRODUCT DATASHEET:
{PRODUCT_DATASHEET}

Write ONLY a LinkedIn launch post.

Requirements:
- 120 to 150 words.
- Professional B2B tone.
- Mention 30 kW.
- Mention 94.5% efficiency.
- Mention industrial pumps, conveyors and material-handling systems.
- Mention 415 V.
- Mention IP55.
- Mention 3-year warranty.
- Naturally include:
  energy-efficient industrial motor
- Do not invent prices.
- Do not invent certifications.
- Do not invent exact savings percentages.
- Do not invent customer results.
- Use ONLY the datasheet.
"""

    linkedin = generate(
        linkedin_prompt,
        250
    )

    save_text(
        "exp7_7C_linkedin.txt",
        linkedin["text"]
    )

    results.append({
        "content": "LinkedIn",
        "input_tokens": linkedin["input_tokens"],
        "output_tokens": linkedin["output_tokens"],
        "latency_ms": linkedin["latency_ms"],
        "error": linkedin["error"]
    })

    # --------------------------------------------------------
    # INSTAGRAM
    # --------------------------------------------------------

    instagram_prompt = f"""
You are creating social media content.

PRODUCT DATASHEET:
{PRODUCT_DATASHEET}

Write ONLY an Instagram caption.

Requirements:
- Maximum 40 words.
- Engaging.
- Mention EcoTorque E200.
- Mention 30 kW or 94.5% efficiency.
- Include hashtags.
- Use ONLY the datasheet.
- Do not invent price.
- Do not invent certifications.
- Do not invent savings percentages.
"""

    instagram = generate(
        instagram_prompt,
        100
    )

    save_text(
        "exp7_7C_instagram.txt",
        instagram["text"]
    )

    results.append({
        "content": "Instagram",
        "input_tokens": instagram["input_tokens"],
        "output_tokens": instagram["output_tokens"],
        "latency_ms": instagram["latency_ms"],
        "error": instagram["error"]
    })

    # --------------------------------------------------------
    # WEBSITE
    # --------------------------------------------------------

    website_prompt = f"""
You are writing a B2B product website description.

PRODUCT DATASHEET:
{PRODUCT_DATASHEET}

Write ONLY the website product blurb.

Requirements:
- Exactly 60 words.
- Professional B2B tone.
- Include these three phrases naturally:

energy-efficient industrial motor
high-efficiency motor
industrial energy savings

- Mention 30 kW.
- Mention 94.5% efficiency.
- Mention industrial applications.
- Use ONLY the datasheet.
- Do not invent price.
- Do not invent certifications.
- Do not invent exact savings percentages.
"""

    website = generate(
        website_prompt,
        120
    )

    save_text(
        "exp7_7C_website.txt",
        website["text"]
    )

    results.append({
        "content": "Website",
        "input_tokens": website["input_tokens"],
        "output_tokens": website["output_tokens"],
        "latency_ms": website["latency_ms"],
        "error": website["error"]
    })

    # --------------------------------------------------------
    # COMBINED FILE
    # --------------------------------------------------------

    combined = (
        "LINKEDIN:\n\n"
        + linkedin["text"]
        + "\n\n"
        + "=" * 70
        + "\n\n"
        + "INSTAGRAM:\n\n"
        + instagram["text"]
        + "\n\n"
        + "=" * 70
        + "\n\n"
        + "WEBSITE:\n\n"
        + website["text"]
    )

    save_text(
        "exp7_7C_product_campaign.txt",
        combined
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    linkedin_words = len(
        linkedin["text"].split()
    )

    instagram_words = len(
        instagram["text"].split()
    )

    website_words = len(
        website["text"].split()
    )

    with open(
        "exp7_7C_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Content",
            "Word Count",
            "Requirement",
            "Status"
        ])

        writer.writerow([
            "LinkedIn",
            linkedin_words,
            "120-150 words",
            "PASS"
            if 120 <= linkedin_words <= 150
            else "FAIL"
        ])

        writer.writerow([
            "Instagram",
            instagram_words,
            "Maximum 40 words",
            "PASS"
            if instagram_words <= 40
            else "FAIL"
        ])

        writer.writerow([
            "Website",
            website_words,
            "Exactly 60 words",
            "PASS"
            if website_words == 60
            else "FAIL"
        ])

    print()
    print("LinkedIn words:", linkedin_words)
    print("Instagram words:", instagram_words)
    print("Website words:", website_words)


# ============================================================
# ABLATION
# ============================================================

def run_ablation():

    print()
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    base_context = """
Client: Meridian Auto
Account age: 7 years
PO: PO-4471
Quantity: 400 units
Original date: 12 Sep
Revised date: 21 Sep
Cause: sub-supplier casting failure
"""

    variants = {

        "Role": """
Write the client delay email.
Maximum 150 words.
State revised date.
Offer expedited freight as exactly one remedy.
Do not admit legal liability.
""",

        "Context": """
You are a senior account manager.
Write a professional client delay notification.
Maximum 150 words.
Offer expedited freight.
""",

        "Tone": """
Write a client delay notification.
Use a professional, accountable, confident tone.
Maximum 150 words.
Offer expedited freight.
""",

        "Word-count constraint": """
Write a professional client delay notification.
State the revised date and offer expedited freight.
""",

        "Output-format specification": """
Write the client delay notification.
Maximum 150 words.
State revised date.
Offer expedited freight.
Return a Subject line followed by the email body.
"""
    }

    results = []

    for name, instructions in variants.items():

        print("Removing:", name)

        prompt = (
            base_context
            + "\n"
            + instructions
            + "\nReturn ONLY the email."
        )

        result = generate(
            prompt,
            300
        )

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
            .replace("-", "")
        )

        filename = (
            "exp7_ablation_"
            + safe_name
            + ".txt"
        )

        save_text(
            filename,
            result["text"]
        )

        results.append({
            "component_removed": name,
            "output_file": filename,
            "latency_ms": result["latency_ms"],
            "error": result["error"]
        })

    with open(
        "exp7_ablation_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "component_removed",
                "output_file",
                "latency_ms",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EXERCISE 7")
    print("=" * 70)

    run_7a()

    run_7b()

    run_7c()

    run_ablation()

    print()
    print("=" * 70)
    print("EXERCISE 7 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()