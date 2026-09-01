"""
Unit 4 - Experiment 2: Engineering-Support Chatbot (NLP + Pre-trained LLM)
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Design an engineering-support chatbot that can answer technical questions
    and provide relevant solutions using NLP techniques.

METHOD:
    1. NLP retrieval  : TF-IDF vectorisation + cosine similarity picks the most
                        relevant fault-and-solution record from a support KB.
    2. Generation     : google/flan-t5-base rewrites that record into a natural,
                        conversational answer for the user.

SETUP:
    pip install -r requirements.txt

RUN:
    python exp02_engineering_support_chatbot.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

MODEL = "google/flan-t5-base"
SIMILARITY_THRESHOLD = 0.10  # below this we treat the query as "not in the KB"

# ---- Technical support knowledge base --------------------------------------
SUPPORT_KB = [
    {
        "problem": "Three phase induction motor overheating and tripping the overload relay",
        "solution": "Check for single phasing and voltage imbalance above 2 percent, "
                    "verify the motor is not overloaded beyond its rated current, clean "
                    "the cooling fan and fins, and confirm the ambient temperature is "
                    "below 40 C. Re-grease bearings if they run hot.",
    },
    {
        "problem": "Centrifugal pump cavitation with noise and vibration",
        "solution": "Cavitation occurs when the available NPSH falls below the required "
                    "NPSH. Raise the suction liquid level, shorten and enlarge the "
                    "suction pipe, remove suction strainer blockage, and throttle the "
                    "discharge valve instead of the suction valve.",
    },
    {
        "problem": "Hydraulic press loses pressure and the ram creeps down",
        "solution": "Inspect the cylinder piston seals for internal leakage, test the "
                    "pressure relief valve setting, check the pilot operated check valve "
                    "for contamination, and top up the hydraulic oil to the correct level.",
    },
    {
        "problem": "Concrete beam develops vertical cracks at midspan",
        "solution": "Vertical midspan cracks indicate flexural overstress. Verify the "
                    "actual loading against the design load, check the tension steel area "
                    "and cover, and strengthen using externally bonded FRP laminates or "
                    "a steel plate after a structural audit.",
    },
    {
        "problem": "Arduino microcontroller keeps resetting randomly during operation",
        "solution": "Random resets usually mean an unstable supply. Add a 100 uF bulk "
                    "capacitor and 0.1 uF decoupling capacitor near the supply pins, avoid "
                    "drawing motor current through the board regulator, and check for "
                    "stack overflow caused by large local arrays or a watchdog timeout.",
    },
    {
        "problem": "Wi-Fi network has high latency and frequent packet loss",
        "solution": "Scan for channel overlap and move to a clear channel, reduce the "
                    "number of clients per access point, check for interference from "
                    "microwave ovens and Bluetooth, and update the access point firmware. "
                    "Prefer the 5 GHz band for dense environments.",
    },
    {
        "problem": "Database query became very slow after the table grew large",
        "solution": "Run the query plan to find full table scans, add a composite index on "
                    "the filter and join columns, avoid functions on indexed columns in the "
                    "WHERE clause, update table statistics, and paginate large result sets.",
    },
    {
        "problem": "Diesel generator emits black smoke and gives low output power",
        "solution": "Black smoke means incomplete combustion. Clean or replace the air "
                    "filter, service the fuel injectors for correct spray pattern, check "
                    "the turbocharger boost pressure, and verify the injection timing and "
                    "fuel quality.",
    },
    {
        "problem": "Welded joint fails at the heat affected zone",
        "solution": "Failure at the heat affected zone indicates excessive heat input or a "
                    "hardened brittle structure. Reduce current and travel speed, preheat "
                    "thick sections, use low hydrogen electrodes stored dry, and apply post "
                    "weld heat treatment.",
    },
    {
        "problem": "Solar PV plant output has dropped compared to last year",
        "solution": "Check module soiling and clean the panels, look for hotspots and "
                    "micro-cracks with a thermal camera, measure string open circuit voltage "
                    "against the datasheet, inspect connectors for corrosion, and confirm the "
                    "inverter is not derating due to high temperature.",
    },
]


def build_retriever():
    """Fit a TF-IDF model over the KB text (problem + solution)."""
    corpus = [f"{item['problem']} {item['solution']}" for item in SUPPORT_KB]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix):
    """Return (best KB record, similarity score) for the user's question."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    best = scores.argmax()
    return SUPPORT_KB[best], float(scores[best])


def main():
    print("Loading pre-trained model, please wait...")
    generator = pipeline("text2text-generation", model=MODEL)
    vectorizer, matrix = build_retriever()

    print("\n=== Engineering Support Chatbot ===")
    print("Describe a technical problem. Type 'quit' to exit.\n")
    print("Example: my induction motor is getting very hot and tripping\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit", "q", ""}:
            print("Bot: Goodbye, happy engineering!")
            break

        record, score = retrieve(query, vectorizer, matrix)

        if score < SIMILARITY_THRESHOLD:
            print("Bot: I do not have a documented solution for that. Please "
                  "rephrase or contact the technical support desk.\n")
            continue

        prompt = (
            "You are an engineering support assistant. Using the reference "
            "solution below, reply to the user politely in 3 to 4 sentences with "
            "clear troubleshooting steps.\n\n"
            f"Known problem: {record['problem']}\n"
            f"Reference solution: {record['solution']}\n\n"
            f"User question: {query}\nAssistant:"
        )
        answer = generator(prompt, max_new_tokens=160)[0]["generated_text"].strip()

        print(f"\n[matched: {record['problem']}  |  similarity: {score:.2f}]")
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
