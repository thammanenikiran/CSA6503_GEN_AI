"""
Unit 4 - Experiment 1: AI Chatbot for Engineering College Queries (Streamlit)
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop an AI chatbot that answers student queries related to an
    engineering college using a pre-trained language model.

SETUP:
    pip install -r requirements.txt

RUN:
    streamlit run exp01_college_chatbot.py

MODEL:
    google/flan-t5-base - an instruction-tuned text2text model that answers a
    question when it is given a context passage (grounded question answering).
"""

import streamlit as st
from transformers import pipeline

MODEL = "google/flan-t5-base"

# ---- College knowledge base ------------------------------------------------
# The chatbot answers ONLY from these facts, so it does not invent details.
COLLEGE_KB = {
    "admission": (
        "Admission to B.E./B.Tech programmes is through the TNEA counselling and "
        "the management quota. Applications open in May and close in July. "
        "Required documents: 10th and 12th marksheets, transfer certificate, "
        "community certificate and 4 passport photos."
    ),
    "courses": (
        "The college offers B.E. in Computer Science, Information Technology, "
        "Artificial Intelligence and Data Science, Electronics and Communication, "
        "Electrical and Electronics, Mechanical and Civil Engineering. "
        "M.E. and Ph.D. programmes are also available."
    ),
    "fees": (
        "The annual tuition fee for B.E. programmes is Rs. 85,000. Hostel fee is "
        "Rs. 60,000 per year including mess charges. Scholarships up to 50 percent "
        "are given for students scoring above 90 percent in 12th standard."
    ),
    "hostel": (
        "Separate hostels are available for boys and girls inside the campus. "
        "Rooms are 3-sharing and air-cooled, with Wi-Fi, laundry, gym and 24x7 "
        "medical support. Hostel gates close at 9 PM."
    ),
    "placement": (
        "The placement cell conducts training from the 5th semester. Recruiters "
        "include TCS, Infosys, Wipro, Cognizant, Zoho and Amazon. The highest "
        "package last year was 24 LPA and the average package was 5.5 LPA with "
        "92 percent of eligible students placed."
    ),
    "exam": (
        "The academic year has two semesters. Internal assessments are held three "
        "times per semester and end-semester exams are in May and December. "
        "A minimum of 75 percent attendance is required to write the exams."
    ),
    "library": (
        "The central library has 45,000 volumes, 120 journals and digital access "
        "to IEEE, Springer and DELNET. It is open from 8 AM to 8 PM on all "
        "working days. Each student can borrow 4 books for 14 days."
    ),
    "transport": (
        "College buses run on 32 routes covering the whole city. The annual bus "
        "fee is Rs. 18,000. Buses reach the campus by 8:30 AM and leave at 4:30 PM."
    ),
}


@st.cache_resource(show_spinner="Loading the pre-trained language model...")
def load_chatbot():
    """Load the model once and reuse it for every question."""
    return pipeline("text2text-generation", model=MODEL)


def find_context(question: str) -> str:
    """Simple keyword retrieval: pick the KB entries related to the question."""
    q = question.lower()
    keywords = {
        "admission": ["admission", "apply", "join", "counsel", "document", "eligib"],
        "courses": ["course", "branch", "department", "programme", "program", "degree"],
        "fees": ["fee", "fees", "cost", "scholarship", "tuition"],
        "hostel": ["hostel", "room", "stay", "accommodation", "mess"],
        "placement": ["placement", "job", "company", "package", "recruit", "salary"],
        "exam": ["exam", "semester", "attendance", "internal", "assessment"],
        "library": ["library", "book", "journal", "ieee", "borrow"],
        "transport": ["bus", "transport", "route", "travel"],
    }
    matched = [topic for topic, words in keywords.items() if any(w in q for w in words)]
    if not matched:
        matched = list(COLLEGE_KB)  # no match -> give the model everything
    return " ".join(COLLEGE_KB[t] for t in matched)


# ---- User interface --------------------------------------------------------
st.set_page_config(page_title="College Enquiry Chatbot", page_icon="🎓")
st.title("🎓 Engineering College Enquiry Chatbot")
st.caption(f"Pre-trained model: {MODEL}")

with st.sidebar:
    st.header("Try asking")
    st.markdown(
        "- What is the admission procedure?\n"
        "- How much is the hostel fee?\n"
        "- Which companies visit for placement?\n"
        "- What are the library timings?\n"
        "- What attendance is needed to write exams?"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

bot = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask about admission, fees, hostel, placement...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    context = find_context(question)
    prompt = (
        "Answer the student's question using only the college information given "
        "below. Reply in two or three complete sentences.\n\n"
        f"College information: {context}\n\nQuestion: {question}\nAnswer:"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = bot(prompt, max_new_tokens=120)[0]["generated_text"].strip()
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
