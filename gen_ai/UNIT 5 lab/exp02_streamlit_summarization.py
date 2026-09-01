"""
Unit 5 - Experiment 2: Text Summarization with a Local LLM (Streamlit)
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop a Streamlit application for text summarization using a locally
    running Large Language Model (served by Ollama).

SETUP:
    pip install streamlit ollama
    ollama serve
    ollama pull llama3.2

RUN:
    streamlit run exp02_streamlit_summarization.py
"""

import streamlit as st
import ollama

MODEL = "llama3.2"

st.set_page_config(page_title="Local LLM Summarizer", page_icon="📝")
st.title("📝 Text Summarization with a Local LLM")
st.caption(f"Powered by Ollama · default model: {MODEL}")

with st.sidebar:
    st.header("Summary settings")
    model = st.text_input("Ollama model", MODEL)
    style = st.radio("Format", ["Paragraph", "Bullet points"])
    length = st.select_slider(
        "Length", ["Very short", "Short", "Medium", "Detailed"], value="Short"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    st.caption("Lower temperature = more faithful, less creative.")


def build_prompt(text: str, style: str, length: str) -> str:
    """Turn the UI choices into a clear instruction for the model."""
    length_map = {
        "Very short": "in one sentence",
        "Short": "in 2-3 sentences",
        "Medium": "in about 5 sentences",
        "Detailed": "in a detailed paragraph of 8-10 sentences",
    }
    fmt = "as concise bullet points" if style == "Bullet points" else "as a flowing paragraph"
    return (
        f"Summarize the following text {length_map[length]}, {fmt}. "
        f"Keep only the key ideas and do NOT add any information that is not "
        f"present in the text.\n\n---\n{text}\n---"
    )


st.write("Paste text or upload a `.txt` file, then click **Summarize**.")

uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
default_text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""
text = st.text_area("Text to summarize", default_text, height=240)

if st.button("Summarize", type="primary"):
    if not text.strip():
        st.warning("Please provide some text to summarize.")
    else:
        try:
            messages = [
                {"role": "system",
                 "content": "You are a precise summarization assistant. You never invent facts."},
                {"role": "user", "content": build_prompt(text, style, length)},
            ]
            st.subheader("Summary")
            placeholder = st.empty()
            summary = ""
            with st.spinner("Summarizing..."):
                stream = ollama.chat(
                    model=model, messages=messages,
                    options={"temperature": temperature}, stream=True,
                )
                for chunk in stream:
                    summary += chunk["message"]["content"]
                    placeholder.markdown(summary + "▌")
            placeholder.markdown(summary)

            in_words, out_words = len(text.split()), len(summary.split())
            ratio = (out_words / in_words * 100) if in_words else 0
            st.info(f"Original: {in_words} words  →  Summary: {out_words} words "
                    f"({ratio:.0f}% of the original length).")
        except Exception as e:
            st.error(
                f"Could not reach Ollama / model '{model}'.\n\n"
                f"Make sure `ollama serve` is running and the model is pulled.\n\n"
                f"Details: {e}"
            )
