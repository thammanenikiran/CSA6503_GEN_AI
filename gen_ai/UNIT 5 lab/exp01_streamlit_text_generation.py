"""
Unit 5 - Experiment 1: Text Generation with a Local LLM (Streamlit)
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop a Streamlit application for text generation using a locally
    running Large Language Model (served by Ollama).

SETUP:
    pip install streamlit ollama
    ollama serve                 # start the local runtime (http://localhost:11434)
    ollama pull llama3.2         # download the model once

RUN:
    streamlit run exp01_streamlit_text_generation.py
"""

import streamlit as st
import ollama

MODEL = "llama3.2"

st.set_page_config(page_title="Local LLM Text Generation", page_icon="✍️")
st.title("✍️ Text Generation with a Local LLM")
st.caption(f"Powered by Ollama · default model: {MODEL}")

# ---- Sidebar: generation controls -----------------------------------------
with st.sidebar:
    st.header("Generation settings")
    model = st.text_input("Ollama model", MODEL)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.1,
                            help="Higher = more creative/random.")
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max new tokens", 32, 1024, 256, 32)
    st.markdown("---")
    st.markdown("**Before running**\n\n1. `ollama serve`\n2. `ollama pull llama3.2`")

# ---- Main: prompt and output ----------------------------------------------
prompt = st.text_area(
    "Enter your prompt",
    "Write a short motivational paragraph about learning to code.",
    height=150,
)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        try:
            st.subheader("Generated text")
            placeholder = st.empty()
            output = ""
            with st.spinner("Generating with the local model..."):
                # stream=True yields the text token-by-token as the model writes it
                stream = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                    },
                    stream=True,
                )
                for chunk in stream:
                    output += chunk.get("response", "")
                    placeholder.markdown(output + "▌")   # ▌ = live typing cursor
            placeholder.markdown(output)
            st.download_button("Download as .txt", output, file_name="generated.txt")
        except Exception as e:
            st.error(
                f"Could not reach Ollama / model '{model}'.\n\n"
                f"Make sure `ollama serve` is running and the model is pulled "
                f"(`ollama pull {model}`).\n\nDetails: {e}"
            )
