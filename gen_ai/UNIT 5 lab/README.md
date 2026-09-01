# Unit 5 Lab - Local Large Language Models with Ollama

**SIMATS Engineering** | Name: G.Sai Teja | Reg No: 192472137

Ten experiments that run a Large Language Model **locally** (no cloud API) using
[Ollama](https://ollama.com) as the runtime, driven from Python and Streamlit.
The last two build a local **Retrieval-Augmented Generation (RAG)** system over
the engineering documents in the `docs/` folder.

---

## 1. One-time setup

**a) Install the Ollama runtime** (this is the program that actually runs the model):

- Download from https://ollama.com/download and install.
- Start the service (it listens on `http://localhost:11434`):

```bash
ollama serve
```

**b) Pull the models** (downloads them once):

```bash
ollama pull llama3.2            # chat / generation model (used by all experiments)
ollama pull nomic-embed-text    # embedding model (only needed for Experiment 9)
```

**c) Install the Python packages:**

```bash
pip install -r requirements.txt
```

> To use a different model, edit the `MODEL = "llama3.2"` line at the top of any
> experiment (e.g. `mistral`, `phi3`, `gemma2`).

---

## 2. The experiments

| # | File | Type | Run command |
|---|------|------|-------------|
| 1 | `exp01_streamlit_text_generation.py` | Streamlit web app | `streamlit run exp01_streamlit_text_generation.py` |
| 2 | `exp02_streamlit_summarization.py` | Streamlit web app | `streamlit run exp02_streamlit_summarization.py` |
| 3 | `exp03_question_answering.py` | Python (CLI) | `python exp03_question_answering.py` |
| 4 | `exp04_translation_paraphrasing.py` | Python (CLI) | `python exp04_translation_paraphrasing.py` |
| 5 | `exp05_ollama_text_generation.py` | Python (CLI) | `python exp05_ollama_text_generation.py` |
| 6 | `exp06_ollama_question_answering.py` | Python (CLI) | `python exp06_ollama_question_answering.py` |
| 7 | `exp07_hallucination_demo.py` | Python (CLI) | `python exp07_hallucination_demo.py` |
| 8 | `exp08_prompt_injection_safeguards.py` | Python (CLI) | `python exp08_prompt_injection_safeguards.py` |
| 9 | `exp09_rag_engineering_docs.py` | Python (RAG) | `python exp09_rag_engineering_docs.py` |
| 10 | `exp10_rag_troubleshooting.py` | Python (RAG) | `python exp10_rag_troubleshooting.py` |

### What each one does
1. **Text generation (Streamlit):** prompt box + sliders for temperature, top-p and length; streams the generated text live.
2. **Summarization (Streamlit):** paste text or upload a `.txt`; choose paragraph/bullets and length; shows the compression ratio.
3. **Question answering:** answers are grounded in a supplied context document and it says so when the answer is not present.
4. **Translation & paraphrasing:** menu-driven translation into any language plus style-controlled paraphrasing.
5. **Ollama text generation:** shows one-shot vs token-by-token streaming with `ollama.generate`.
6. **Ollama question answering:** a *conversational* assistant that remembers earlier turns so follow-up questions work.
7. **Hallucinations:** compares ungrounded vs grounded answers on trap prompts and explains why hallucinations happen and how to reduce them.
8. **Prompt injection & safeguards:** a naive bot leaks a secret; a hardened bot blocks the attack with five layered defences (responsible AI).
9. **RAG over engineering docs:** ChromaDB vector database + Ollama `nomic-embed-text` embeddings; answers technical questions from `docs/`.
10. **RAG troubleshooting:** sentence-transformers embeddings + ChromaDB; turns a fault description into numbered step-by-step repair recommendations.

---

## 3. Sample documents

The `docs/` folder holds three short technical guides used by Experiments 9 and 10:

- `centrifugal_pump.md`
- `induction_motor.md`
- `hydraulic_press.md`

Drop your own `.md` or `.txt` files in there and the RAG experiments will index them automatically.

---

## 4. Troubleshooting

- **`Could not reach Ollama` / connection refused:** the runtime is not running. Start it with `ollama serve`.
- **`model not found`:** pull it first, e.g. `ollama pull llama3.2`.
- **Experiment 9 embedding error:** run `ollama pull nomic-embed-text`.
- **Slow first response:** the model loads into memory on the first call; later calls are faster.
- **No GPU:** everything still runs on CPU, just more slowly. Smaller models (e.g. `llama3.2`, `phi3`) respond fastest.
