# Unit 4 Lab - Multimodal Generative AI Applications

**SIMATS Engineering** | Name: G.Sai Teja | Reg No: 192472137

Ten experiments that use **pre-trained Hugging Face models** to build multimodal
generative AI applications: chatbots, text-to-image, speech-to-text,
text-to-speech, summarisation, translation, resume screening and a research
assistant. Every model is downloaded automatically on first run and cached
locally, so no API key and no paid cloud service is required.

---

## 1. Setup

```bash
pip install -r requirements.txt
```

Optional but recommended: a CUDA GPU. Everything also runs on CPU, only slower.
The first run of each experiment downloads its model (a few hundred MB to
~2.5 GB) into `~/.cache/huggingface`.

---

## 2. The experiments

| # | File | Task | Model used | Run command |
|---|------|------|------------|-------------|
| 1 | `exp01_college_chatbot.py` | College enquiry chatbot | flan-t5-base | `streamlit run exp01_college_chatbot.py` |
| 2 | `exp02_engineering_support_chatbot.py` | Technical support chatbot (TF-IDF + LLM) | flan-t5-base | `python exp02_engineering_support_chatbot.py` |
| 3 | `exp03_text_to_image.py` | Text-to-image (bridge / robot) | sd-turbo | `python exp03_text_to_image.py` |
| 4 | `exp04_prompt_comparison.py` | Prompt comparison grid | sd-turbo | `python exp04_prompt_comparison.py` |
| 5 | `exp05_speech_to_text.py` | Speech-to-text query | whisper-small | `python exp05_speech_to_text.py` |
| 6 | `exp06_text_to_speech.py` | Text-to-speech | speecht5_tts | `python exp06_text_to_speech.py` |
| 7 | `exp07_document_summarizer.py` | Document summarisation | bart-large-cnn | `python exp07_document_summarizer.py` |
| 8 | `exp08_machine_translation.py` | English to Indian language | nllb-200-distilled-600M | `python exp08_machine_translation.py tamil` |
| 9 | `exp09_resume_screening.py` | Resume ranking vs job description | all-MiniLM-L6-v2 | `python exp09_resume_screening.py` |
| 10 | `exp10_research_assistant.py` | Research assistant | flan-t5-base + bart | `python exp10_research_assistant.py` |

---

## 3. Folders

| Folder | Contents |
|--------|----------|
| `docs/` | Long engineering documents used by Exp 7 (summarisation) and Exp 8 (translation) |
| `resumes/` | Five sample candidate resumes used by Exp 9 |
| `outputs/` | Everything the experiments generate: images, wav files, summaries, reports |

---

## 4. Notes on each experiment

1. **College chatbot** - a small knowledge base of admission, fees, hostel,
   placement, exam, library and transport facts. Keyword retrieval selects the
   relevant facts and the language model turns them into an answer, so the bot
   stays grounded instead of hallucinating.
2. **Engineering support chatbot** - the NLP part is TF-IDF vectorisation plus
   cosine similarity over ten documented fault-and-solution records; the LLM part
   rewrites the matched record as a conversational reply. If similarity is below
   the threshold the bot says it does not know.
3. **Text-to-image** - `sd-turbo` needs only 4 denoising steps and no classifier
   free guidance, which is why it is usable without a GPU. Pass your own prompt
   as a command line argument.
4. **Prompt comparison** - the same seed is used for all four prompts, so every
   visible difference is caused by the wording alone. The script saves the four
   images and a labelled 2x2 comparison sheet.
5. **Speech-to-text** - run with no arguments to record 8 seconds from the
   microphone, or pass an audio file path to transcribe an existing recording.
6. **Text-to-speech** - SpeechT5 plus the HiFi-GAN vocoder and a CMU-Arctic
   speaker embedding. Long text is split into sentences and the audio is joined.
7. **Summarisation** - map-reduce summarisation: the document is chunked, each
   chunk is summarised, and the chunk summaries are summarised again.
8. **Translation** - NLLB-200 supports Tamil, Hindi, Telugu, Kannada, Malayalam,
   Bengali, Marathi and Gujarati. Translation is done sentence by sentence.
9. **Resume screening** - final score = 70% semantic similarity between the
   resume and the job description + 30% required-skill coverage, then the
   candidates are ranked and labelled SHORTLIST / MAYBE / REJECT.
10. **Research assistant** - generates definition, working principle,
    applications, advantages and challenges for any topic, extracts TF-IDF
    keywords, writes a concise summary and saves a markdown report.

---

## 5. Troubleshooting

| Problem | Fix |
|---------|-----|
| Model download is slow | It happens once per model; afterwards it loads from cache |
| `OutOfMemoryError` on GPU | Run on CPU: set `CUDA_VISIBLE_DEVICES=""` before the command |
| No microphone in Exp 5 | Pass an audio file path instead: `python exp05_speech_to_text.py query.wav` |
| No sound in Exp 6 | The wav file is still written to `outputs/exp06_speech.wav` |
| `sentencepiece` build error | Upgrade pip first: `pip install --upgrade pip setuptools wheel` |
