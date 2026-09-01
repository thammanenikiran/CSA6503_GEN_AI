"""
Unit 4 - Experiment 8: English to Indian Language Machine Translation
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop a machine translation application that translates an engineering
    document from English into another Indian language using a pre-trained
    translation model.

MODEL:
    facebook/nllb-200-distilled-600M - Meta's "No Language Left Behind" model,
    which supports 200 languages including Tamil, Hindi, Telugu, Kannada,
    Malayalam, Bengali, Marathi and Gujarati.

RUN:
    python exp08_machine_translation.py                       # Tamil, default doc
    python exp08_machine_translation.py hindi
    python exp08_machine_translation.py tamil docs/smart_grid.txt
"""

import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-distilled-600M"
BASE = Path(__file__).parent
DEFAULT_DOC = BASE / "docs" / "smart_grid.txt"

# NLLB uses FLORES-200 language codes.
LANGUAGES = {
    "tamil": "tam_Taml",
    "hindi": "hin_Deva",
    "telugu": "tel_Telu",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "bengali": "ben_Beng",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
}


def split_sentences(text):
    """Translate sentence by sentence - NMT models work best on single sentences."""
    sentences = []
    for line in text.split("\n"):
        for part in line.split(". "):
            part = part.strip()
            if part:
                sentences.append(part if part.endswith(".") else part + ".")
    return sentences


def main():
    args = sys.argv[1:]
    language = args[0].lower() if args else "tamil"
    doc_path = Path(args[1]) if len(args) > 1 else DEFAULT_DOC

    if language not in LANGUAGES:
        sys.exit(f"Supported languages: {', '.join(LANGUAGES)}")
    if not doc_path.exists():
        sys.exit(f"Document not found: {doc_path}")

    target_code = LANGUAGES[language]
    text = doc_path.read_text(encoding="utf-8")

    print(f"Loading {MODEL} (first run downloads ~2.5 GB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    target_id = tokenizer.convert_tokens_to_ids(target_code)

    sentences = split_sentences(text)
    print(f"\nTranslating {len(sentences)} sentences: English -> {language.title()}\n")

    translated = []
    for i, sentence in enumerate(sentences, start=1):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        output = model.generate(**inputs, forced_bos_token_id=target_id, max_length=256)
        result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        translated.append(result)
        print(f"[{i}/{len(sentences)}] EN : {sentence}")
        print(f"        {language[:2].upper()} : {result}\n")

    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"exp08_{doc_path.stem}_{language}.txt"
    out.write_text("\n".join(translated), encoding="utf-8")
    print(f"Translation saved to: {out}")


if __name__ == "__main__":
    main()
