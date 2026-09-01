"""
Unit 4 - Experiment 6: Text-to-Speech for Engineering Text
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop a Text-to-Speech application that converts engineering-related text
    into natural-sounding speech using a pre-trained AI model.

MODEL:
    microsoft/speecht5_tts  - pre-trained text-to-speech transformer
    microsoft/speecht5_hifigan - vocoder that turns the spectrogram into audio
    A speaker embedding from the CMU-Arctic dataset gives the voice its identity.

RUN:
    python exp06_text_to_speech.py
    python exp06_text_to_speech.py "Ohm's law states that current is proportional to voltage."
"""

import sys
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

TTS_MODEL = "microsoft/speecht5_tts"
VOCODER = "microsoft/speecht5_hifigan"
SAMPLE_RATE = 16000
OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_TEXT = (
    "An induction motor works on the principle of electromagnetic induction. "
    "The rotating magnetic field produced by the stator induces a current in the "
    "rotor, and the interaction of the two fields produces torque."
)


def load_models():
    print("Loading the pre-trained text-to-speech models...")
    processor = SpeechT5Processor.from_pretrained(TTS_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER)

    # A 512-dim x-vector that defines the speaker's voice.
    embeddings = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker = torch.tensor(embeddings[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker


def split_sentences(text, limit=200):
    """SpeechT5 works best on short inputs, so speak one sentence at a time."""
    parts, current = [], ""
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}.".strip()
        if len(candidate) > limit and current:
            parts.append(current)
            current = f"{sentence}."
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def main():
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    processor, model, vocoder, speaker = load_models()

    print("\n=== Text-to-Speech ===")
    print(f"Text: {text}\n")

    chunks = []
    for i, part in enumerate(split_sentences(text), start=1):
        print(f"Synthesising part {i}: {part[:60]}...")
        inputs = processor(text=part, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker, vocoder=vocoder)
        chunks.append(speech)

    audio = torch.cat(chunks).numpy()

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "exp06_speech.wav"
    sf.write(path, audio, SAMPLE_RATE)

    print(f"\nAudio saved to: {path}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.1f} seconds")

    # Play it back if a playback library is available.
    try:
        import sounddevice as sd
        print("Playing...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
    except Exception:
        print("(install sounddevice to hear it automatically, or open the wav file)")


if __name__ == "__main__":
    main()
