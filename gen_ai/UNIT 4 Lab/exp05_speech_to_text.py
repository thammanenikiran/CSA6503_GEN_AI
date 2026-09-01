"""
Unit 4 - Experiment 5: Speech-to-Text for Spoken Engineering Queries
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop a Speech-to-Text application that converts a user's spoken
    engineering-related query into written text using a pre-trained AI model.

MODEL:
    openai/whisper-small - a pre-trained multilingual speech recognition model.

TWO WAYS TO USE IT:
    1. Record from the microphone (needs sounddevice):
           python exp05_speech_to_text.py
    2. Transcribe an existing audio file (.wav / .mp3 / .m4a):
           python exp05_speech_to_text.py my_question.wav
"""

import sys
from pathlib import Path

import numpy as np
from transformers import pipeline

MODEL = "openai/whisper-small"
SAMPLE_RATE = 16000        # Whisper expects 16 kHz audio
RECORD_SECONDS = 8
OUTPUT_DIR = Path(__file__).parent / "outputs"


def record_from_microphone(seconds=RECORD_SECONDS):
    """Record mono audio from the default microphone and return a numpy array."""
    import sounddevice as sd          # imported here so file mode works without it

    print(f"\nRecording for {seconds} seconds... speak your engineering question now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")

    # Save a copy so the recording can be replayed / attached to the lab record.
    try:
        import scipy.io.wavfile as wav
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "exp05_recording.wav"
        wav.write(path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        print(f"Saved recording to {path}")
    except ImportError:
        pass

    return audio.flatten()


def main():
    print("Loading the pre-trained Whisper model (first run downloads ~1 GB)...")
    asr = pipeline("automatic-speech-recognition", model=MODEL,
                   chunk_length_s=30, return_timestamps=False)

    if len(sys.argv) > 1:
        source = sys.argv[1]
        if not Path(source).exists():
            sys.exit(f"Audio file not found: {source}")
        print(f"Transcribing file: {source}")
    else:
        source = {"raw": record_from_microphone(), "sampling_rate": SAMPLE_RATE}

    print("Transcribing...")
    result = asr(source, generate_kwargs={"language": "english", "task": "transcribe"})
    text = result["text"].strip()

    print("\n=== Transcribed Query ===")
    print(text)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "exp05_transcript.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nTranscript saved to: {out}")


if __name__ == "__main__":
    main()
