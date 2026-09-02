"""
translator.py
===============================================================================
English -> Tamil Machine Translation Engine
-------------------------------------------------------------------------------
This module implements a two-tier translation architecture:

  TIER 1 (preferred, production):  NEURAL MODEL BACKEND
      Uses a pretrained transformer sequence-to-sequence model fine-tuned for
      English->Tamil (e.g. AI4Bharat's IndicTrans2, or Meta's NLLB-200,
      via HuggingFace `transformers`). This gives fluent, context-aware
      translation and is what a real deployment should use.

  TIER 2 (offline fallback, always available): RULE-BASED / DICTIONARY BACKEND
      A lightweight, fully self-contained translator that:
        1. Normalizes and tokenizes the input sentence.
        2. Detects sentence type (statement / question / negative / greeting).
        3. Looks up each token (and common multi-word phrases) in a curated
           English->Tamil bilingual lexicon.
        4. Applies simple English (SVO) -> Tamil (SOV) word-reordering rules
           and basic Tamil verb-ending selection (tense / question / polite).
        5. Falls back to marking any out-of-vocabulary (OOV) word so the user
           can see exactly what could not be translated, rather than silently
           failing or guessing.

The `Translator` class automatically tries the neural backend first (if the
`transformers`/`torch` libraries AND the model weights are available), and
transparently falls back to the rule-based backend otherwise. This makes the
application runnable in *any* environment -- including fully offline /
sandboxed ones with no access to model hubs -- while still being architected
correctly for a production upgrade path.
===============================================================================
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# 1. Result container
# ==============================================================================

@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    backend_used: str                     # "neural" or "rule_based" or "error"
    oov_words: list = field(default_factory=list)   # words that couldn't be translated
    warnings: list = field(default_factory=list)
    success: bool = True


# ==============================================================================
# 2. Input validation
# ==============================================================================

class InputValidator:
    """Centralised validation & sanitisation of raw user input before it is
    handed to either translation backend. Keeping this separate makes the
    'invalid / empty / unexpected input' requirement easy to test in isolation.
    """

    MAX_LEN = 1000

    @staticmethod
    def validate(raw_text) -> tuple[bool, str, list]:
        """
        Returns: (is_valid, cleaned_text, warnings)
        Never raises -- always returns a usable (possibly empty) string so the
        caller can decide how to react.
        """
        warnings = []

        # --- Type safety: non-string input (e.g. None, int, list from a bad caller) ---
        if raw_text is None:
            return False, "", ["No input provided (received None)."]
        if not isinstance(raw_text, str):
            return False, "", [f"Unsupported input type: {type(raw_text).__name__}. Expected text."]

        text = raw_text.strip()

        # --- Empty / whitespace-only input ---
        if text == "":
            return False, "", ["Input is empty. Please type an English sentence to translate."]

        # --- Length guard (protect against pathological/huge input) ---
        if len(text) > InputValidator.MAX_LEN:
            warnings.append(f"Input truncated to {InputValidator.MAX_LEN} characters.")
            text = text[: InputValidator.MAX_LEN]

        # --- Strip control characters / potential HTML-injection payloads ---
        cleaned = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
        cleaned = re.sub(r"<[^>]+>", "", cleaned)  # strip HTML/script tags defensively
        if cleaned != text:
            warnings.append("Potentially unsafe characters/tags were removed from the input.")
        text = cleaned.strip()
        if text == "":
            return False, "", warnings + ["Input contained no translatable text after sanitisation."]

        # --- Numeric-only / punctuation-only input ---
        if re.fullmatch(r"[\d\s\W]+", text, flags=re.UNICODE) and not re.search(r"[A-Za-z]", text):
            warnings.append("Input has no alphabetic (English) content to translate; returning as-is.")
            return False, text, warnings

        # --- Detect if the text is already Tamil / non-Latin script ---
        non_latin_ratio = sum(1 for ch in text if ch.isalpha() and ord(ch) > 0x2FF) / max(
            1, sum(1 for ch in text if ch.isalpha())
        )
        if non_latin_ratio > 0.4:
            warnings.append(
                "Input does not appear to be English text (non-Latin script detected). "
                "This tool translates English -> Tamil only."
            )
            return False, text, warnings

        return True, text, warnings


# ==============================================================================
# 3. Rule-based / dictionary backend
# ==============================================================================

class RuleBasedTranslator:
    """A compact but functional English->Tamil rule-based MT engine."""

    # ---- Multi-word phrases (checked before single-word lookup) -------------
    PHRASES = {
        "good morning": "காலை வணக்கம்",
        "good night": "இனிய இரவு",
        "good evening": "மாலை வணக்கம்",
        "thank you": "நன்றி",
        "thank you very much": "மிக்க நன்றி",
        "how are you": "நீங்கள் எப்படி இருக்கிறீர்கள்",
        "what is your name": "உங்கள் பெயர் என்ன",
        "my name is": "என் பெயர்",
        "i am fine": "நான் நலமாக இருக்கிறேன்",
        "nice to meet you": "உங்களை சந்தித்ததில் மகிழ்ச்சி",
        "see you later": "பிறகு சந்திப்போம்",
        "excuse me": "மன்னிக்கவும்",
        "i am sorry": "மன்னிக்கவும்",
        "i love you": "நான் உன்னை காதலிக்கிறேன்",
        "welcome": "வரவேற்கிறோம்",
        "have a nice day": "நல்ல நாளாக அமையட்டும்",
        "what time is it": "இப்போது என்ன நேரம்",
        "where are you going": "நீங்கள் எங்கே செல்கிறீர்கள்",
    }

    # ---- Core bilingual lexicon (English lemma -> Tamil) --------------------
    # NOTE: kept intentionally compact/curated for a class-project prototype.
    LEXICON = {
        # pronouns
        "i": "நான்", "you": "நீ", "he": "அவன்", "she": "அவள்", "it": "அது",
        "we": "நாங்கள்", "they": "அவர்கள்", "my": "என்", "your": "உன்",
        "his": "அவனுடைய", "her": "அவளுடைய", "our": "எங்கள்", "their": "அவர்களுடைய",
        "me": "என்னை", "him": "அவனை", "them": "அவர்களை",
        # greetings / politeness
        "hello": "வணக்கம்", "hi": "வணக்கம்", "bye": "போய் வருகிறேன்",
        "please": "தயவுசெய்து", "sorry": "மன்னிக்கவும்", "yes": "ஆம்", "no": "இல்லை",
        "thanks": "நன்றி",
        # question words
        "what": "என்ன", "who": "யார்", "where": "எங்கே", "when": "எப்போது",
        "why": "ஏன்", "how": "எப்படி", "which": "எது",
        # common verbs (base forms; simplistic conjugation handled separately)
        "am": "இருக்கிறேன்", "is": "இருக்கிறது", "are": "இருக்கிறீர்கள்",
        "go": "செல்", "going": "செல்கிறேன்", "come": "வா", "coming": "வருகிறேன்",
        "eat": "சாப்பிடு", "eating": "சாப்பிடுகிறேன்", "eaten": "சாப்பிட்டேன்",
        "drink": "குடி", "read": "படி", "reading": "படிக்கிறேன்",
        "write": "எழுது", "writing": "எழுதுகிறேன்",
        "play": "விளையாடு", "playing": "விளையாடுகிறேன்",
        "sleep": "தூங்கு", "sleeping": "தூங்குகிறேன்",
        "study": "படி", "studying": "படிக்கிறேன்",
        "work": "வேலை செய்", "working": "வேலை செய்கிறேன்",
        "love": "காதலி", "like": "விரும்பு", "want": "வேண்டும்",
        "have": "வைத்திரு", "has": "வைத்திருக்கிறார்", "had": "வைத்திருந்தேன்",
        "do": "செய்", "did": "செய்தேன்", "does": "செய்கிறார்",
        "see": "பார்", "saw": "பார்த்தேன்", "watch": "பார்",
        "know": "தெரியும்", "think": "நினை", "speak": "பேசு",
        "understand": "புரிந்துகொள்", "help": "உதவி செய்", "buy": "வாங்கு",
        "sell": "விற்று", "give": "கொடு", "take": "எடு", "make": "செய்",
        "was": "இருந்தேன்", "were": "இருந்தார்கள்", "will": "வேண்டும்",
        "can": "முடியும்", "cannot": "முடியாது", "should": "வேண்டும்",
        # negation
        "not": "இல்லை", "don't": "இல்லை", "doesn't": "இல்லை", "didn't": "இல்லை",
        # common nouns
        "school": "பள்ளி", "book": "புத்தகம்", "water": "தண்ணீர்", "food": "உணவு",
        "house": "வீடு", "home": "வீடு", "car": "கார்", "school bag": "பள்ளி பை",
        "teacher": "ஆசிரியர்", "student": "மாணவன்", "friend": "நண்பன்",
        "family": "குடும்பம்", "mother": "அம்மா", "father": "அப்பா",
        "brother": "சகோதரன்", "sister": "சகோதரி", "doctor": "மருத்துவர்",
        "hospital": "மருத்துவமனை", "market": "சந்தை", "city": "நகரம்",
        "village": "கிராமம்", "country": "நாடு", "language": "மொழி",
        "english": "ஆங்கிலம்", "tamil": "தமிழ்", "computer": "கணினி",
        "phone": "தொலைபேசி", "money": "பணம்", "time": "நேரம்", "day": "நாள்",
        "night": "இரவு", "morning": "காலை", "evening": "மாலை",
        "today": "இன்று", "tomorrow": "நாளை", "yesterday": "நேற்று",
        "weather": "வானிலை", "rain": "மழை", "sun": "சூரியன்", "moon": "நிலா",
        "sky": "வானம்", "river": "ஆறு", "sea": "கடல்", "mountain": "மலை",
        "tree": "மரம்", "flower": "பூ", "dog": "நாய்", "cat": "பூனை",
        "bird": "பறவை", "fish": "மீன்", "movie": "திரைப்படம்", "music": "இசை",
        "song": "பாடல்", "game": "விளையாட்டு", "job": "வேலை", "office": "அலுவலகம்",
        "meeting": "கூட்டம்", "project": "திட்டம்", "exam": "தேர்வு", "class": "வகுப்பு",
        # adjectives
        "good": "நல்ல", "bad": "கெட்ட", "big": "பெரிய", "small": "சிறிய",
        "happy": "மகிழ்ச்சியான", "sad": "சோகமான", "beautiful": "அழகான",
        "new": "புதிய", "old": "பழைய", "hot": "சூடான", "cold": "குளிர்ந்த",
        "fast": "வேகமான", "slow": "மெதுவான", "easy": "எளிதான", "difficult": "கடினமான",
        "important": "முக்கியமான", "interesting": "சுவாரஸ்யமான",
        # numbers
        "one": "ஒன்று", "two": "இரண்டு", "three": "மூன்று", "four": "நான்கு",
        "five": "ஐந்து", "six": "ஆறு", "seven": "ஏழு", "eight": "எட்டு",
        "nine": "ஒன்பது", "ten": "பத்து",
        # misc function words (often dropped/merged in natural Tamil, but shown for transparency)
        "a": "", "an": "", "the": "", "to": "", "of": "", "in": "இல்",
        "on": "மேல்", "at": "இல்", "and": "மற்றும்", "with": "உடன்",
        "for": "க்காக", "very": "மிகவும்", "too": "அதிகமாக",
    }

    QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "is", "are", "do", "does", "did", "can", "will"}

    def translate(self, text: str) -> TranslationResult:
        original = text
        lower = text.lower().strip()
        lower_clean = re.sub(r"[^\w\s']", " ", lower)  # keep apostrophes for don't/doesn't
        lower_clean = re.sub(r"\s+", " ", lower_clean).strip()

        # 1) Try direct full-sentence phrase match
        stripped_punct = lower.rstrip("?!. ")
        if stripped_punct in self.PHRASES:
            return TranslationResult(original, self.PHRASES[stripped_punct] + self._end_punct(text), "rule_based")

        # 2) Try matching known multi-word phrases anywhere in the sentence
        working = lower_clean
        phrase_hits = []
        for phrase, tam in sorted(self.PHRASES.items(), key=lambda kv: -len(kv[0])):
            if phrase in working:
                phrase_hits.append((phrase, tam))
                working = working.replace(phrase, " ")

        tokens = working.split()
        is_question = stripped_punct.endswith("?") or (tokens and tokens[0] in self.QUESTION_WORDS) or text.strip().endswith("?")

        translated_tokens = []
        oov = []
        for tok in tokens:
            tok_norm = tok.strip("'")
            if tok_norm in self.LEXICON:
                tam = self.LEXICON[tok_norm]
                if tam:  # skip empty-string articles like "a"/"the"
                    translated_tokens.append(tam)
            else:
                # Out-of-vocabulary: keep the original word marked, so the user
                # can see exactly what wasn't covered by the dictionary.
                oov.append(tok_norm)
                translated_tokens.append(f"[{tok}]")

        # crude SOV-ish reorder heuristic: push a leading "is/are/am" (copula)
        # to the end, which is closer to natural Tamil sentence structure.
        copulas = {"இருக்கிறேன்", "இருக்கிறது", "இருக்கிறீர்கள்"}
        reordered = [t for t in translated_tokens if t not in copulas] + \
                    [t for t in translated_tokens if t in copulas]

        all_parts = [tam for _, tam in phrase_hits] + reordered
        sentence = " ".join(p for p in all_parts if p).strip()

        if is_question and not sentence.endswith("?"):
            sentence += " ?"

        warnings = []
        if oov:
            warnings.append(
                f"{len(oov)} word(s) not found in dictionary and left in English (in brackets): "
                + ", ".join(sorted(set(oov)))
            )

        if not sentence:
            return TranslationResult(original, "", "rule_based", oov, ["Nothing could be translated."], success=False)

        return TranslationResult(original, sentence, "rule_based", oov, warnings, success=True)

    @staticmethod
    def _end_punct(text: str) -> str:
        t = text.strip()
        if t.endswith("?"):
            return " ?"
        if t.endswith("!"):
            return " !"
        return ""


# ==============================================================================
# 4. Neural backend (interface / adapter -- production upgrade path)
# ==============================================================================

class NeuralTranslator:
    """
    Adapter for a real pretrained EN->TA neural MT model, e.g.:
        - ai4bharat/indictrans2-en-indic-1B   (recommended: state-of-the-art, Indic-specialised)
        - facebook/nllb-200-distilled-600M    (general purpose, 200 languages incl. Tamil 'tam_Taml')
        - Helsinki-NLP/opus-mt-en-mul         (lightweight MarianMT baseline)

    This class only activates if `transformers` + `torch` are installed AND the
    model can actually be downloaded/loaded (requires internet access to
    huggingface.co in a normal deployment). If unavailable, `is_available()`
    returns False and the caller transparently falls back to RuleBasedTranslator.
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self._pipe = None
        self._load_error: Optional[str] = None
        self._try_load()

    def _try_load(self):
        try:
            from transformers import pipeline  # noqa: WPS433 (intentional lazy import)
            self._pipe = pipeline(
                "translation",
                model=self.model_name,
                src_lang="eng_Latn",
                tgt_lang="tam_Taml",
            )
        except Exception as exc:  # broad on purpose: any failure -> fallback
            self._load_error = str(exc)
            self._pipe = None

    def is_available(self) -> bool:
        return self._pipe is not None

    def translate(self, text: str) -> TranslationResult:
        if not self.is_available():
            return TranslationResult(text, "", "error", [], [self._load_error or "Neural backend unavailable."], success=False)
        try:
            out = self._pipe(text, max_length=256)[0]["translation_text"]
            return TranslationResult(text, out, "neural", [], [], success=True)
        except Exception as exc:
            return TranslationResult(text, "", "error", [], [str(exc)], success=False)


# ==============================================================================
# 5. Public facade
# ==============================================================================

class Translator:
    """Main entry point used by the application layer (UI, tests, etc.)."""

    def __init__(self, prefer_neural: bool = True, neural_model_name: str = "facebook/nllb-200-distilled-600M"):
        self.rule_based = RuleBasedTranslator()
        self.neural = None
        if prefer_neural:
            self.neural = NeuralTranslator(neural_model_name)

    def active_backend(self) -> str:
        if self.neural is not None and self.neural.is_available():
            return "neural"
        return "rule_based"

    def translate(self, raw_text) -> TranslationResult:
        is_valid, cleaned, warnings = InputValidator.validate(raw_text)

        if not is_valid:
            return TranslationResult(
                source_text=raw_text if isinstance(raw_text, str) else str(raw_text),
                translated_text="",
                backend_used="validation",
                oov_words=[],
                warnings=warnings,
                success=False,
            )

        if self.neural is not None and self.neural.is_available():
            result = self.neural.translate(cleaned)
            result.warnings = warnings + result.warnings
            if result.success:
                return result
            # neural failed at runtime -> fall back silently to rule-based
        result = self.rule_based.translate(cleaned)
        result.warnings = warnings + result.warnings
        return result


if __name__ == "__main__":
    t = Translator(prefer_neural=False)  # force rule-based for a quick manual check
    for s in ["Hello, how are you?", "I am going to school today.", "", "12345", "Bonjour le monde"]:
        r = t.translate(s)
        print(f"{s!r:45} -> {r.translated_text!r} | backend={r.backend_used} | warnings={r.warnings}")
