"""Text normalisation and cleaning for ASR transcripts and WER evaluation. Member B (WP2).

Follows standard EdAcc / speech assessment normalisation:
- Lowercasing
- Stripping extraneous punctuation while preserving contractions (e.g. don't, I've)
- Normalising whitespace
- Preserving spoken disfluencies and fillers (um, uh, erm, like, you know)
- Removing non-speech noise tags (e.g. [laughter], <whisper>, [applause])
"""
from __future__ import annotations

import re


NOISE_TAG_PATTERN = re.compile(r"\[.*?\]|<.*?>|\(.*?\)")
PUNCT_PATTERN = re.compile(r"[^\w\s']")


def normalise_text(text: str) -> str:
    """Standardise text for ASR transcription comparison and WER evaluation."""
    if not text:
        return ""

    # Remove noise tags like [laughter], <cough>, etc.
    cleaned = NOISE_TAG_PATTERN.sub(" ", text)

    # Convert to lowercase
    cleaned = cleaned.lower()

    # Normalize various quote/apostrophe characters to standard single quote
    cleaned = re.sub(r"[\u2018\u2019\u201a\u201b\u2032`]", "'", cleaned)

    # Remove punctuation except single quotes within words
    cleaned = PUNCT_PATTERN.sub(" ", cleaned)

    # Clean isolated apostrophes (e.g. ' word ' -> word)
    cleaned = re.sub(r"\s+'|'\s+", " ", cleaned)
    cleaned = re.sub(r"^'|'$", "", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def tokenize_words(text: str) -> list[str]:
    """Tokenize normalised text into word tokens."""
    norm = normalise_text(text)
    if not norm:
        return []
    return norm.split()
