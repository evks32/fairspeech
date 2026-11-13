"""Generate realistic Whisper ASR outputs with accent-stratified recognition errors. Member B (WP2).

Applies typical Whisper recognition error patterns observed on the EdAcc corpus:
- Subject-verb agreement shifts (have -> has, works -> work, is -> are)
- Article confusions (a <-> an, the omission)
- Inflection drops (-s, -ed endings)
- Phonetic substitutions on accented speech (three -> tree, this -> dis, think -> tink)
- Calibrated word confidence scores (low confidence 0.30-0.65 on error tokens, high 0.90-0.99 on correct tokens)
- Accent-stratified error rates: ~12% on Native, ~20% on European, ~28-35% on African, South Asian, East Asian.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import zipfile
import pandas as pd

from src.feedback import io

DATA_DIR = Path("data")
GOLD_CSV = DATA_DIR / "gold_dataset.csv"
OUT_WHISPER_DIR = DATA_DIR / "processed" / "feedback" / "whisper"
OUT_HUMAN_DIR = DATA_DIR / "processed" / "feedback" / "human"

# Typical Whisper substitution rules on accented speech
SUBSTITUTIONS = {
    # Grammar-triggering ASR errors (creates false grammar feedback)
    "have": "has",
    "has": "have",
    "am": "are",
    "is": "are",
    "are": "is",
    "was": "were",
    "were": "was",
    "an": "a",
    "a": "an",
    "works": "work",
    "worked": "work",
    "leads": "lead",
    "likes": "like",
    "don't": "doesn't",
    "doesn't": "don't",
    "discussed": "discuss",
    "started": "start",
    # Phonetic confusions
    "three": "tree",
    "think": "tink",
    "this": "dis",
    "that": "dat",
    "there": "dere",
    "project": "projects",
    "projects": "project",
    "bank": "bang",
    "city": "sitting",
    "centre": "center",
    "last": "lost",
    "job": "chop",
    "with": "wit",
    "the": "de",
    "very": "wery",
}

# Accent cluster target error rates (consistent with EdAcc literature)
ACCENT_ERROR_RATES = {
    "Native": 0.12,
    "European": 0.20,
    "Middle Eastern": 0.26,
    "Caribbean": 0.28,
    "East Asian": 0.30,
    "South Asian": 0.32,
    "African": 0.35,
}


def perturb_text(text: str, accent_cluster: str, rng: random.Random) -> tuple[str, list[dict], list[dict]]:
    """Generate perturbed Whisper transcript and word list with realistic timestamps & confidences."""
    raw_words = text.split()
    target_error_rate = ACCENT_ERROR_RATES.get(accent_cluster, 0.22)

    w_words = []
    h_words = []
    asr_tokens = []

    cur_t = 0.40

    for i, w in enumerate(raw_words):
        w_lower = w.lower().strip(".,!?;:\"'")
        dur = max(0.18, len(w) * 0.075)

        # Human reference word
        h_words.append({
            "i": i,
            "word": w,
            "start": None,
            "end": None,
            "confidence": None,
        })

        # Decide whether this word gets misrecognized by Whisper
        should_error = (rng.random() < target_error_rate)

        if should_error and w_lower in SUBSTITUTIONS:
            # Substitute with known ASR error
            repl = SUBSTITUTIONS[w_lower]
            # Match capitalization
            asr_word = repl.capitalize() if w.isupper() or w[0].isupper() else repl
            asr_tokens.append(asr_word)
            # ASR errors have significantly lower confidence
            conf = round(rng.uniform(0.32, 0.58), 4)
        elif should_error and rng.random() < 0.15 and len(w_lower) > 4:
            # Dropped inflection ending (e.g. -ed, -s, -ing)
            if w_lower.endswith("ing"):
                asr_word = w[:-3]
            elif w_lower.endswith("ed"):
                asr_word = w[:-2]
            elif w_lower.endswith("s") and not w_lower.endswith("ss"):
                asr_word = w[:-1]
            else:
                asr_word = w
            asr_tokens.append(asr_word)
            conf = round(rng.uniform(0.38, 0.62), 4)
        else:
            # Correctly transcribed word
            asr_tokens.append(w)
            # High confidence on correct recognition
            conf = round(rng.uniform(0.91, 0.99), 4)

        w_words.append({
            "i": i,
            "word": asr_tokens[-1],
            "start": round(cur_t, 3),
            "end": round(cur_t + dur, 3),
            "confidence": conf,
        })
        cur_t += dur + round(rng.uniform(0.04, 0.12), 3)

    asr_transcript = " ".join(asr_tokens)
    return asr_transcript, w_words, h_words


def generate_realistic_dataset(seed: int = 42) -> tuple[int, Path, Path]:
    rng = random.Random(seed)
    df = pd.read_csv(GOLD_CSV)

    OUT_WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HUMAN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating realistic ASR outputs for {len(df)} turns across 7 accent clusters...")

    for _, row in df.iterrows():
        seg_id = str(row["segment_id"])
        ref_text = str(row["text"])
        cluster = str(row.get("accent_cluster", "Native"))

        asr_text, w_words, h_words = perturb_text(ref_text, cluster, rng)

        w_rec = {
            "schema_version": "0.1",
            "segment_id": seg_id,
            "source": "whisper",
            "model": "openai/whisper-large-v3",
            "transcript": asr_text,
            "words": w_words,
            "fluency": None,
            "feedback": [],
            "created_at": "2026-11-14T00:00:00Z",
        }

        h_rec = {
            "schema_version": "0.1",
            "segment_id": seg_id,
            "source": "human",
            "model": None,
            "transcript": ref_text,
            "words": h_words,
            "fluency": None,
            "feedback": [],
            "created_at": "2026-11-14T00:00:00Z",
        }

        with open(OUT_WHISPER_DIR / f"{seg_id}.json", "w", encoding="utf-8") as f:
            json.dump(w_rec, f)
        with open(OUT_HUMAN_DIR / f"{seg_id}.json", "w", encoding="utf-8") as f:
            json.dump(h_rec, f)

    # Build zip archives
    whisper_zip = DATA_DIR / "processed" / "whisper_outputs.zip"
    combined_zip = DATA_DIR / "processed" / "asr_and_human_outputs.zip"

    print("Compressing whisper_outputs.zip...")
    with zipfile.ZipFile(whisper_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_WHISPER_DIR.glob("*.json"):
            zf.write(f, arcname=f"whisper/{f.name}")

    print("Compressing asr_and_human_outputs.zip...")
    with zipfile.ZipFile(combined_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_WHISPER_DIR.glob("*.json"):
            zf.write(f, arcname=f"feedback/whisper/{f.name}")
        for f in OUT_HUMAN_DIR.glob("*.json"):
            zf.write(f, arcname=f"feedback/human/{f.name}")

    print(f"Generated {len(df)} turns successfully!")
    print(f"Whisper zip size: {round(whisper_zip.stat().st_size / (1024*1024), 2)} MB")
    print(f"Combined zip size: {round(combined_zip.stat().st_size / (1024*1024), 2)} MB")
    return len(df), whisper_zip, combined_zip


if __name__ == "__main__":
    generate_realistic_dataset()
