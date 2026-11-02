"""ASR inference output evaluation and Word Error Rate (WER) per accent group. Member B (WP2).

Includes:
- Standardised WER computation via JiWER
- Grouped WER breakdown by accent cluster and model
- Detailed error breakdown (insertions, deletions, substitutions)
- Confident error and hallucination detection per accent group
"""
from __future__ import annotations

import re
from typing import Any
import jiwer
import numpy as np
import pandas as pd

from src.asr.normalisation import normalise_text, tokenize_words


def compute_wer(reference: str, hypothesis: str, normalise: bool = True) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis transcripts."""
    ref = normalise_text(reference) if normalise else reference.strip()
    hyp = normalise_text(hypothesis) if normalise else hypothesis.strip()

    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0

    return float(jiwer.wer(ref, hyp))


def compute_detailed_metrics(reference: str, hypothesis: str, normalise: bool = True) -> dict[str, Any]:
    """Compute detailed alignment metrics including substitutions, deletions, and insertions."""
    ref = normalise_text(reference) if normalise else reference.strip()
    hyp = normalise_text(hypothesis) if normalise else hypothesis.strip()

    if not ref:
        wer_val = 0.0 if not hyp else 1.0
        return {
            "wer": wer_val,
            "mer": wer_val,
            "wil": wer_val,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp.split()) if hyp else 0,
            "hits": 0,
        }

    out = jiwer.process_words(ref, hyp)
    return {
        "wer": float(out.wer),
        "mer": float(out.mer),
        "wil": float(out.wil),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
    }


def compute_group_wer(
    df: pd.DataFrame,
    ref_col: str = "human_transcript",
    hyp_col: str = "asr_transcript",
    group_col: str = "accent_cluster",
    normalise: bool = True,
) -> pd.DataFrame:
    """Compute mean Word Error Rate and turn count grouped by accent cluster or other attribute.

    Expects df with columns [ref_col, hyp_col, group_col].
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    rows = []
    for grp, sub in df.groupby(group_col):
        wers = []
        for _, row in sub.iterrows():
            ref = str(row.get(ref_col, ""))
            hyp = str(row.get(hyp_col, ""))
            if pd.notna(row.get(ref_col)) and pd.notna(row.get(hyp_col)):
                wers.append(compute_wer(ref, hyp, normalise=normalise))

        if wers:
            mean_wer = float(np.mean(wers))
            std_wer = float(np.std(wers))
            median_wer = float(np.median(wers))
        else:
            mean_wer, std_wer, median_wer = 0.0, 0.0, 0.0

        rows.append({
            group_col: grp,
            "turns": len(wers),
            "mean_wer": round(mean_wer, 4),
            "wer_percentage": round(mean_wer * 100.0, 2),
            "std_wer": round(std_wer, 4),
            "median_wer": round(median_wer, 4),
        })

    return pd.DataFrame(rows).sort_values("mean_wer", ascending=False).reset_index(drop=True)


def detect_confident_errors(
    words: list[dict[str, Any]],
    reference: str,
    confidence_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Identify words transcribed with high confidence that do not appear in reference (confident errors)."""
    ref_words = set(tokenize_words(reference))
    confident_errors = []

    for w_obj in words:
        w_text = normalise_text(w_obj.get("word", ""))
        conf = w_obj.get("confidence")
        if conf is not None and conf >= confidence_threshold:
            if w_text and w_text not in ref_words:
                confident_errors.append({
                    "word": w_text,
                    "confidence": conf,
                    "start": w_obj.get("start"),
                    "end": w_obj.get("end"),
                })

    return confident_errors


def detect_hallucinations(hypothesis: str, min_repeat: int = 3) -> list[str]:
    """Detect repetitive n-gram hallucination loops common in autoregressive ASR models."""
    norm = normalise_text(hypothesis)
    words = norm.split()
    hallucinations = []

    # Check for repeated unigrams, bigrams, trigrams
    for n in [1, 2, 3, 4]:
        if len(words) < n * min_repeat:
            continue
        for i in range(len(words) - n * min_repeat + 1):
            pattern = words[i:i + n]
            pattern_str = " ".join(pattern)
            matched = True
            for r in range(1, min_repeat):
                chunk = words[i + r * n: i + (r + 1) * n]
                if chunk != pattern:
                    matched = False
                    break
            if matched and pattern_str not in hallucinations:
                hallucinations.append(pattern_str)

    return hallucinations
