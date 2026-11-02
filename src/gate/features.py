"""Gate features extraction per feedback flag. Member B (WP2).

Unit of prediction: One grammar or filler flag produced from an ASR transcript.
Label: 'false' (ASR-induced) or 'genuine'.

Features extracted strictly according to Proposal Appendix B (Table 4):
1. confidence: ASR confidence of the flagged word (or min across multi-word flag)
2. conf_window_min: minimum confidence in +-2-word context window
3. conf_window_mean: mean confidence in +-2-word context window
4. agreement_3way: exact agreement (1/0) of Whisper, CrisperWhisper, and Vosk on the flagged word
5. agreement_ratio: proportion of available ASR models that agree on the flagged word
6. category: grammar / filler rule category (one-hot encoded downstream)
7. word_freq_zipf: word frequency on the Zipf scale (wordfreq)
8. word_len: character length of the flagged word
9. pause_before: pause duration (seconds) immediately preceding the flagged word
10. pause_after: pause duration (seconds) immediately following the flagged word
11. local_speech_rate: speaking rate (words per second) in the context window
12. segment_mean_confidence: mean confidence across all words in the segment

STRICTLY EXCLUDED FEATURES:
- Accent cluster / accent label
- Speaker ID
(Neither may be inputs to the classifier; they are used only for stratified splitting and evaluation).
"""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
import wordfreq

NUMERIC_FEATURE_COLS = [
    "confidence",
    "conf_window_min",
    "conf_window_mean",
    "agreement_3way",
    "agreement_ratio",
    "word_freq_zipf",
    "word_len",
    "pause_before",
    "pause_after",
    "local_speech_rate",
    "segment_mean_confidence",
]

CATEGORICAL_FEATURE_COLS = [
    "category",
]


def _clean_word(w: str) -> str:
    """Normalize word for dictionary lookup and cross-model comparison."""
    return re.sub(r"[^\w]", "", w).lower()


def build_features(
    asr_records: dict[str, list[dict[str, Any]]] | list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Extract gate features for each feedback flag.

    Accepts:
    - A flat list of feedback JSON records conforming to data contracts
    - A dict mapping segment_id -> list of records across models
    - A dict mapping source -> list of records
    - A DataFrame with existing flag annotations
    """
    if isinstance(asr_records, pd.DataFrame):
        df = asr_records.copy()
        # Ensure all expected feature columns exist
        for col in NUMERIC_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        for col in CATEGORICAL_FEATURE_COLS:
            if col not in df.columns:
                df[col] = "other"
        return df

    # Normalize records into segment_id -> {source: record}
    seg_source_map: dict[str, dict[str, dict[str, Any]]] = {}

    if isinstance(asr_records, list):
        for rec in asr_records:
            s_id = rec.get("segment_id", "seg")
            src = rec.get("source", "unknown")
            seg_source_map.setdefault(s_id, {})[src] = rec
    elif isinstance(asr_records, dict):
        for key, val in asr_records.items():
            if isinstance(val, list):
                for rec in val:
                    s_id = rec.get("segment_id", key)
                    src = rec.get("source", key)
                    seg_source_map.setdefault(s_id, {})[src] = rec
            elif isinstance(val, dict):
                seg_source_map[key] = val

    rows = []

    for seg_id, sources in seg_source_map.items():
        # Cleaned word lists for cross-model agreement comparison
        model_word_lists: dict[str, list[str]] = {}
        for src, rec in sources.items():
            raw_words = [_clean_word(w.get("word", "")) for w in rec.get("words", [])]
            model_word_lists[src] = [w for w in raw_words if w]

        for src, rec in sources.items():
            if src == "human":
                continue  # Gate trains on ASR flags, not human reference feedback

            words = rec.get("words", [])
            n_words = len(words)
            if n_words == 0:
                continue

            # Segment mean confidence (ignoring None values like in CrisperWhisper)
            valid_confs = [w["confidence"] for w in words if w.get("confidence") is not None]
            seg_mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.50

            # Process feedback flags for this record
            flags = rec.get("feedback", [])
            for flag in flags:
                flag_id = flag.get("flag_id", f"{seg_id}:{src}:{len(rows)}")
                w_start = flag.get("word_start", 0)
                w_end = flag.get("word_end", w_start)

                # Clamp indices to word list bounds
                w_start = max(0, min(w_start, n_words - 1))
                w_end = max(w_start, min(w_end, n_words - 1))

                flag_words = words[w_start:w_end + 1]
                flag_text = flag.get("text", " ".join(w.get("word", "") for w in flag_words))
                first_word_clean = _clean_word(flag_words[0].get("word", "")) if flag_words else ""

                # Feature 1: Flag word confidence
                explicit_conf = flag.get("asr_confidence")
                if explicit_conf is not None:
                    word_conf = float(explicit_conf)
                else:
                    word_confs = [w["confidence"] for w in flag_words if w.get("confidence") is not None]
                    word_conf = float(min(word_confs)) if word_confs else seg_mean_conf

                # Features 2 & 3: +-2-word window min and mean confidence
                win_start = max(0, w_start - 2)
                win_end = min(n_words, w_end + 3)
                win_words = words[win_start:win_end]
                win_confs = [w["confidence"] for w in win_words if w.get("confidence") is not None]
                conf_win_min = float(min(win_confs)) if win_confs else word_conf
                conf_win_mean = float(np.mean(win_confs)) if win_confs else word_conf

                # Features 4 & 5: Cross-model agreement (Whisper, CrisperWhisper, Vosk)
                agreed_models = 0
                total_models_available = len(model_word_lists)
                for other_src, other_words in model_word_lists.items():
                    # Check approximate positional match (+-2 words)
                    search_start = max(0, w_start - 2)
                    search_end = min(len(other_words), w_end + 3)
                    candidate_slice = other_words[search_start:search_end]
                    if first_word_clean in candidate_slice:
                        agreed_models += 1

                agreement_3way = 1 if agreed_models >= 3 else 0
                agreement_ratio = (agreed_models / total_models_available) if total_models_available > 0 else 1.0

                # Feature 6: Rule category
                category = flag.get("category", "other")
                if not category:
                    category = "other"

                # Feature 7 & 8: Word frequency (Zipf scale) and word length
                try:
                    word_freq = float(wordfreq.zipf_frequency(first_word_clean, "en")) if first_word_clean else 0.0
                except Exception:
                    word_freq = 0.0
                word_len = len(first_word_clean)

                # Feature 9: Pause before
                pause_before = 0.0
                if w_start > 0:
                    prev_end = words[w_start - 1].get("end")
                    curr_start = flag_words[0].get("start")
                    if prev_end is not None and curr_start is not None and curr_start > prev_end:
                        pause_before = float(curr_start - prev_end)

                # Feature 10: Pause after
                pause_after = 0.0
                if w_end < n_words - 1:
                    curr_end = flag_words[-1].get("end")
                    next_start = words[w_end + 1].get("start")
                    if curr_end is not None and next_start is not None and next_start > curr_end:
                        pause_after = float(next_start - curr_end)

                # Feature 11: Local speech rate (words per second in window)
                win_t_start = win_words[0].get("start")
                win_t_end = win_words[-1].get("end")
                if win_t_start is not None and win_t_end is not None and win_t_end > win_t_start:
                    dur = win_t_end - win_t_start
                    local_rate = float(len(win_words) / dur) if dur > 0.1 else 2.5
                else:
                    local_rate = 2.5

                label = flag.get("label", "unknown")

                rows.append({
                    "flag_id": flag_id,
                    "segment_id": seg_id,
                    "source": src,
                    "rule_id": flag.get("rule_id", "UNKNOWN"),
                    "category": category,
                    "text": flag_text,
                    "label": label,
                    # Numeric model features
                    "confidence": round(word_conf, 4),
                    "conf_window_min": round(conf_win_min, 4),
                    "conf_window_mean": round(conf_win_mean, 4),
                    "agreement_3way": agreement_3way,
                    "agreement_ratio": round(agreement_ratio, 4),
                    "word_freq_zipf": round(word_freq, 4),
                    "word_len": word_len,
                    "pause_before": round(pause_before, 4),
                    "pause_after": round(pause_after, 4),
                    "local_speech_rate": round(local_rate, 4),
                    "segment_mean_confidence": round(seg_mean_conf, 4),
                })

    return pd.DataFrame(rows)
