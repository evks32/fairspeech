"""Global confidence threshold baseline gating (RQ2 baseline). Member B (WP2).

As defined in Proposal Appendix B:
- Baseline model: A single global threshold on the ASR confidence score of the flagged word.
- Decision rule: A flag is SHOWN if word confidence >= threshold; SUPPRESSED if confidence < threshold.
- Threshold selection: Swept over [0.0, 1.0] on the validation split to achieve at least 90%
  recall of genuine flags (kept genuine flags / all genuine flags >= 0.90).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class ConfidenceThresholdBaseline:
    """Confidence threshold baseline filter for feedback flags."""

    def __init__(self, target_genuine_recall: float = 0.90):
        self.target_genuine_recall = target_genuine_recall
        self.threshold: float = 0.50
        self.calibrated_recall: float = 1.0

    def fit(self, val_df: pd.DataFrame) -> "ConfidenceThresholdBaseline":
        """Calibrate the threshold on validation split to satisfy target genuine recall."""
        clean = val_df[val_df["label"].isin(["false", "genuine"])].copy()
        if len(clean) == 0:
            self.threshold = 0.50
            return self

        confs = clean["confidence"].fillna(0.0).values
        is_genuine = (clean["label"] == "genuine").values
        total_genuine = np.sum(is_genuine)

        if total_genuine == 0:
            self.threshold = 0.50
            return self

        # Candidate thresholds from 0.0 to 1.0 in steps of 0.01
        candidates = np.linspace(0.0, 1.0, 101)
        best_t = 0.0
        best_rec = 1.0

        for t in candidates:
            kept_genuine = np.sum(is_genuine & (confs >= t))
            rec = kept_genuine / total_genuine
            if rec >= self.target_genuine_recall:
                best_t = t  # choose highest threshold that still preserves >= 90% genuine flags
                best_rec = rec

        self.threshold = float(best_t)
        self.calibrated_recall = float(best_rec)
        return self

    def predict_shown(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean array indicating whether each flag is shown (True) or suppressed (False)."""
        confs = df["confidence"].fillna(0.0).values
        return confs >= self.threshold

    def evaluate(self, test_df: pd.DataFrame, group_col: str = "accent_cluster") -> dict[str, Any]:
        """Evaluate baseline on test split."""
        clean = test_df[test_df["label"].isin(["false", "genuine"])].copy()
        if len(clean) == 0:
            return {"error": "No valid labeled test flags"}

        shown = self.predict_shown(clean)
        is_genuine = (clean["label"] == "genuine").values
        is_false = (clean["label"] == "false").values

        total_genuine = int(np.sum(is_genuine))
        kept_genuine = int(np.sum(is_genuine & shown))
        genuine_recall = (kept_genuine / total_genuine) if total_genuine > 0 else 1.0

        total_false = int(np.sum(is_false))
        suppressed_false = int(np.sum(is_false & ~shown))
        false_suppression_rate = (suppressed_false / total_false) if total_false > 0 else 0.0

        # FFR per accent cluster
        groups = clean[group_col].unique() if group_col in clean.columns else ["all"]
        ffr_raw: dict[str, float] = {}
        ffr_baseline: dict[str, float] = {}

        for g in groups:
            mask = (clean[group_col] == g) if group_col in clean.columns else np.ones(len(clean), dtype=bool)
            sub = clean[mask]
            sub_shown = shown[mask]

            n_total = len(sub)
            n_raw_false = int(np.sum(sub["label"] == "false"))
            ffr_raw[str(g)] = round(n_raw_false / max(1, n_total), 4)

            n_kept = int(np.sum(sub_shown))
            n_kept_false = int(np.sum(sub_shown & (sub["label"] == "false").values))
            ffr_baseline[str(g)] = round(n_kept_false / max(1, n_kept), 4)

        gap_raw = float(max(ffr_raw.values()) - min(ffr_raw.values())) if ffr_raw else 0.0
        gap_baseline = float(max(ffr_baseline.values()) - min(ffr_baseline.values())) if ffr_baseline else 0.0

        return {
            "threshold": self.threshold,
            "genuine_recall": round(genuine_recall, 4),
            "false_suppression_rate": round(false_suppression_rate, 4),
            "ffr_raw": ffr_raw,
            "ffr_baseline": ffr_baseline,
            "gap_raw": round(gap_raw, 4),
            "gap_baseline": round(gap_baseline, 4),
            "gap_reduction": round(gap_raw - gap_baseline, 4),
        }
