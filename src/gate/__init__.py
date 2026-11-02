"""Member B: feedback gate (RQ2).

Threshold baseline, feature extraction, logistic regression and LightGBM training,
evaluation on test split, speaker bootstrap CIs, and leave-one-accent-out.
See proposal Appendix B.
"""
from __future__ import annotations

from src.gate.baseline import ConfidenceThresholdBaseline
from src.gate.evaluate import (
    evaluate_gate,
    evaluate_leave_one_accent_out,
    run_shap_analysis,
    speaker_bootstrap_ci,
)
from src.gate.features import NUMERIC_FEATURE_COLS, build_features
from src.gate.train import (
    FeedbackGate,
    make_conversation_disjoint_split,
    train_gate,
)

__all__ = [
    "build_features",
    "NUMERIC_FEATURE_COLS",
    "ConfidenceThresholdBaseline",
    "FeedbackGate",
    "train_gate",
    "make_conversation_disjoint_split",
    "evaluate_gate",
    "speaker_bootstrap_ci",
    "evaluate_leave_one_accent_out",
    "run_shap_analysis",
]
