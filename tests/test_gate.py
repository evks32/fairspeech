"""Unit tests for feedback gate feature extraction, baseline, training, and evaluation. Member B (WP2)."""
import numpy as np
import pandas as pd
import pytest

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


def _make_dummy_records():
    return [
        {
            "segment_id": "seg_1",
            "source": "whisper",
            "words": [
                {"i": 0, "word": "I", "start": 0.0, "end": 0.2, "confidence": 0.95},
                {"i": 1, "word": "has", "start": 0.2, "end": 0.5, "confidence": 0.35},
                {"i": 2, "word": "done", "start": 0.5, "end": 0.9, "confidence": 0.90},
            ],
            "feedback": [
                {
                    "flag_id": "seg_1:whisper:0",
                    "type": "grammar",
                    "rule_id": "HE_VERB_AGR",
                    "category": "subject_verb_agreement",
                    "word_start": 0,
                    "word_end": 1,
                    "text": "I has",
                    "asr_confidence": 0.35,
                    "label": "false",
                    "shown": True,
                }
            ],
        },
        {
            "segment_id": "seg_2",
            "source": "whisper",
            "words": [
                {"i": 0, "word": "um", "start": 0.0, "end": 0.3, "confidence": 0.88},
                {"i": 1, "word": "hello", "start": 0.4, "end": 0.8, "confidence": 0.96},
            ],
            "feedback": [
                {
                    "flag_id": "seg_2:whisper:0",
                    "type": "filler",
                    "rule_id": "FILLER_UM",
                    "category": "filler",
                    "word_start": 0,
                    "word_end": 0,
                    "text": "um",
                    "asr_confidence": 0.88,
                    "label": "genuine",
                    "shown": True,
                }
            ],
        },
    ]


def _make_synthetic_gate_df(n_samples: int = 60) -> pd.DataFrame:
    rows = []
    clusters = ["Native", "East Asian", "European", "South Asian", "African", "Caribbean", "Middle Eastern"]
    for i in range(n_samples):
        is_false = (i % 2 == 0)
        conf = 0.35 if is_false else 0.92
        cluster = clusters[i % len(clusters)]
        speaker = f"spk_{i % 6}"
        conv = f"conv_{i % 3}"
        rows.append({
            "flag_id": f"flag_{i}",
            "segment_id": f"seg_{i}",
            "conversation_id": conv,
            "speaker_id": speaker,
            "accent_cluster": cluster,
            "source": "whisper",
            "label": "false" if is_false else "genuine",
            "confidence": conf,
            "conf_window_min": conf - 0.05,
            "conf_window_mean": conf,
            "agreement_3way": 0 if is_false else 1,
            "agreement_ratio": 0.33 if is_false else 1.0,
            "category": "subject_verb_agreement" if is_false else "filler",
            "rule_id": f"RULE_{i}",
            "word_freq_zipf": 4.5,
            "word_len": 4,
            "pause_before": 0.15,
            "pause_after": 0.10,
            "local_speech_rate": 2.8,
            "segment_mean_confidence": 0.82,
        })
    return pd.DataFrame(rows)


def test_feature_extraction():
    recs = _make_dummy_records()
    df = build_features(recs)
    assert len(df) == 2
    assert "confidence" in df.columns
    assert "conf_window_min" in df.columns
    assert "agreement_3way" in df.columns
    assert "word_freq_zipf" in df.columns
    assert "pause_before" in df.columns
    assert "segment_mean_confidence" in df.columns


def test_excluded_features_not_in_inputs():
    """Verify that accent_cluster and speaker_id are strictly excluded from gate model inputs."""
    df = _make_synthetic_gate_df(20)
    gate = FeedbackGate(model_type="lightgbm")
    gate.fit(df)
    for col in gate.feature_names:
        assert "accent" not in col.lower()
        assert "speaker" not in col.lower()


def test_confidence_threshold_baseline():
    df = _make_synthetic_gate_df(40)
    base = ConfidenceThresholdBaseline(target_genuine_recall=0.90)
    base.fit(df)
    assert 0.0 <= base.threshold <= 1.0
    eval_res = base.evaluate(df)
    assert "genuine_recall" in eval_res
    assert eval_res["genuine_recall"] >= 0.85
    assert "gap_raw" in eval_res
    assert "gap_baseline" in eval_res


def test_conversation_disjoint_split():
    df = _make_synthetic_gate_df(30)
    split_df = make_conversation_disjoint_split(df, train_frac=0.70, val_frac=0.15)
    assert "split" in split_df.columns
    # Check that conversations don't leak between splits
    for conv, sub in split_df.groupby("conversation_id"):
        assert sub["split"].nunique() == 1


def test_gate_training_and_eval():
    df = _make_synthetic_gate_df(50)
    gate, summary = train_gate(df, target_genuine_recall=0.90)
    assert gate is not None
    assert summary["model_type"] == "lightgbm"
    assert 0.0 <= gate.threshold <= 1.0

    eval_res = evaluate_gate(gate, df, n_bootstrap=50)
    assert "pr_auc" in eval_res
    assert "genuine_recall_gate" in eval_res
    assert eval_res["genuine_recall_gate"] >= 0.85
    assert "gap_raw" in eval_res
    assert "gap_gate" in eval_res
    assert "overall_ffr_ci" in eval_res


def test_speaker_bootstrap_ci():
    df = _make_synthetic_gate_df(30)
    def dummy_metric(d):
        return float(np.mean(d["confidence"]))
    pt, low, high = speaker_bootstrap_ci(df, dummy_metric, n_resamples=50)
    assert low <= pt <= high


def test_leave_one_accent_out():
    df = _make_synthetic_gate_df(40)
    res = evaluate_leave_one_accent_out(df, target_genuine_recall=0.90)
    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    assert "held_out_cluster" in res.columns
    assert "raw_ffr" in res.columns
    assert "gated_ffr" in res.columns


def test_shap_analysis():
    df = _make_synthetic_gate_df(20)
    gate = FeedbackGate(model_type="lightgbm")
    gate.fit(df)
    shap_res = run_shap_analysis(gate, df)
    assert "shap_importance" in shap_res or "fallback" in shap_res
