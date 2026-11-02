"""Evaluate the feedback gate on test split and cross-accent generalisation. Member B (WP2).

Specifications from Proposal Appendix B (Table 4):
- PR-AUC, ROC-AUC, Precision, Recall for false flags
- FFR per accent group before gating, with baseline threshold, and with trained gate
- Gap (worst FFR - best FFR) and gap reduction
- Genuine flag recall preserved on test set (must be >= 0.90)
- Speaker-level bootstrap 95% confidence intervals (1,000 resamples)
- Leave-one-accent-out evaluation across all 7 EdAcc clusters
- Ablations: without confidence features, without cross-model agreement
- SHAP feature importance analysis
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.gate.features import NUMERIC_FEATURE_COLS
from src.gate.train import FeedbackGate


def speaker_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute 95% confidence intervals using a speaker-level block bootstrap."""
    speaker_col = "speaker_id" if "speaker_id" in df.columns else ("speaker" if "speaker" in df.columns else "segment_id")
    unique_speakers = df[speaker_col].unique()

    point_estimate = float(metric_fn(df))
    if len(unique_speakers) < 2:
        return point_estimate, point_estimate, point_estimate

    # Group records by speaker for fast lookup
    speaker_groups = [group for _, group in df.groupby(speaker_col)]
    n_speakers = len(speaker_groups)

    rng = np.random.RandomState(random_state)
    estimates = []

    for _ in range(n_resamples):
        sample_indices = rng.choice(n_speakers, size=n_speakers, replace=True)
        sampled_df = pd.concat([speaker_groups[idx] for idx in sample_indices], ignore_index=True)
        try:
            val = float(metric_fn(sampled_df))
            if not np.isnan(val):
                estimates.append(val)
        except Exception:
            continue

    if not estimates:
        return point_estimate, point_estimate, point_estimate

    low = float(np.percentile(estimates, 100 * (alpha / 2.0)))
    high = float(np.percentile(estimates, 100 * (1.0 - alpha / 2.0)))
    return point_estimate, low, high


def evaluate_gate(
    gate: FeedbackGate,
    test_df: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Comprehensive evaluation of feedback gate against baseline and raw ASR flags."""
    df = test_df.copy()
    if segments is not None and "segment_id" in df.columns:
        merge_cols = [c for c in ["segment_id", "speaker_id", "speaker", "accent_cluster", "accent"] if c in segments.columns and c not in df.columns]
        if merge_cols:
            df = df.merge(segments[["segment_id"] + merge_cols].drop_duplicates("segment_id"), on="segment_id", how="left")

    clean = df[df["label"].isin(["false", "genuine"])].copy()
    if len(clean) == 0:
        return {"error": "No labeled test flags"}

    y_true = (clean["label"] == "false").astype(int).values
    probs_false = gate.predict_proba_false(clean)

    # Decision masks
    shown_gate = gate.predict_shown(clean, use_baseline=False)
    shown_baseline = gate.predict_shown(clean, use_baseline=True)

    # Suppressed predictions (True = predicted False/error to suppress)
    pred_suppress_gate = ~shown_gate
    pred_suppress_baseline = ~shown_baseline

    # Classification metrics
    pr_auc = float(average_precision_score(y_true, probs_false))
    try:
        roc_auc = float(roc_auc_score(y_true, probs_false))
    except Exception:
        roc_auc = 0.50

    precision_gate = float(precision_score(y_true, pred_suppress_gate, zero_division=0))
    recall_gate = float(recall_score(y_true, pred_suppress_gate, zero_division=0))

    # Genuine recall (Target >= 0.90)
    is_genuine = (clean["label"] == "genuine").values
    total_genuine = max(1, int(np.sum(is_genuine)))
    genuine_recall_baseline = float(np.sum(is_genuine & shown_baseline) / total_genuine)
    genuine_recall_gate = float(np.sum(is_genuine & shown_gate) / total_genuine)

    # FFR per accent cluster helper
    group_col = "accent_cluster" if "accent_cluster" in clean.columns else "source"
    groups = clean[group_col].unique()

    def _calc_group_ffrs(shown_mask: np.ndarray) -> dict[str, float]:
        res = {}
        for g in groups:
            sub = clean[clean[group_col] == g]
            sub_shown = shown_mask[clean[group_col] == g]
            n_shown = int(np.sum(sub_shown))
            n_false = int(np.sum(sub_shown & (sub["label"] == "false").values))
            ffr = (n_false / n_shown) if n_shown > 0 else 0.0
            res[str(g)] = round(ffr, 4)
        return res

    ffr_raw = _calc_group_ffrs(np.ones(len(clean), dtype=bool))
    ffr_baseline = _calc_group_ffrs(shown_baseline)
    ffr_gate = _calc_group_ffrs(shown_gate)

    gap_raw = float(max(ffr_raw.values()) - min(ffr_raw.values())) if ffr_raw else 0.0
    gap_baseline = float(max(ffr_baseline.values()) - min(ffr_baseline.values())) if ffr_baseline else 0.0
    gap_gate = float(max(ffr_gate.values()) - min(ffr_gate.values())) if ffr_gate else 0.0

    # Overall FFR bootstrap CI
    def _overall_ffr(sub_df: pd.DataFrame) -> float:
        shown = gate.predict_shown(sub_df, use_baseline=False)
        n_shown = np.sum(shown)
        n_false = np.sum(shown & (sub_df["label"] == "false").values)
        return float(n_false / n_shown) if n_shown > 0 else 0.0

    ffr_pt, ffr_ci_low, ffr_ci_high = speaker_bootstrap_ci(
        clean, _overall_ffr, n_resamples=min(1000, n_bootstrap)
    )

    # Feature Importance (LightGBM gain or LR weights)
    feature_importance: dict[str, float] = {}
    if hasattr(gate.model, "feature_importances_"):
        for name, imp in zip(gate.feature_names, gate.model.feature_importances_):
            feature_importance[name] = float(imp)
    elif hasattr(gate.model, "coef_"):
        for name, coef in zip(gate.feature_names, gate.model.coef_[0]):
            feature_importance[name] = float(abs(coef))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision_false_flags": round(precision_gate, 4),
        "recall_false_flags": round(recall_gate, 4),
        "genuine_recall_baseline": round(genuine_recall_baseline, 4),
        "genuine_recall_gate": round(genuine_recall_gate, 4),
        "ffr_by_group_raw": ffr_raw,
        "ffr_by_group_baseline": ffr_baseline,
        "ffr_by_group_gate": ffr_gate,
        "gap_raw": round(gap_raw, 4),
        "gap_baseline": round(gap_baseline, 4),
        "gap_gate": round(gap_gate, 4),
        "gap_reduction": round(gap_raw - gap_gate, 4),
        "overall_ffr_ci": {
            "point": round(ffr_pt, 4),
            "ci_95_low": round(ffr_ci_low, 4),
            "ci_95_high": round(ffr_ci_high, 4),
        },
        "feature_importance": feature_importance,
    }


def evaluate_leave_one_accent_out(
    df: pd.DataFrame,
    target_genuine_recall: float = 0.90,
) -> pd.DataFrame:
    """Perform leave-one-accent-out evaluation across all 7 accent clusters in EdAcc."""
    if "accent_cluster" not in df.columns:
        raise ValueError("accent_cluster column required for leave-one-accent-out evaluation.")

    clusters = sorted(df["accent_cluster"].dropna().unique())
    results = []

    for held_out in clusters:
        train_data = df[df["accent_cluster"] != held_out].copy()
        test_data = df[df["accent_cluster"] == held_out].copy()

        if len(test_data) == 0 or len(train_data) == 0:
            continue

        gate = FeedbackGate(model_type="lightgbm", target_genuine_recall=target_genuine_recall)
        try:
            gate.fit(train_data, val_df=None)
            eval_res = evaluate_gate(gate, test_data, n_bootstrap=100)

            # Compute test cluster FFR raw vs gated
            raw_ffr = eval_res["ffr_by_group_raw"].get(str(held_out), 0.0)
            gated_ffr = eval_res["ffr_by_group_gate"].get(str(held_out), 0.0)

            results.append({
                "held_out_cluster": held_out,
                "n_flags": len(test_data),
                "pr_auc": eval_res.get("pr_auc", 0.0),
                "genuine_recall": eval_res.get("genuine_recall_gate", 0.0),
                "raw_ffr": raw_ffr,
                "gated_ffr": gated_ffr,
                "ffr_reduction": round(raw_ffr - gated_ffr, 4),
            })
        except Exception as e:
            results.append({
                "held_out_cluster": held_out,
                "n_flags": len(test_data),
                "error": str(e),
            })

    return pd.DataFrame(results)


def run_shap_analysis(gate: FeedbackGate, df: pd.DataFrame) -> dict[str, Any]:
    """Compute SHAP feature importances using TreeExplainer (LightGBM) or LinearExplainer (LR)."""
    clean = df[df["label"].isin(["false", "genuine"])].copy()
    if len(clean) == 0:
        return {"error": "No valid flags for SHAP"}

    X = gate._prepare_features(clean, fit=False)

    try:
        import shap
        if gate.model_type == "lightgbm":
            explainer = shap.TreeExplainer(gate.model)
            shap_values = explainer.shap_values(X)
            # Binary classification: use positive class values
            vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_abs_shap = np.mean(np.abs(vals), axis=0)
        else:
            explainer = shap.LinearExplainer(gate.model, X)
            shap_values = explainer.shap_values(X)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        shap_importance = {
            name: round(float(val), 4)
            for name, val in zip(gate.feature_names, mean_abs_shap)
        }
        # Sort descending
        shap_sorted = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
        return {"shap_importance": shap_sorted}
    except Exception as e:
        return {"error": str(e), "fallback": gate.feature_importance if hasattr(gate, "feature_importance") else {}}
