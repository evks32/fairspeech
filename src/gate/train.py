"""Train the feedback gate classifier (RQ2). Member B (WP2).

Specifications from Proposal Appendix B (Table 4):
- Models:
  1. Logistic Regression with L2 regularisation
  2. LightGBM gradient-boosted trees (main model)
- Cross-validation: Grouped 5-fold cross-validation (groups = conversation_id) on the training set only
- Imbalance and fairness: class weights ('balanced') + sample weights balancing accent group contributions
- Decision threshold: Calibrated on the validation split so genuine-flag recall >= 0.90
- Target unit: One grammar or filler flag (1 = False / ASR error, 0 = Genuine)
- Excluded features: accent_cluster and speaker_id must NOT be input features
"""
from __future__ import annotations

from typing import Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.gate.baseline import ConfidenceThresholdBaseline
from src.gate.features import NUMERIC_FEATURE_COLS


class FeedbackGate:
    """Feedback gate classifier predicting whether an ASR feedback flag is false (ASR-induced)."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        target_genuine_recall: float = 0.90,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.target_genuine_recall = target_genuine_recall
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: list[str] = []
        self.threshold: float = 0.50
        self.baseline = ConfidenceThresholdBaseline(target_genuine_recall=target_genuine_recall)
        self.model: Any = None
        self.cv_results: dict[str, Any] = {}

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract and preprocess feature matrix (excluding speaker_id and accent_cluster)."""
        # Ensure all numeric feature columns exist
        num_df = pd.DataFrame(index=df.index)
        for col in NUMERIC_FEATURE_COLS:
            num_df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)

        num_vals = num_df.values

        # Category column one-hot encoded
        cat_vals = df[["category"]].fillna("other").astype(str).values if "category" in df.columns else np.full((len(df), 1), "other")

        if fit:
            num_scaled = self.scaler.fit_transform(num_vals)
            cat_encoded = self.encoder.fit_transform(cat_vals)
            cat_names = list(self.encoder.get_feature_names_out(["category"]))
            self.feature_names = list(NUMERIC_FEATURE_COLS) + cat_names
        else:
            num_scaled = self.scaler.transform(num_vals)
            cat_encoded = self.encoder.transform(cat_vals)

        return np.hstack([num_scaled, cat_encoded])

    def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Compute sample weights to balance accent group contributions to the loss."""
        if "accent_cluster" in df.columns and len(df["accent_cluster"].dropna()) > 0:
            counts = df["accent_cluster"].value_counts()
            n_total = len(df)
            n_groups = len(counts)
            group_weights = {grp: n_total / (n_groups * cnt) for grp, cnt in counts.items()}
            return df["accent_cluster"].map(group_weights).fillna(1.0).values
        return np.ones(len(df), dtype=float)

    def cross_validate(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Perform Grouped 5-Fold Cross-Validation on training set grouped by conversation."""
        clean = train_df[train_df["label"].isin(["false", "genuine"])].copy()
        if len(clean) == 0:
            return {"pr_auc_mean": 0.0, "roc_auc_mean": 0.0}

        # Groups for GroupKFold: conversation_id (fallback to segment_id)
        if "conversation_id" in clean.columns and clean["conversation_id"].nunique() >= n_splits:
            groups = clean["conversation_id"].values
        elif "speaker_id" in clean.columns and clean["speaker_id"].nunique() >= n_splits:
            groups = clean["speaker_id"].values
        else:
            groups = clean["segment_id"].values

        n_actual_splits = min(n_splits, len(np.unique(groups)))
        if n_actual_splits < 2:
            return {"pr_auc_mean": 0.50, "roc_auc_mean": 0.50}

        gkf = GroupKFold(n_splits=n_actual_splits)
        pr_aucs = []
        roc_aucs = []

        X = self._prepare_features(clean, fit=True)
        y = (clean["label"] == "false").astype(int).values
        weights = self._compute_sample_weights(clean)

        for tr_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], weights[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue

            if self.model_type == "lightgbm":
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=5,
                    random_state=self.random_state,
                    class_weight="balanced",
                    verbose=-1,
                )
            else:
                clf = LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=1000,
                )

            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            probs = clf.predict_proba(X_va)[:, 1]

            pr_aucs.append(float(average_precision_score(y_va, probs)))
            roc_aucs.append(float(roc_auc_score(y_va, probs)))

        self.cv_results = {
            "pr_auc_mean": round(float(np.mean(pr_aucs)), 4) if pr_aucs else 0.50,
            "pr_auc_std": round(float(np.std(pr_aucs)), 4) if pr_aucs else 0.0,
            "roc_auc_mean": round(float(np.mean(roc_aucs)), 4) if roc_aucs else 0.50,
            "roc_auc_std": round(float(np.std(roc_aucs)), 4) if roc_aucs else 0.0,
        }
        return self.cv_results

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "FeedbackGate":
        """Fit model on train_df and calibrate decision threshold on val_df."""
        clean_train = train_df[train_df["label"].isin(["false", "genuine"])].copy()
        if len(clean_train) == 0:
            raise ValueError("No valid training flags with labels 'false' or 'genuine'.")

        y_train = (clean_train["label"] == "false").astype(int).values
        X_train = self._prepare_features(clean_train, fit=True)
        sample_weight = self._compute_sample_weights(clean_train)

        # Run CV for diagnostic reporting
        self.cross_validate(clean_train)

        # Fit chosen classifier
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                random_state=self.random_state,
                class_weight="balanced",
                verbose=-1,
            )
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model = LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1000,
            )
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

        # Calibrate baseline confidence threshold
        eval_df = val_df if (val_df is not None and len(val_df) > 0) else clean_train
        self.baseline.fit(eval_df)

        # Calibrate decision threshold on validation set
        # Flag is SHOWN if prob_false < threshold
        # Genuine recall = kept genuine flags / total genuine flags
        eval_clean = eval_df[eval_df["label"].isin(["false", "genuine"])].copy()
        if len(eval_clean) > 0:
            probs = self.predict_proba_false(eval_clean)
            is_genuine = (eval_clean["label"] == "genuine").values
            total_genuine = np.sum(is_genuine)

            if total_genuine > 0:
                best_t = 0.50
                # Search candidate thresholds
                for t in np.linspace(0.01, 0.99, 100):
                    kept_genuine = np.sum(is_genuine & (probs < t))
                    rec = kept_genuine / total_genuine
                    if rec >= self.target_genuine_recall:
                        best_t = t  # pick largest threshold that satisfies genuine recall >= target
                self.threshold = float(best_t)
            else:
                self.threshold = 0.50

        return self

    def predict_proba_false(self, df: pd.DataFrame) -> np.ndarray:
        """Return estimated probability of each flag being False (ASR error)."""
        X = self._prepare_features(df, fit=False)
        return self.model.predict_proba(X)[:, 1]

    def predict_shown(self, df: pd.DataFrame, use_baseline: bool = False) -> np.ndarray:
        """Return boolean mask indicating whether flags should be shown to learner."""
        if use_baseline:
            return self.baseline.predict_shown(df)
        probs_false = self.predict_proba_false(df)
        return probs_false < self.threshold


def make_conversation_disjoint_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a 70/15/15 conversation-disjoint split ensuring no speaker/conversation leak."""
    out = df.copy()
    unit_col = "conversation_id" if "conversation_id" in out.columns else ("speaker_id" if "speaker_id" in out.columns else "segment_id")

    rng = np.random.RandomState(random_state)
    unique_units = pd.Series(out[unit_col].unique())
    shuffled_units = unique_units.sample(frac=1.0, random_state=random_state).values

    n_tot = len(shuffled_units)
    n_tr = int(n_tot * train_frac)
    n_va = int(n_tot * val_frac)

    train_set = set(shuffled_units[:n_tr])
    val_set = set(shuffled_units[n_tr:n_tr + n_va])
    test_set = set(shuffled_units[n_tr + n_va:])

    split_map = {}
    for u in train_set:
        split_map[u] = "train"
    for u in val_set:
        split_map[u] = "val"
    for u in test_set:
        split_map[u] = "test"

    out["split"] = out[unit_col].map(split_map).fillna("train")
    return out


def train_gate(
    features: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    model_type: str = "lightgbm",
    target_genuine_recall: float = 0.90,
) -> Tuple[FeedbackGate, dict[str, Any]]:
    """Fit feedback gate and return trained model and training summary."""
    df = features.copy()
    if segments is not None and "segment_id" in df.columns:
        merge_cols = [c for c in ["segment_id", "split", "conversation_id", "speaker_id", "speaker", "accent_cluster", "accent"] if c in segments.columns and c not in df.columns]
        if merge_cols:
            df = df.merge(segments[["segment_id"] + merge_cols].drop_duplicates("segment_id"), on="segment_id", how="left")

    if "split" not in df.columns or "train" not in df["split"].values:
        df = make_conversation_disjoint_split(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    gate = FeedbackGate(model_type=model_type, target_genuine_recall=target_genuine_recall)
    gate.fit(train_df, val_df)

    summary = {
        "model_type": gate.model_type,
        "calibrated_gate_threshold": round(gate.threshold, 4),
        "baseline_confidence_threshold": round(gate.baseline.threshold, 4),
        "target_genuine_recall": target_genuine_recall,
        "n_train_flags": len(train_df),
        "n_val_flags": len(val_df),
        "n_test_flags": len(test_df),
        "cv_results": gate.cv_results,
        "feature_names": gate.feature_names,
    }
    return gate, summary
