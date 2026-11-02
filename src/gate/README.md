# Feedback Gate (Work Package 2 – Member B)

**Lead:** Sanjay Eshwaradurai Vasantthisri (`Member B`, ST93197)

The FairSpeech feedback gate acts as a middleware between speech recognition and language feedback engines, mitigating the propagation of ASR recognition errors into unfair learner penalties (addressing **RQ2**).

## Gate Specification (Proposal Appendix B)

### 1. Target Unit & Labels
- **Target unit:** One grammar or filler flag generated from an ASR transcript.
- **Labels:** `false` (ASR-induced error) vs. `genuine` (reproduced from human reference transcript). Ambiguous alignments are excluded from training.

### 2. Feature Architecture
- **Flagged word confidence:** ASR confidence / probability of the flagged word.
- **Context window confidence:** Min and mean confidence in a $\pm 2$-word context window.
- **Cross-model agreement:** Exact 3-way agreement (`agreement_3way`) and agreement ratio across Whisper, CrisperWhisper, and Vosk.
- **Rule category:** One-hot encoded grammar/filler category (e.g. `subject_verb_agreement`, `filler`, `tense`).
- **Lexical features:** Word frequency on the Zipf scale (`wordfreq`) and word length.
- **Timing & Prosody:** Pause duration immediately preceding and following the flag, and local speech rate.
- **Segment summary:** Mean confidence of all words in the turn.
- **Strictly Excluded Features:** **Accent cluster/label and speaker ID are strictly excluded** from model inputs to ensure the classifier evaluates speech recognition reliability, not speaker identity.

### 3. Models & Training Protocol
- **Baseline:** Global confidence threshold sweep $\tau \in [0.0, 1.0]$.
- **Learned Models:**
  - Logistic Regression ($L_2$ regularised).
  - LightGBM gradient-boosted decision trees (main model).
- **Cross-Validation:** Grouped 5-fold cross-validation (`GroupKFold` on `conversation_id`) strictly on the training set.
- **Imbalance & Fairness:** Class weighting (`balanced`) plus sample weighting balancing representation across all 7 EdAcc accent clusters.
- **Decision Threshold Constraint:** Calibrated on the validation set such that **genuine-flag recall is maintained at $\ge 90\%$** ($\ge 0.90$).

### 4. Evaluation Suite
- **Metrics:** Precision, Recall, PR-AUC, ROC-AUC for false flags; False Feedback Rate ($\text{FFR}_g$) before and after gating per accent cluster; accent gap ($\max_g \text{FFR}_g - \min_g \text{FFR}_g$) and gap reduction.
- **Confidence Intervals:** 95% confidence intervals from a 1,000-iteration speaker-level block bootstrap.
- **Leave-One-Accent-Out Evaluation:** Systematic cross-validation holding out each of the 7 accent clusters in turn to verify cross-accent generalisation.
- **SHAP Analysis:** TreeExplainer feature importance attribution ranking features by contribution to error suppression.

## Directory Structure

- [`features.py`](file:///src/gate/features.py): Feature extraction pipeline conforming to Table 4.
- [`baseline.py`](file:///src/gate/baseline.py): Global confidence threshold sweep and evaluation.
- [`train.py`](file:///src/gate/train.py): Grouped CV training, class/sample weighting, and validation threshold calibration.
- [`evaluate.py`](file:///src/gate/evaluate.py): Comprehensive evaluation, bootstrap CIs, leave-one-accent-out, and SHAP analysis.
