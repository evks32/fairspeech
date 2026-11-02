# ASR Inference & Evaluation (Work Package 2 – Member B)

**Lead:** Sanjay Eshwaradurai Vasantthisri (`Member B`, ST93197)

This directory contains the Automatic Speech Recognition (ASR) inference pipelines, text normalisation routines, and Word Error Rate (WER) evaluation suite for the FairSpeech project.

## Models Evaluated

1. **Whisper (`openai/whisper-large-v3` / `faster-whisper`)**:
   - Primary high-accuracy ASR system.
   - Extracts word-level start/end timestamps and word confidence scores (token probabilities).
2. **CrisperWhisper (`nyrahealth/CrisperWhisper`)**:
   - Interspeech 2024 model specialized in verbatim transcription and conversational speech disfluencies (fillers: *um*, *uh*, *erm*).
   - Used for the cross-model agreement feature in the feedback gate.
   - Per specification (Table 4), CrisperWhisper produces no word confidence scores by default and is used for agreement.
3. **Vosk (Kaldi baseline)**:
   - Traditional offline HMM-DNN speech recognition model.
   - Fast, local CPU inference with word timestamps and posterior confidences.

## Directory Structure

- [`normalisation.py`](file:///src/asr/normalisation.py): Standardises text for transcript alignment and evaluation (lowercasing, punctuation stripping while preserving contractions, noise tag removal).
- [`eval_wer.py`](file:///src/asr/eval_wer.py): Word Error Rate (WER) computation per turn and aggregate group WER by accent cluster. Also identifies confident errors (high confidence words that are ASR mistakes) and repetition hallucinations.
- [`whisper_asr.py`](file:///src/asr/whisper_asr.py): Whisper wrapper with word timestamp and confidence extraction.
- [`crisper_whisper.py`](file:///src/asr/crisper_whisper.py): CrisperWhisper wrapper for verbatim transcription.
- [`vosk_asr.py`](file:///src/asr/vosk_asr.py): Vosk offline baseline wrapper.
- [`transcribe.py`](file:///src/asr/transcribe.py): Unified transcription interface producing valid FairSpeech feedback schema records.

## Usage

```python
from src.asr import transcribe_multi, compute_group_wer

# Run 3-way transcription across Whisper, CrisperWhisper, and Vosk
records = transcribe_multi("audio.wav", segment_id="edacc_c001_s01_t001")

# Compute WER per accent cluster
group_wer_df = compute_group_wer(df, group_col="accent_cluster")
```
