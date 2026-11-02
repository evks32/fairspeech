"""Member B: Whisper, CrisperWhisper and Vosk inference, WER evaluation and confidence gating.

Exports:
- normalise_text, tokenize_words
- compute_wer, compute_group_wer, compute_detailed_metrics
- detect_confident_errors, detect_hallucinations
- WhisperASR, CrisperWhisperASR, VoskASR
- transcribe_segment, transcribe_multi, format_asr_record
"""
from __future__ import annotations

from src.asr.crisper_whisper import CrisperWhisperASR
from src.asr.eval_wer import (
    compute_detailed_metrics,
    compute_group_wer,
    compute_wer,
    detect_confident_errors,
    detect_hallucinations,
)
from src.asr.normalisation import normalise_text, tokenize_words
from src.asr.transcribe import format_asr_record, transcribe_multi, transcribe_segment
from src.asr.vosk_asr import VoskASR
from src.asr.whisper_asr import WhisperASR

__all__ = [
    "normalise_text",
    "tokenize_words",
    "compute_wer",
    "compute_group_wer",
    "compute_detailed_metrics",
    "detect_confident_errors",
    "detect_hallucinations",
    "WhisperASR",
    "CrisperWhisperASR",
    "VoskASR",
    "transcribe_segment",
    "transcribe_multi",
    "format_asr_record",
]
