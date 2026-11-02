"""Unified ASR transcription interface and multi-model inference runner. Member B (WP2).

Conforms strictly to the FairSpeech data contract (docs/data_contracts.md):
- schema_version: "0.1"
- segment_id: Unique identifier matching segment table
- source: "whisper" | "crisperwhisper" | "vosk" | "human"
- model: Identifier of model used
- transcript: Full text as produced
- words: Array of {i, word, start, end, confidence}
- created_at: ISO 8601 UTC timestamp
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.asr.crisper_whisper import CrisperWhisperASR
from src.asr.vosk_asr import VoskASR
from src.asr.whisper_asr import WhisperASR


def format_asr_record(
    segment_id: str,
    source: str,
    model: str,
    transcript: str,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format ASR output into the standard data contract dictionary."""
    return {
        "schema_version": "0.1",
        "segment_id": segment_id,
        "source": source,
        "model": model,
        "transcript": transcript,
        "words": words,
        "fluency": None,
        "feedback": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def transcribe_segment(
    audio_path: str | Path,
    segment_id: str = "seg_001",
    model: str = "whisper",
    fallback_text: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Transcribe a single audio segment using the designated model."""
    model_lower = model.lower()
    if "crisper" in model_lower:
        asr = CrisperWhisperASR(**kwargs)
        res = asr.transcribe(audio_path, fallback_text=fallback_text)
    elif "vosk" in model_lower:
        asr = VoskASR(**kwargs)
        res = asr.transcribe(audio_path, fallback_text=fallback_text)
    else:
        asr = WhisperASR(**kwargs)
        res = asr.transcribe(audio_path, fallback_text=fallback_text)

    return format_asr_record(
        segment_id=segment_id,
        source=res["source"],
        model=res["model"],
        transcript=res["transcript"],
        words=res["words"],
    )


def transcribe_multi(
    audio_path: str | Path,
    segment_id: str = "seg_001",
    fallback_texts: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Transcribe an audio segment across Whisper, CrisperWhisper, and Vosk for cross-model agreement."""
    fallbacks = fallback_texts or {}
    results = {}

    for src in ["whisper", "crisperwhisper", "vosk"]:
        fb_txt = fallbacks.get(src)
        rec = transcribe_segment(
            audio_path=audio_path,
            segment_id=segment_id,
            model=src,
            fallback_text=fb_txt,
        )
        results[src] = rec

    return results
