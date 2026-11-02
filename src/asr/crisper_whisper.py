"""CrisperWhisper ASR wrapper for verbatim transcription and filler capture. Member B (WP2).

CrisperWhisper (Zusag et al., Interspeech 2024; nyrahealth/CrisperWhisper) provides:
- High-precision word timestamps
- Verbatim transcript preservation (spoken disfluencies and fillers like um, uh)
- Used for cross-model agreement in the feedback gate (Table 4)

Note per specification (Table 4): CrisperWhisper gives no word confidence by default
and is used primarily for the cross-model agreement feature.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CrisperWhisperASR:
    """Wrapper for CrisperWhisper inference."""

    def __init__(self, model_id: str = "nyrahealth/CrisperWhisper", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load CrisperWhisper via transformers pipeline; fallback to mock if unavailable."""
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                return_timestamps="word",
            )
            logger.info("Loaded CrisperWhisper model: %s", self.model_id)
        except Exception:
            self.pipeline = None
            logger.info("CrisperWhisper not loaded. Operating in mock/simulation mode.")

    def transcribe(self, audio_path: str | Path, fallback_text: str | None = None) -> dict[str, Any]:
        """Transcribe audio returning verbatim words and timestamps (confidence is None per Table 4)."""
        if self.pipeline is not None:
            res = self.pipeline(str(audio_path))
            chunks = res.get("chunks", [])
            words = []
            for i, chunk in enumerate(chunks):
                ts = chunk.get("timestamp", (None, None))
                words.append({
                    "i": i,
                    "word": chunk.get("text", "").strip(),
                    "start": round(float(ts[0]), 3) if ts[0] is not None else None,
                    "end": round(float(ts[1]), 3) if ts[1] is not None else None,
                    "confidence": None,  # CrisperWhisper provides no word confidences by default
                })
            return {
                "source": "crisperwhisper",
                "model": self.model_id,
                "transcript": res.get("text", "").strip(),
                "words": words,
            }

        # Mock / fallback mode
        text = fallback_text or "um I has led three projects in my last job"
        raw_words = text.split()
        words = []
        cur_t = 0.1
        for i, w in enumerate(raw_words):
            dur = max(0.2, len(w) * 0.08)
            words.append({
                "i": i,
                "word": w,
                "start": round(cur_t, 3),
                "end": round(cur_t + dur, 3),
                "confidence": None,  # Per Table 4 specification
            })
            cur_t += dur + 0.05

        return {
            "source": "crisperwhisper",
            "model": "mock-crisperwhisper",
            "transcript": text,
            "words": words,
        }
