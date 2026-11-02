"""Vosk offline baseline ASR wrapper. Member B (WP2).

Vosk provides a traditional Kaldi/HMM-DNN offline baseline ASR system:
- Fast, local CPU inference
- Word-level timestamps and posterior confidence scores
- Used as a baseline model and for cross-model agreement in the feedback gate
"""
from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoskASR:
    """Wrapper for Vosk speech recognition."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load Vosk model if available."""
        if not self.model_path:
            logger.info("Vosk model path not specified. Operating in mock/simulation mode.")
            return

        try:
            import vosk
            if Path(self.model_path).exists():
                self.model = vosk.Model(str(self.model_path))
                logger.info("Loaded Vosk model from: %s", self.model_path)
            else:
                logger.warning("Vosk model path not found: %s", self.model_path)
        except ImportError:
            logger.info("Vosk package not installed. Operating in mock/simulation mode.")

    def transcribe(self, audio_path: str | Path, fallback_text: str | None = None) -> dict[str, Any]:
        """Transcribe a 16kHz mono WAV audio file with word timestamps and confidence."""
        if self.model is not None and Path(audio_path).exists():
            import vosk
            wf = wave.open(str(audio_path), "rb")
            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            words = []
            full_text_parts = []
            word_idx = 0

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    full_text_parts.append(res.get("text", ""))
                    for w in res.get("result", []):
                        words.append({
                            "i": word_idx,
                            "word": w.get("word", "").strip(),
                            "start": round(float(w.get("start", 0.0)), 3),
                            "end": round(float(w.get("end", 0.0)), 3),
                            "confidence": round(float(w.get("conf", 0.80)), 4),
                        })
                        word_idx += 1

            final_res = json.loads(rec.FinalResult())
            full_text_parts.append(final_res.get("text", ""))
            for w in final_res.get("result", []):
                words.append({
                    "i": word_idx,
                    "word": w.get("word", "").strip(),
                    "start": round(float(w.get("start", 0.0)), 3),
                    "end": round(float(w.get("end", 0.0)), 3),
                    "confidence": round(float(w.get("conf", 0.80)), 4),
                })
                word_idx += 1

            return {
                "source": "vosk",
                "model": "vosk-kaldi-baseline",
                "transcript": " ".join([p for p in full_text_parts if p]).strip(),
                "words": words,
            }

        # Mock / simulation mode
        text = fallback_text or "I has led three project in my last job"
        raw_words = text.split()
        words = []
        cur_t = 0.4
        for i, w in enumerate(raw_words):
            dur = max(0.18, len(w) * 0.075)
            # Kaldi confidence scores are typically slightly lower than Whisper
            conf = 0.82 if w.lower() not in ["has", "project"] else 0.41
            words.append({
                "i": i,
                "word": w,
                "start": round(cur_t, 3),
                "end": round(cur_t + dur, 3),
                "confidence": conf,
            })
            cur_t += dur + 0.05

        return {
            "source": "vosk",
            "model": "mock-vosk",
            "transcript": text,
            "words": words,
        }
