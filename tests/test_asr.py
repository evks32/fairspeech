"""Unit tests for ASR inference, normalisation, and WER evaluation. Member B (WP2)."""
from pathlib import Path
import pandas as pd
import pytest

from src.asr.crisper_whisper import CrisperWhisperASR
from src.asr.eval_wer import (
    compute_detailed_metrics,
    compute_group_wer,
    compute_wer,
    detect_confident_errors,
    detect_hallucinations,
)
from src.asr.normalisation import normalise_text, tokenize_words
from src.asr.transcribe import transcribe_multi, transcribe_segment
from src.asr.vosk_asr import VoskASR
from src.asr.whisper_asr import WhisperASR


def test_normalise_text_basic():
    raw = "Hello, world! I've [laughter] done 3 projects."
    norm = normalise_text(raw)
    assert "laughter" not in norm
    assert "!" not in norm
    assert "," not in norm
    assert "i've" in norm
    assert norm == "hello world i've done 3 projects"


def test_normalise_text_fillers():
    raw = "Um, uh, you know, I think... erm, it worked."
    norm = normalise_text(raw)
    assert "um" in norm
    assert "uh" in norm
    assert "erm" in norm
    assert "you know" in norm


def test_tokenize_words():
    words = tokenize_words("I has led three projects.")
    assert words == ["i", "has", "led", "three", "projects"]


def test_compute_wer_identical():
    ref = "I have led three projects"
    hyp = "I have led three projects"
    assert compute_wer(ref, hyp) == 0.0


def test_compute_wer_different():
    ref = "I have led three projects"
    hyp = "I has led three project"
    wer = compute_wer(ref, hyp)
    assert 0.0 < wer <= 1.0


def test_compute_detailed_metrics():
    ref = "I have led three projects"
    hyp = "I has led three projects today"
    metrics = compute_detailed_metrics(ref, hyp)
    assert metrics["substitutions"] == 1
    assert metrics["insertions"] == 1
    assert metrics["hits"] == 4
    assert metrics["wer"] > 0.0


def test_compute_group_wer():
    df = pd.DataFrame([
        {"human_transcript": "I have led", "asr_transcript": "I have led", "accent_cluster": "Native"},
        {"human_transcript": "I have led", "asr_transcript": "I has led", "accent_cluster": "East Asian"},
        {"human_transcript": "three projects", "asr_transcript": "tree project", "accent_cluster": "East Asian"},
    ])
    res = compute_group_wer(df, group_col="accent_cluster")
    assert len(res) == 2
    assert "Native" in res["accent_cluster"].values
    assert "East Asian" in res["accent_cluster"].values
    native_wer = res[res["accent_cluster"] == "Native"]["mean_wer"].values[0]
    asian_wer = res[res["accent_cluster"] == "East Asian"]["mean_wer"].values[0]
    assert native_wer == 0.0
    assert asian_wer > 0.0


def test_detect_confident_errors():
    words = [
        {"word": "I", "confidence": 0.98},
        {"word": "has", "confidence": 0.91},
        {"word": "done", "confidence": 0.95},
    ]
    ref = "I have done"
    errs = detect_confident_errors(words, ref, confidence_threshold=0.85)
    assert len(errs) == 1
    assert errs[0]["word"] == "has"


def test_detect_hallucinations():
    hyp = "thank you thank you thank you"
    halls = detect_hallucinations(hyp, min_repeat=3)
    assert "thank you" in halls


def test_asr_wrappers_and_transcription():
    # Test Whisper wrapper in simulation mode
    w_asr = WhisperASR(model_size="tiny", device="cpu")
    w_res = w_asr.transcribe("dummy.wav", fallback_text="test transcript here")
    assert w_res["source"] == "whisper"
    assert len(w_res["words"]) == 3
    assert all("confidence" in w for w in w_res["words"])

    # Test CrisperWhisper wrapper (confidence is None per spec)
    cw_asr = CrisperWhisperASR(device="cpu")
    cw_res = cw_asr.transcribe("dummy.wav", fallback_text="um test here")
    assert cw_res["source"] == "crisperwhisper"
    assert all(w["confidence"] is None for w in cw_res["words"])

    # Test Vosk wrapper
    v_asr = VoskASR()
    v_res = v_asr.transcribe("dummy.wav", fallback_text="test vosk here")
    assert v_res["source"] == "vosk"

    # Multi-model transcription
    multi = transcribe_multi("dummy.wav", segment_id="seg_42")
    assert "whisper" in multi
    assert "crisperwhisper" in multi
    assert "vosk" in multi
    assert multi["whisper"]["segment_id"] == "seg_42"
