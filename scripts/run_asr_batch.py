"""Batch ASR processing script for EdAcc segments (FS-016). Member B (WP2).

Reads segments from data/gold_dataset.csv (or a specified CSV/split),
runs Whisper (or faster-whisper / fallback simulation for testing),
and saves individual JSON files conforming to docs/data_contracts.md:
    data/processed/feedback/whisper/<segment_id>.json

Usage:
    python -m scripts.run_asr_batch --limit 50
    python -m scripts.run_asr_batch --model whisper --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.asr.transcribe import transcribe_segment
from src.feedback import io

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "feedback"


def run_batch_transcription(
    segments_csv: Path | str = DATA_DIR / "gold_dataset.csv",
    model: str = "whisper",
    limit: int | None = None,
    output_dir: Path | None = None,
) -> int:
    """Batch transcribe segments and save each to data/processed/feedback/<model>/<segment_id>.json."""
    csv_path = Path(segments_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Segments CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if limit is not None and limit > 0:
        df = df.head(limit)

    target_dir = (output_dir or PROCESSED_DIR) / model
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running batch ASR ({model}) for {len(df)} segments -> {target_dir}")
    saved_count = 0

    for _, row in df.iterrows():
        seg_id = str(row["segment_id"])
        ref_text = str(row.get("text", row.get("human_transcript", "")))
        audio_path = row.get("audio_path", f"data/interim/audio/{seg_id}.wav")

        # Run transcription with fallback text if audio file is not present locally
        record = transcribe_segment(
            audio_path=audio_path,
            segment_id=seg_id,
            model=model,
            fallback_text=ref_text,
        )

        out_file = target_dir / f"{seg_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        saved_count += 1

    print(f"Successfully saved {saved_count} {model} outputs to {target_dir}")
    return saved_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe EdAcc segments.")
    parser.add_argument("--csv", type=str, default="data/gold_dataset.csv", help="Path to segments CSV.")
    parser.add_argument("--model", type=str, default="whisper", choices=["whisper", "crisperwhisper", "vosk"], help="ASR model to run.")
    parser.add_argument("--limit", type=int, default=25, help="Number of segments to process (default: 25 for quick run, 0 for all).")
    args = parser.parse_args()

    lim = None if args.limit <= 0 else args.limit
    run_batch_transcription(segments_csv=args.csv, model=args.model, limit=lim)
