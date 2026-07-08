"""Dataset acquisition and manifest I/O shared across whisper/scripts/*.py."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Iterator

import librosa
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

from .config import PipelineConfig


def _stream_split(cfg: PipelineConfig, split: str, max_samples: int) -> Iterator[dict]:
    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=split, streaming=True)
    # decode=False keeps this to raw bytes instead of triggering `datasets`' default
    # audio decoder (torchcodec, which needs a matching FFmpeg install) - we decode
    # ourselves via soundfile below, which has no external dependency.
    ds = ds.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sample_rate, decode=False))
    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        yield example


def download_split(cfg: PipelineConfig, split: str, max_samples: int) -> pd.DataFrame:
    """Pull up to `max_samples` examples of `split` from the configured dataset, write
    each clip as a wav file under cfg.wav_dir, and write/return the manifest."""
    cfg.wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, example in enumerate(_stream_split(cfg, split, max_samples)):
        audio_field = example[cfg.audio_column]
        array, sr = sf.read(io.BytesIO(audio_field["bytes"]))
        if sr != cfg.sample_rate:
            array = librosa.resample(array, orig_sr=sr, target_sr=cfg.sample_rate)
            sr = cfg.sample_rate

        file_name = f"{split}_{i:04d}.wav"
        sf.write(cfg.wav_dir / file_name, array, sr)
        duration_sec = len(array) / sr
        rows.append(
            {
                "file_name": file_name,
                "transcript": example[cfg.text_column],
                "duration_sec": round(duration_sec, 3),
                "split": split,
                "source_dataset": cfg.dataset_id,
                "source_id": example.get("id", i),
            }
        )

    df = pd.DataFrame(rows)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.manifest_path(split), index=False)
    _update_dataset_card(cfg, split, len(df))
    return df


def _update_dataset_card(cfg: PipelineConfig, split: str, num_samples: int) -> None:
    path = cfg.dataset_card_path()
    card = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    card["dataset_id"] = cfg.dataset_id
    card["dataset_config"] = cfg.dataset_config
    card["language"] = cfg.language_name
    card["license"] = "CC-BY-4.0"
    card.setdefault("splits", {})
    card["splits"][split] = {
        "num_samples": num_samples,
        "sample_rate": cfg.sample_rate,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")


def load_manifest(cfg: PipelineConfig, split: str) -> pd.DataFrame:
    path = cfg.manifest_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for split '{split}' at {path}. Run "
            f"`python whisper/scripts/prepare_dataset.py --language {cfg.language} "
            f"--split {split}` first."
        )
    return pd.read_csv(path)


def manifest_to_dataset(cfg: PipelineConfig, split: str) -> Dataset:
    """Load the manifest for `split` as a plain HF Dataset with `audio_path` and
    `transcript` columns. Deliberately does NOT use `datasets`' `Audio` feature to
    decode - that requires torchcodec/FFmpeg. finetune.py reads each wav directly via
    soundfile in its own preprocessing step instead, same as prepare_dataset.py."""
    df = load_manifest(cfg, split).copy()
    df["audio_path"] = df["file_name"].apply(lambda name: str(cfg.wav_dir / name))
    return Dataset.from_pandas(df[["audio_path", "transcript"]], preserve_index=False)
