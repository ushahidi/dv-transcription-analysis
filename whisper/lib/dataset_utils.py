"""Dataset acquisition and manifest I/O shared across whisper/scripts/*.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
This is where we download a handful of real speech recordings (with their
correct written transcripts) from a public dataset on the internet, and save
them to data/<language>/ (shared with every other engine folder - see
data/README.md) in a simple format: one .wav sound file per clip, plus a
"manifest" - a spreadsheet-like CSV file listing every clip's filename, its
correct transcript, how long it is, etc.

`download_split` here is only ever invoked for `train`/`validation` (via
whisper/scripts/prepare_dataset.py, for fine-tuning) - `test`/`eval` are
downloaded exclusively by data/scripts/prepare_dataset.py instead, so there's
only ever one writer for any given split. run_baseline.py never downloads
anything itself; it just reads whatever manifest + wav files already exist in
data/<language>/, regardless of which script wrote them.

A NOTE ON WHY WE DECODE AUDIO OURSELVES:
The Hugging Face `datasets` library can normally decode audio automatically,
but its newer versions do that using a tool called "torchcodec", which in turn
needs "FFmpeg" installed on the computer. Rather than requiring every user of
this project to install FFmpeg, we deliberately tell `datasets` NOT to decode
the audio itself (`decode=False`) and instead decode the raw audio bytes
ourselves using a much simpler, dependency-free library called `soundfile`.
"""

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
    """Connect to the online dataset and hand back examples one at a time.

    `split` means which portion of the dataset we want - "train" (for
    fine-tuning), "validation" (for checking progress during training), or
    "test" (for the final accuracy report). `streaming=True` means we don't
    download the *entire* dataset up front - we only pull as many examples as
    we actually ask for (`max_samples`), which is much faster for small tests.
    """
    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=split, streaming=True)
    # decode=False keeps this to raw, un-decoded audio bytes instead of
    # triggering `datasets`' default audio decoder (torchcodec, which needs a
    # matching FFmpeg install on the computer) - we decode the bytes ourselves
    # a few lines down using `soundfile`, which needs nothing extra installed.
    ds = ds.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sample_rate, decode=False))
    for i, example in enumerate(ds):
        if i >= max_samples:
            # We only wanted `max_samples` examples - stop asking for more.
            break
        yield example


def download_split(cfg: PipelineConfig, split: str, max_samples: int) -> pd.DataFrame:
    """Download up to `max_samples` clips for `split`, save each as a .wav file,
    and write a manifest CSV describing them all. Returns that same manifest as
    a table (pandas DataFrame) for convenience.

    This is the function whisper/scripts/prepare_dataset.py calls directly.
    """
    # Make sure the destination folder exists before we try writing files into it.
    cfg.wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # Will become one row per audio clip in the final manifest CSV.
    for i, example in enumerate(_stream_split(cfg, split, max_samples)):
        # `example` is one record from the dataset - a dictionary containing
        # the transcript text and the raw (not-yet-decoded) audio bytes.
        audio_field = example[cfg.audio_column]

        # Turn the raw audio bytes into an actual array of numbers (the sound
        # wave) plus its sample rate (how many numbers represent one second of
        # audio), using `soundfile` - no FFmpeg needed.
        array, sr = sf.read(io.BytesIO(audio_field["bytes"]))

        # If the clip's native sample rate doesn't match what we want (16kHz,
        # the rate Whisper expects), resample it - i.e. mathematically stretch
        # or compress the audio so it represents the same sound at the target
        # rate. In practice FLEURS (our default dataset) is already 16kHz, so
        # this rarely triggers, but it keeps things correct if that changes.
        if sr != cfg.sample_rate:
            array = librosa.resample(array, orig_sr=sr, target_sr=cfg.sample_rate)
            sr = cfg.sample_rate

        # Save this one clip as its own .wav file, named e.g. "test_0003.wav".
        file_name = f"{split}_{i:04d}.wav"
        sf.write(cfg.wav_dir / file_name, array, sr)
        duration_sec = len(array) / sr

        # Record everything we'll want to know about this clip later.
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

    # Turn the list of clip records into a proper table and save it as a CSV
    # (a plain-text spreadsheet format anyone can open in Excel/Notepad).
    df = pd.DataFrame(rows)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.manifest_path(split), index=False)

    # Also update a small "receipt" file recording exactly what was downloaded
    # and when, for future reference (see _update_dataset_card below).
    _update_dataset_card(cfg, split, len(df))
    return df


def _update_dataset_card(cfg: PipelineConfig, split: str, num_samples: int) -> None:
    """Keep a small JSON "receipt" file (dataset_card.json) recording which
    dataset/version we pulled data from, and when, for each split we've
    downloaded so far. This makes it possible to answer "where did this data
    come from, and is it still current?" months later without guessing.
    """
    path = cfg.dataset_card_path()
    # If a card already exists (e.g. we already downloaded "train" before and
    # are now adding "test"), load it so we can add to it instead of erasing it.
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
    """Read back the manifest CSV for a split that was already downloaded.

    Used by run_baseline.py (to know which clips to test) and, indirectly, by
    manifest_to_dataset below (to prepare data for fine-tuning). Raises a clear
    error telling the user which command to run first if the data isn't there yet.
    """
    path = cfg.manifest_path(split)
    if not path.exists():
        # 'test'/'eval' are downloaded by the shared data/scripts/prepare_dataset.py
        # (not this folder's own prepare_dataset.py, which is scoped to
        # train/validation for fine-tuning - see whisper/lib/dataset_utils.py's
        # module docstring) - point at whichever one actually owns this split.
        if split in ("test", "eval"):
            command = f"python data/scripts/prepare_dataset.py --language {cfg.language} --split {split}"
        else:
            command = f"python whisper/scripts/prepare_dataset.py --language {cfg.language} --split {split}"
        raise FileNotFoundError(f"No manifest for split '{split}' at {path}. Run `{command}` first.")
    return pd.read_csv(path)


def manifest_to_dataset(cfg: PipelineConfig, split: str) -> Dataset:
    """Load the manifest for `split` as a Hugging Face Dataset object (the
    format the fine-tuning code expects), with columns `audio_path` (where the
    .wav file lives on disk) and `transcript` (the correct text).

    Deliberately does NOT use `datasets`' built-in `Audio` feature to decode the
    sound files - as explained at the top of this file, that would require
    FFmpeg to be installed. Instead, whisper/scripts/finetune.py opens each .wav
    file itself using `soundfile` when it actually needs the audio.
    """
    df = load_manifest(cfg, split).copy()
    df["audio_path"] = df["file_name"].apply(lambda name: str(cfg.wav_dir / name))
    return Dataset.from_pandas(df[["audio_path", "transcript"]], preserve_index=False)
