"""Tests for data/lib/dataset_utils.py's download_split.

Mocks `datasets.load_dataset` (no real network access) with small fake
streaming datasets, so these run offline. Covers:
  - combining multiple source splits (test+validation, mirroring the real
    'eval' split) into one local split, with source_split correctly tagged
    per row
  - --max-samples caps the TOTAL across every source split combined,
    applied greedily in the order source_splits is given (all of the first
    split before touching the second)
  - the manifest CSV actually gets the right columns/values written to disk
"""

import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import data.lib.dataset_utils as dataset_utils
from data.lib.config import DatasetConfig


def _wav_bytes(duration_sec: float = 0.5, sr: int = 16000) -> bytes:
    audio = np.zeros(int(duration_sec * sr), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class _FakeStreamingDataset:
    """Minimal stand-in for a HF `datasets` streaming split - just enough
    surface for _stream_split to call .cast_column(...) then iterate."""

    def __init__(self, examples):
        self._examples = examples

    def cast_column(self, column, feature):
        return self  # examples already carry ready-to-decode raw wav bytes

    def __iter__(self):
        return iter(self._examples)


def _make_examples(n: int, prefix: str):
    return [
        {"transcription": f"{prefix} transcript {i}", "audio": {"bytes": _wav_bytes()}, "id": f"{prefix}-{i}"}
        for i in range(n)
    ]


@pytest.fixture
def cfg(tmp_path):
    raw = {
        "language_name": "swahili",
        "language_code": "sw",
        "dataset_id": "google/fleurs",
        "dataset_config": "sw_ke",
        "text_column": "transcription",
        "audio_column": "audio",
        "sample_rate": 16000,
    }
    return DatasetConfig(language="kiswahili", raw=raw, language_dir=tmp_path / "kiswahili")


@pytest.fixture
def fake_load_dataset(monkeypatch):
    """Route load_dataset(..., split=X, ...) to a small fake dataset keyed
    by split name - "test" has 3 examples, "validation" has 2, matching the
    real test(487)+validation(211) shape at a testable scale."""
    datasets_by_split = {
        "test": _FakeStreamingDataset(_make_examples(3, "test")),
        "validation": _FakeStreamingDataset(_make_examples(2, "validation")),
    }

    def _fake_load_dataset(dataset_id, dataset_config, split, streaming):
        return datasets_by_split[split]

    monkeypatch.setattr(dataset_utils, "load_dataset", _fake_load_dataset)
    return datasets_by_split


def test_combines_source_splits_with_correct_source_split_tags(cfg, fake_load_dataset):
    df = dataset_utils.download_split(cfg, local_split="eval", source_splits=["test", "validation"])

    assert len(df) == 5  # 3 from test + 2 from validation, uncapped
    assert list(df["source_split"]) == ["test", "test", "test", "validation", "validation"]
    assert list(df["file_name"]) == [f"eval_{i:04d}.wav" for i in range(5)]
    assert all(df["split"] == "eval")


def test_max_samples_caps_greedily_across_source_splits(cfg, fake_load_dataset):
    # 3 from "test" (all of it) + only 1 of "validation"'s 2, stopping the
    # moment the total cap is hit - not split proportionally, not rounded.
    df = dataset_utils.download_split(cfg, local_split="eval", source_splits=["test", "validation"], max_samples=4)

    assert len(df) == 4
    assert list(df["source_split"]) == ["test", "test", "test", "validation"]


def test_manifest_csv_is_written_with_expected_columns(cfg, fake_load_dataset):
    dataset_utils.download_split(cfg, local_split="eval", source_splits=["test", "validation"])

    manifest_path = cfg.manifest_path("eval")
    assert manifest_path.exists()
    on_disk = pd.read_csv(manifest_path)
    assert list(on_disk.columns) == [
        "file_name",
        "transcript",
        "duration_sec",
        "split",
        "source_dataset",
        "source_split",
        "source_id",
    ]
    assert len(on_disk) == 5


def test_load_manifest_reads_back_what_download_split_wrote(cfg, fake_load_dataset):
    dataset_utils.download_split(cfg, local_split="eval", source_splits=["test", "validation"])
    df = dataset_utils.load_manifest(cfg, "eval")
    assert len(df) == 5


def test_load_manifest_missing_split_raises_with_helpful_message(cfg):
    with pytest.raises(FileNotFoundError, match="data/scripts/prepare_dataset.py"):
        dataset_utils.load_manifest(cfg, "eval")
