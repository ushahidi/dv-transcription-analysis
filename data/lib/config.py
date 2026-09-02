"""Config loading for data/scripts/prepare_dataset.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Where whisper/, gemini/, chirp/, and wav2vec2/ each used to download their
own private copy of the same FLEURS clips (deliberately, while the test set
was a fixed 20 samples - see each folder's README for that rationale), this
folder is the one place a language's audio actually gets downloaded to, once
the test set grew past 500 clips and four separate copies stopped being
cheap. Every engine folder's own lib/config.py now points its wav_dir/
manifest_path at data/<language>/ instead of downloading a copy itself - see
e.g. whisper/lib/config.py's DATA_ROOT.

Kept deliberately small: this is only ever about "what to download and where
to put it" - no model settings, no train hyperparameters. Those still live
in each engine's own config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DATA_ROOT = Path(__file__).resolve().parent.parent


class DatasetConfig:
    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language
        self.language_name: str = raw["language_name"]
        self.language_code: str = raw["language_code"]
        self.dataset_id: str = raw["dataset_id"]
        self.dataset_config: str = raw["dataset_config"]
        self.text_column: str = raw.get("text_column", "transcription")
        self.audio_column: str = raw.get("audio_column", "audio")
        self.sample_rate: int = raw.get("sample_rate", 16000)

        self.language_dir = language_dir
        self.wav_dir = language_dir / "wav"

    def manifest_path(self, split: str) -> Path:
        return self.language_dir / f"manifest_{split}.csv"

    def dataset_card_path(self) -> Path:
        return self.language_dir / "dataset_card.json"


def load_config(language: str) -> DatasetConfig:
    language_dir = DATA_ROOT / language
    config_path = language_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for language '{language}' at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DatasetConfig(language=language, raw=raw, language_dir=language_dir)
