"""Config loading shared by every whisper/scripts/*.py entry point.

Each language directory (whisper/english/, whisper/kiswahili/, ...) holds a
config.yaml with only what differs for that language (dataset, language
name/code, and any CPU-smoke-test hyperparameter overrides). It is deep-merged
on top of lib/default_config.yaml, which holds the GPU-scale defaults. This is
the mechanism that lets the same scripts run on a CPU box now and a GPU box
later: only the config values change, not the code.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import yaml

WHISPER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = WHISPER_ROOT / "lib" / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class PipelineConfig:
    """Merged default + per-language config, with paths and device resolved."""

    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language
        self.language_name: str = raw["language_name"]
        self.language_code: str = raw["language_code"]
        self.dataset_id: str = raw["dataset_id"]
        self.dataset_config: str = raw["dataset_config"]
        self.text_column: str = raw.get("text_column", "transcription")
        self.audio_column: str = raw.get("audio_column", "audio")

        self.model_name: str = raw["model"]["name"]
        self.device: str = _resolve_device(raw["model"]["device"])
        # fp16 training/inference isn't supported on CPU - force it off regardless
        # of what any config says whenever we didn't resolve to a CUDA device.
        self.fp16: bool = bool(raw["model"]["fp16"]) and self.device == "cuda"

        self.sample_rate: int = raw["data"]["sample_rate"]
        self.max_samples: dict = raw["data"]["max_samples"]

        self.train: dict = raw["train"]

        self.language_dir = language_dir
        self.data_dir = language_dir / "data"
        self.wav_dir = self.data_dir / "wav"
        self.results_dir = language_dir / "results"
        self.models_dir = language_dir / "models"

    def manifest_path(self, split: str) -> Path:
        return self.data_dir / f"manifest_{split}.csv"

    def dataset_card_path(self) -> Path:
        return self.data_dir / "dataset_card.json"

    def output_dir(self, model_name: str) -> Path:
        safe_name = model_name.split("/")[-1]
        suffix = self.train.get("output_dir_suffix", "finetuned")
        return self.models_dir / f"{safe_name}-{suffix}"


def load_config(language: str, config_path: Optional[str] = None) -> PipelineConfig:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)

    language_dir = WHISPER_ROOT / language
    override_path = Path(config_path) if config_path else language_dir / "config.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"No config found for language '{language}' at {override_path}")
    with open(override_path, "r", encoding="utf-8") as f:
        override_raw = yaml.safe_load(f)

    merged = _deep_merge(default_raw, override_raw)
    return PipelineConfig(language=language, raw=merged, language_dir=language_dir)
