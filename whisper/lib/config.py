"""Config loading shared by every whisper/scripts/*.py entry point.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Every script in this project (download data, test accuracy, fine-tune, export)
needs to know things like: which dataset to use, which language, how big a model,
how many training steps, etc. Rather than hard-coding those choices into the
scripts themselves, they live in small YAML text files (config.yaml) - one shared
"defaults" file, plus one small file per language that only lists what's
different for that language.

This file is the piece of code that reads those YAML files, combines them, and
hands back a single easy-to-use object with everything already figured out
(e.g. "should this run on CPU or GPU?" is decided here, once, instead of in
every script).

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

# WHISPER_ROOT points at the "whisper/" folder itself (two levels up from this
# file, which lives at whisper/lib/config.py). Everything else in this file
# builds paths relative to that, so the code doesn't care whether it's run from
# Windows, the project root, or anywhere else.
WHISPER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = WHISPER_ROOT / "lib" / "default_config.yaml"

# The eval/test data (wav files + manifest) now lives in one shared folder
# at the repo root, not a private whisper/<language>/data/ copy - see
# data/scripts/prepare_dataset.py for why (four engine folders each keeping
# their own copy of the same 500+-clip FLEURS set stopped being cheap).
# whisper/'s own prepare_dataset.py/finetune.py (train/validation, for actual
# fine-tuning - not something the other engine folders do) still write here
# too, so there is still only ever one copy on disk.
SHARED_DATA_ROOT = WHISPER_ROOT.parent / "data"


def _deep_merge(base: dict, override: dict) -> dict:
    """Combine two settings dictionaries into one.

    Think of `base` as the shared defaults (default_config.yaml) and `override`
    as a specific language's tweaks (e.g. english/config.yaml). Anything the
    language file mentions wins; anything it doesn't mention falls back to the
    shared default. This works even for nested settings (e.g. the "train:"
    section can override just `max_steps` without having to repeat every other
    training setting).
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Both sides have a sub-section with this name (e.g. "train:") -
            # merge inside it instead of replacing the whole section.
            merged[key] = _deep_merge(merged[key], value)
        else:
            # A plain value (or a brand new key) - the override simply wins.
            merged[key] = value
    return merged


def _resolve_device(requested: str) -> str:
    """Turn the config's `device: auto` setting into an actual device name.

    "auto" means: use a graphics card (GPU/"cuda") if one is available and set
    up correctly, otherwise fall back to the regular processor ("cpu"). This is
    what lets the exact same config file be used on this CPU-only laptop and,
    later, on a machine with a GPU (e.g. Google Colab) - nothing needs to change
    by hand.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class PipelineConfig:
    """A single object holding every setting a script needs, already resolved.

    After this object is built, none of the scripts need to read YAML files or
    worry about "CPU vs GPU" logic themselves - they just read attributes off
    this object, e.g. `cfg.model_name`, `cfg.device`, `cfg.data_dir`.
    """

    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language

        # --- Basic identity of this language ---
        # `language_name` must match how Whisper's own tokenizer refers to the
        # language (e.g. "swahili", not the short code "sw") - see
        # whisper/kiswahili/config.yaml for a worked example of the difference.
        self.language_name: str = raw["language_name"]
        self.language_code: str = raw["language_code"]

        # --- Where the training/testing data comes from ---
        # dataset_id/dataset_config point at a specific dataset on Hugging Face
        # (e.g. "google/fleurs", config "sw_ke" for Kiswahili). text_column and
        # audio_column tell us which columns of that dataset hold the transcript
        # text and the audio clip, in case a different dataset uses different
        # column names later.
        self.dataset_id: str = raw["dataset_id"]
        self.dataset_config: str = raw["dataset_config"]
        self.text_column: str = raw.get("text_column", "transcription")
        self.audio_column: str = raw.get("audio_column", "audio")

        # --- Which Whisper model, and where it runs ---
        self.model_name: str = raw["model"]["name"]
        self.device: str = _resolve_device(raw["model"]["device"])

        # fp16 ("half precision") speeds up training/inference on a GPU, but it
        # is not supported when running on a plain CPU. Rather than trust every
        # config file to remember that, we force it off here automatically
        # whenever we're not actually running on a GPU ("cuda") - so a config
        # left with `fp16: true` can never accidentally break a CPU run.
        self.fp16: bool = bool(raw["model"]["fp16"]) and self.device == "cuda"

        # --- Audio + dataset-size settings ---
        self.sample_rate: int = raw["data"]["sample_rate"]
        self.max_samples: dict = raw["data"]["max_samples"]

        # --- Training hyperparameters (learning rate, batch size, etc.) ---
        # Kept as a plain dictionary (rather than named attributes) because the
        # exact set of training settings can grow over time without this file
        # needing to change - see whisper/scripts/finetune.py for how it's used.
        self.train: dict = raw["train"]

        # --- Folder layout for this language ---
        # e.g. for English: whisper/english/ (results/, models/ - this
        # folder's own outputs) but data/english/ (wav files + manifest -
        # shared with every other engine folder, see SHARED_DATA_ROOT above).
        self.language_dir = language_dir
        self.data_dir = SHARED_DATA_ROOT / language
        self.wav_dir = self.data_dir / "wav"
        self.results_dir = language_dir / "results"
        self.models_dir = language_dir / "models"

    def manifest_path(self, split: str) -> Path:
        """Path to the CSV list of audio clips + transcripts for one split
        (e.g. "train", "validation", or "test")."""
        return self.data_dir / f"manifest_{split}.csv"

    def dataset_card_path(self) -> Path:
        """Path to a small JSON file recording exactly which dataset/version was
        downloaded and when - kept for reproducibility (see dataset_utils.py)."""
        return self.data_dir / "dataset_card.json"

    def output_dir(self, model_name: str) -> Path:
        """Folder where a fine-tuned model for this language + base model gets
        saved, e.g. whisper/english/models/whisper-tiny-finetuned/."""
        safe_name = model_name.split("/")[-1]
        suffix = self.train.get("output_dir_suffix", "finetuned")
        return self.models_dir / f"{safe_name}-{suffix}"


def load_config(language: str, config_path: Optional[str] = None) -> PipelineConfig:
    """The main entry point every script calls: `load_config("english")`,
    `load_config("kiswahili")`, etc.

    Steps:
      1. Read the shared defaults file (lib/default_config.yaml).
      2. Read that language's own config.yaml (or a path passed in via
         --config, e.g. to point at the GPU-scale defaults directly on Colab).
      3. Merge the two together (the language file's values win).
      4. Wrap the result in a PipelineConfig object with paths and the
         CPU/GPU device already figured out.
    """
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
