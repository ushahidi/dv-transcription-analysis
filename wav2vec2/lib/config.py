"""Config loading shared by every wav2vec2/scripts/*.py entry point.

Same job and shape as whisper/lib/config.py (see that file's docstring for
the full rationale) - duplicated here rather than imported so this folder
stays independent of whisper/gemini/chirp, per this repo's established
per-provider-folder convention. Lighter than whisper/lib/config.py in one way
(no `train:` section - this folder only runs baseline evaluation, not
fine-tuning) but otherwise identical: same deep-merge default+language-override
pattern, same device resolution.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import yaml

WAV2VEC2_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = WAV2VEC2_ROOT / "lib" / "default_config.yaml"

# The eval data (wav files + manifest) lives in one shared folder at the repo
# root, not a private wav2vec2/<language>/data/ copy - see
# data/scripts/prepare_dataset.py for why, and wav2vec2/README.md for what
# changed here.
SHARED_DATA_ROOT = WAV2VEC2_ROOT.parent / "data"


def _deep_merge(base: dict, override: dict) -> dict:
    """Combine two settings dictionaries into one - see
    whisper/lib/config.py's `_deep_merge` for the full explanation."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_device(requested: str) -> str:
    """"auto" -> cuda if available, else cpu - see whisper/lib/config.py's
    `_resolve_device` for the full explanation."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class PipelineConfig:
    """A single object holding every setting a script needs, already resolved."""

    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language

        # --- Basic identity of this language ---
        self.language_name: str = raw["language_name"]
        self.language_code: str = raw["language_code"]

        # --- Where the test data comes from (same source as whisper/,
        # gemini/, chirp/ - see wav2vec2/kiswahili/config.yaml for why this
        # reproduces the same clips) ---
        self.dataset_id: str = raw["dataset_id"]
        self.dataset_config: str = raw["dataset_config"]
        self.text_column: str = raw.get("text_column", "transcription")
        self.audio_column: str = raw.get("audio_column", "audio")

        # --- Which wav2vec2/CTC checkpoint, and where it runs ---
        self.model_name: str = raw["model"]["name"]
        self.device: str = _resolve_device(raw["model"]["device"])
        self.fp16: bool = bool(raw["model"].get("fp16", False)) and self.device == "cuda"

        # --- Adapter-based multilingual checkpoints (e.g. Meta's MMS) ---
        # A single-language checkpoint like thinkKenya's is ready to use as
        # loaded; an adapter-based one (model.type: mms) needs a target
        # language *adapter* selected before its first forward pass - see
        # wav2vec2/lib/model_utils.py's load_model/load_processor. lang_code
        # is deliberately separate from language_code above: MMS uses its own
        # ISO 639-3-ish codes (e.g. "swh" for Swahili), not necessarily the
        # same convention this project already uses for folder/display names.
        self.model_type: str = raw["model"].get("type", "ctc")
        self.mms_lang_code: Optional[str] = raw["model"].get("lang_code")

        # --- Audio + dataset-size settings ---
        self.sample_rate: int = raw["data"]["sample_rate"]
        self.max_samples: dict = raw["data"]["max_samples"]

        # --- Folder layout for this language ---
        # results/ is this folder's own output; data/ (wav files + manifest)
        # is shared with every other engine folder - see SHARED_DATA_ROOT above.
        self.language_dir = language_dir
        self.data_dir = SHARED_DATA_ROOT / language
        self.wav_dir = self.data_dir / "wav"
        self.results_dir = language_dir / "results"

    def manifest_path(self, split: str) -> Path:
        return self.data_dir / f"manifest_{split}.csv"

    def dataset_card_path(self) -> Path:
        return self.data_dir / "dataset_card.json"


def load_config(language: str, config_path: Optional[str] = None) -> PipelineConfig:
    """The main entry point every script calls: `load_config("kiswahili")`, etc."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)

    language_dir = WAV2VEC2_ROOT / language
    override_path = Path(config_path) if config_path else language_dir / "config.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"No config found for language '{language}' at {override_path}")
    with open(override_path, "r", encoding="utf-8") as f:
        override_raw = yaml.safe_load(f)

    merged = _deep_merge(default_raw, override_raw)
    return PipelineConfig(language=language, raw=merged, language_dir=language_dir)
