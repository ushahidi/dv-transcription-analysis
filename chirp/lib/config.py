"""Config loading shared by every chirp/scripts/*.py entry point.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Same job and shape as gemini/lib/config.py (see that file's docstring for the
full rationale) - duplicated here rather than imported so chirp/ stays
independent of gemini/ and whisper/ alike. The only real difference is the
settings this engine needs: a Cloud Speech-to-Text model id + BCP-47 language
code(s) + a GCP region, instead of a Gemini model id.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import yaml

CHIRP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = CHIRP_ROOT / "lib" / "default_config.yaml"

# The eval data (wav files + manifest) lives in one shared folder at the repo
# root, not a private chirp/<language>/data/ copy - see
# data/scripts/prepare_dataset.py for why, and chirp/README.md for what
# changed here.
SHARED_DATA_ROOT = CHIRP_ROOT.parent / "data"


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


class PipelineConfig:
    """A single object holding every setting a script needs, already resolved."""

    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language

        # --- Basic identity of this language ---
        # Only language_name is actually used here (the results JSON's
        # "language" field) - dataset_id/dataset_config/text_column/
        # audio_column/language_code used to live here too, back when this
        # folder downloaded its own copy of the data; now that data/ owns
        # downloading (see data/lib/config.py's DatasetConfig), keeping
        # unused duplicates here would just be one more place to forget to
        # update.
        self.language_name: str = raw["language_name"]

        # --- Cloud Speech-to-Text settings ---
        self.stt_model: str = raw["stt"]["model"]
        self.stt_language_codes: List[str] = raw["stt"]["language_codes"]
        self.gcp_location: str = raw["gcp"]["location"]

        # --- Folder layout for this language ---
        # results/ is this folder's own output; data/ (wav files + manifest)
        # is shared with every other engine folder - see SHARED_DATA_ROOT above.
        self.language_dir = language_dir
        self.data_dir = SHARED_DATA_ROOT / language
        self.wav_dir = self.data_dir / "wav"
        self.results_dir = language_dir / "results"

    def manifest_path(self, split: str) -> Path:
        return self.data_dir / f"manifest_{split}.csv"


def load_config(language: str, config_path: Optional[str] = None) -> PipelineConfig:
    """The main entry point every script calls: `load_config("kiswahili")`, etc."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)

    language_dir = CHIRP_ROOT / language
    override_path = Path(config_path) if config_path else language_dir / "config.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"No config found for language '{language}' at {override_path}")
    with open(override_path, "r", encoding="utf-8") as f:
        override_raw = yaml.safe_load(f)

    merged = _deep_merge(default_raw, override_raw)
    return PipelineConfig(language=language, raw=merged, language_dir=language_dir)
