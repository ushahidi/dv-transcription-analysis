"""Config loading shared by every gemini/scripts/*.py entry point.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Same job as whisper/lib/config.py, deliberately re-implemented here rather than
imported, so this folder never depends on anything under whisper/ (see
gemini/README.md for why). Every script in this folder (download data, run a
baseline) needs to know things like which dataset to use, which language, which
Gemini model to call - those live in small YAML files (config.yaml), one shared
"defaults" file plus one small file per language listing only what's different.

This is lighter than whisper/lib/config.py on purpose: there's no local model to
put on a device, no fp16 flag, no training hyperparameters - Gemini is called over
the network with a model name, nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

# GEMINI_ROOT points at the "gemini/" folder itself (two levels up from this
# file, which lives at gemini/lib/config.py).
GEMINI_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = GEMINI_ROOT / "lib" / "default_config.yaml"

# The eval data (wav files + manifest) lives in one shared folder at the repo
# root, not a private gemini/<language>/data/ copy - see
# data/scripts/prepare_dataset.py for why, and gemini/README.md for what
# changed here.
SHARED_DATA_ROOT = GEMINI_ROOT.parent / "data"


def _deep_merge(base: dict, override: dict) -> dict:
    """Combine two settings dictionaries into one - see
    whisper/lib/config.py's `_deep_merge` for the full explanation. Identical
    logic, duplicated rather than imported (this folder is deliberately
    decoupled from whisper/lib)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class PipelineConfig:
    """A single object holding every setting a script needs, already resolved.

    After this object is built, scripts just read attributes off it, e.g.
    `cfg.gemini_model`, `cfg.wav_dir` - they never read YAML directly.
    """

    def __init__(self, language: str, raw: dict, language_dir: Path):
        self.language = language

        # --- Basic identity of this language ---
        # Only language_name is actually used here (in the transcription
        # prompt and the results JSON's "language" field) - dataset_id,
        # dataset_config, text_column, audio_column, and language_code used
        # to live here too, back when this folder downloaded its own copy of
        # the data; now that data/ owns downloading (see
        # data/lib/config.py's DatasetConfig, the equivalent for those
        # fields), keeping unused duplicates here would just be one more
        # place to forget to update.
        self.language_name: str = raw["language_name"]

        # --- Which Gemini model to call ---
        self.gemini_model: str = raw["gemini"]["model"]

        # --- Folder layout for this language ---
        # results/ is this folder's own output; data/ (wav files + manifest)
        # is shared with every other engine folder - see SHARED_DATA_ROOT above.
        self.language_dir = language_dir
        self.data_dir = SHARED_DATA_ROOT / language
        self.wav_dir = self.data_dir / "wav"
        self.results_dir = language_dir / "results"

    def manifest_path(self, split: str) -> Path:
        """Path to the CSV list of audio clips + transcripts for one split
        (e.g. "train", "validation", "test", or "eval")."""
        return self.data_dir / f"manifest_{split}.csv"


def load_config(language: str, config_path: Optional[str] = None) -> PipelineConfig:
    """The main entry point every script calls: `load_config("kiswahili")`, etc.

    Steps:
      1. Read the shared defaults file (lib/default_config.yaml).
      2. Read that language's own config.yaml (or a path passed in via
         --config).
      3. Merge the two together (the language file's values win).
      4. Wrap the result in a PipelineConfig object with paths already figured
         out.
    """
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)

    language_dir = GEMINI_ROOT / language
    override_path = Path(config_path) if config_path else language_dir / "config.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"No config found for language '{language}' at {override_path}")
    with open(override_path, "r", encoding="utf-8") as f:
        override_raw = yaml.safe_load(f)

    merged = _deep_merge(default_raw, override_raw)
    return PipelineConfig(language=language, raw=merged, language_dir=language_dir)
