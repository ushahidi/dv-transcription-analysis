"""Manifest I/O for gemini/scripts/run_baseline.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Just reads back a manifest CSV that was already downloaded - the actual
downloading now happens once, centrally, in data/scripts/prepare_dataset.py
(see that file's docstring, and gemini/README.md, for why this folder no
longer downloads its own copy of the data). `load_manifest`'s logic is
unchanged from before; only where the manifest/wav files actually live
changed (gemini/lib/config.py's `data_dir` now points at the shared
data/<language>/ folder).
"""

from __future__ import annotations

import pandas as pd

from .config import PipelineConfig


def load_manifest(cfg: PipelineConfig, split: str) -> pd.DataFrame:
    """Read back the manifest CSV for a split that was already downloaded."""
    path = cfg.manifest_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for split '{split}' at {path}. Run "
            f"`python data/scripts/prepare_dataset.py --language {cfg.language} "
            f"--split {split}` first."
        )
    return pd.read_csv(path)
