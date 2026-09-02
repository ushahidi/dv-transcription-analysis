"""Manifest I/O for chirp/scripts/run_baseline.py.

Identical to gemini/lib/dataset_utils.py - see that file's docstring. The
actual downloading now happens once, centrally, in
data/scripts/prepare_dataset.py.
"""

from __future__ import annotations

import pandas as pd

from .config import PipelineConfig


def load_manifest(cfg: PipelineConfig, split: str) -> pd.DataFrame:
    path = cfg.manifest_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for split '{split}' at {path}. Run "
            f"`python data/scripts/prepare_dataset.py --language {cfg.language} "
            f"--split {split}` first."
        )
    return pd.read_csv(path)
