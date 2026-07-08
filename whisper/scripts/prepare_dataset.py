"""CLI: pull a small subset of a language's configured dataset (google/fleurs by
default) and write it to whisper/<language>/data/ as wav files + a manifest CSV.

Usage:
    python whisper/scripts/prepare_dataset.py --language english --split test --max-samples 20
    python whisper/scripts/prepare_dataset.py --language kiswahili --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import load_config
from lib.dataset_utils import download_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["english", "kiswahili"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Defaults to data.max_samples[<split>] in the language's config.yaml.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to a config.yaml (defaults to whisper/<language>/config.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    max_samples = args.max_samples or cfg.max_samples.get(args.split, 20)

    print(
        f"Downloading {max_samples} '{args.split}' samples of "
        f"{cfg.dataset_id}/{cfg.dataset_config} for {cfg.language_name}..."
    )
    df = download_split(cfg, args.split, max_samples)
    print(f"Wrote {len(df)} clips to {cfg.wav_dir}")
    print(f"Manifest: {cfg.manifest_path(args.split)}")


if __name__ == "__main__":
    main()
