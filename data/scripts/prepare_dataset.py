"""CLI: pull a language's audio+transcripts once, shared by every engine
folder (whisper/, gemini/, chirp/, wav2vec2/), into data/<language>/ as wav
files + a manifest CSV.

WHY THIS REPLACES EACH ENGINE'S OWN prepare_dataset.py:
Every engine folder used to download its own private copy of the same FLEURS
clips - deliberate while the test set was a fixed 20 samples (cheap to
duplicate, and it kept each folder's code fully independent - see e.g.
gemini/README.md's "Why a separate folder" section). That stopped being
cheap once the eval set grew past 500 clips - four copies of ~700 audio
clips is real, avoidable disk/bandwidth. This script is now the ONE place
that downloads; every engine's lib/config.py points its wav_dir/manifest_path
at data/<language>/ instead (see e.g. whisper/lib/config.py's DATA_ROOT).

Usage:
    # The large-scale eval set used for the four-engine comparison: FLEURS
    # sw_ke's test (487) + validation (211) splits combined, 698 clips.
    python data/scripts/prepare_dataset.py --language kiswahili --split eval

    # A single underlying HF split, uncapped or capped - e.g. for a future
    # fine-tuning run that wants `train`:
    python data/scripts/prepare_dataset.py --language kiswahili --split train --max-samples 80
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
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "kiswahili"],
        help="Which language's data to download (matches a data/<language>/ folder).",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Local split name to save under (e.g. 'eval', 'train', 'validation', 'test'). "
        "'eval' defaults --source-splits to test+validation if not given explicitly.",
    )
    parser.add_argument(
        "--source-splits",
        default=None,
        help="Comma-separated underlying Hugging Face split(s) to pull and concatenate "
        "(e.g. 'test,validation'). Defaults to a single split matching --split, except "
        "'eval', which defaults to 'test,validation'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the TOTAL number of clips across all --source-splits combined. "
        "Omit for no cap - pulls every available clip (this is what 'eval' uses).",
    )
    args = parser.parse_args()

    cfg = load_config(args.language)

    if args.source_splits:
        source_splits = [s.strip() for s in args.source_splits.split(",") if s.strip()]
    elif args.split == "eval":
        source_splits = ["test", "validation"]
    else:
        source_splits = [args.split]

    print(
        f"Downloading '{args.split}' (from {'+'.join(source_splits)}) of "
        f"{cfg.dataset_id}/{cfg.dataset_config} for {cfg.language_name}"
        f"{f', capped at {args.max_samples}' if args.max_samples else ' (uncapped)'}..."
    )

    df = download_split(cfg, args.split, source_splits, args.max_samples)

    print(f"Wrote {len(df)} clips to {cfg.wav_dir}")
    print(f"Manifest: {cfg.manifest_path(args.split)}")


if __name__ == "__main__":
    main()
