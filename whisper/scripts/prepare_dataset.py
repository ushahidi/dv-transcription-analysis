"""CLI: pull a small subset of a language's configured dataset (google/fleurs by
default) and write it to data/<language>/ (shared with every other engine
folder - see data/README.md) as wav files + a manifest CSV.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
This is step 1 of fine-tuning specifically: before whisper/scripts/finetune.py
can train on anything, it needs a `train`/`validation` split downloaded. This
script is scoped to those two splits ONLY - `test`/`eval` are deliberately not
options here anymore. Both of those live in data/<language>/ too, but they're
written exclusively by data/scripts/prepare_dataset.py, so there's only ever
one writer for any given split (see data/README.md's "Prefer one writer"
note) - this script and that one never fight over the same file.

For a baseline WER comparison against the other engine folders, use:
    python data/scripts/prepare_dataset.py --language <language> --split eval

Usage:
    python whisper/scripts/prepare_dataset.py --language english --split train --max-samples 80
    python whisper/scripts/prepare_dataset.py --language kiswahili --split validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives in whisper/scripts/, but it needs to import code from
# whisper/lib/ (a sibling folder). This line adds the whisper/ folder itself to
# Python's search path so `import lib.config` etc. below can find it, no
# matter which directory this script is actually run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import load_config
from lib.dataset_utils import download_split


def main() -> None:
    # --- Read the command-line options the user passed in ---
    # e.g. `--language english --split test --max-samples 20`
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "kiswahili"],
        help="Which language's data to download (matches a whisper/<language>/ folder).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Which fine-tuning split to pull: 'train' or 'validation'. For 'test' or "
        "'eval' (the baseline WER comparison set), use "
        "data/scripts/prepare_dataset.py instead - see this file's module docstring.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="How many clips to download. Defaults to data.max_samples[<split>] in the "
        "language's config.yaml if not given here.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to a config.yaml (defaults to whisper/<language>/config.yaml).",
    )
    args = parser.parse_args()

    # Load this language's settings (which dataset, sample rate, etc.) - see
    # whisper/lib/config.py for exactly how this works.
    cfg = load_config(args.language, args.config)

    # If the user didn't specify --max-samples on the command line, fall back
    # to whatever number is configured for this split in config.yaml (defaulting
    # to 20 if even that isn't set, just to be safe).
    max_samples = args.max_samples or cfg.max_samples.get(args.split, 20)

    print(
        f"Downloading {max_samples} '{args.split}' samples of "
        f"{cfg.dataset_id}/{cfg.dataset_config} for {cfg.language_name}..."
    )

    # This is the actual download + save-to-disk work - see
    # whisper/lib/dataset_utils.py's download_split function for the details.
    df = download_split(cfg, args.split, max_samples)

    print(f"Wrote {len(df)} clips to {cfg.wav_dir}")
    print(f"Manifest: {cfg.manifest_path(args.split)}")


if __name__ == "__main__":
    # This "if" only runs main() when the file is executed directly as a
    # script (e.g. `python prepare_dataset.py`), not if it were ever imported
    # from another Python file instead.
    main()
