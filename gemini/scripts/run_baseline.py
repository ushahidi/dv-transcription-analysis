"""CLI: run a Gemini model over a language's local test set and report WER plus
a per-sample confidence proxy, in the same shape as whisper/scripts/run_baseline.py.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
The "how accurate is Gemini?" script - Gemini's counterpart to
whisper/scripts/run_baseline.py. It takes every audio clip in a language's test
set (downloaded earlier by prepare_dataset.py), asks Gemini to transcribe each
one, and compares that guess against the known-correct transcript, reporting:

  - WER (Word Error Rate) per clip and overall.
  - A best-effort confidence proxy per clip, where Gemini's response exposes one
    (see gemini/lib/gemini_client.py - unlike Whisper, this isn't guaranteed).

Results are saved to gemini/<language>/results/ as JSON + CSV, in the exact same
shape whisper/scripts/run_baseline.py produces, so the two are directly
diffable/comparable.

Usage:
    python gemini/scripts/run_baseline.py --language kiswahili --split test
    python gemini/scripts/run_baseline.py --language kiswahili --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib.config import load_config
from lib.dataset_utils import load_manifest
from lib.gemini_client import make_client, transcribe
from lib.metrics import compute_wer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "kiswahili"],
        help="Which language's test set to evaluate against.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model id (e.g. gemini-flash-latest). Defaults to gemini.model in "
        "config.yaml.",
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=["train", "validation", "test", "eval"],
        help="Which downloaded split to evaluate against. 'eval' (the default) is the "
        "shared, larger FLEURS test+validation set in ../data/ - see data/README.md.",
    )
    parser.add_argument("--config", default=None, help="Override path to a config.yaml.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N rows of the split (e.g. --limit 100 out of a "
        "698-clip 'eval' split) - for a quicker/cheaper run without downloading a "
        "separate smaller split.",
    )
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_name = args.model or cfg.gemini_model

    # The manifest is the spreadsheet listing every clip in this split and its
    # correct transcript - produced earlier by prepare_dataset.py.
    manifest = load_manifest(cfg, args.split)
    if args.limit:
        manifest = manifest.head(args.limit)

    client = make_client()

    per_sample = []
    for i, row in manifest.iterrows():
        audio_path = cfg.wav_dir / row["file_name"]
        print(f"[{i + 1}/{len(manifest)}] {row['file_name']}...", end=" ", flush=True)

        # A single clip failing outright (e.g. Gemini staying under sustained
        # "high demand" 503s well past gemini_client.transcribe's own
        # retry/backoff budget) shouldn't take down the whole run - that's
        # real, observed behaviour on the free tier, not a hypothetical.
        # Record it as a failed sample and keep going, rather than losing
        # every clip that already succeeded.
        try:
            result = transcribe(client, audio_path, cfg.language_name, model_name)
            hypothesis = result["hypothesis"]
            confidence = result["confidence"]
            wer = compute_wer([row["transcript"]], [hypothesis])
            error = None
            print(f"wer={wer:.3f}")
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
            hypothesis = None
            confidence = None
            wer = None
            error = str(exc)
            print(f"FAILED: {error}")

        per_sample.append(
            {
                "file_name": row["file_name"],
                "reference": row["transcript"],
                "hypothesis": hypothesis,
                "wer": wer,
                "confidence": confidence,
                "error": error,
            }
        )

    succeeded = [r for r in per_sample if r["error"] is None]
    failed = [r for r in per_sample if r["error"] is not None]

    references = [r["reference"] for r in succeeded]
    hypotheses = [r["hypothesis"] for r in succeeded]
    overall_wer = compute_wer(references, hypotheses) if succeeded else None

    # Not every clip is guaranteed a confidence value (see gemini/lib/gemini_client.py)
    # - average only over the ones that have one, rather than treating a
    # missing value as 0 and skewing the mean down.
    confidences = [r["confidence"] for r in succeeded if r["confidence"] is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None

    result = {
        "language": cfg.language_name,
        "model": model_name,
        "split": args.split,
        "num_samples": len(per_sample),
        "num_failed": len(failed),
        "overall_wer": overall_wer,
        "mean_confidence": mean_confidence,
        "date": datetime.now(timezone.utc).isoformat(),
        "samples": per_sample,
    }

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.split("/")[-1]
    json_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.json"
    csv_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(per_sample).to_csv(csv_path, index=False)

    wer_str = f"{overall_wer:.3f}" if overall_wer is not None else "n/a"
    conf_str = f"{mean_confidence:.3f}" if mean_confidence is not None else "n/a"
    print(
        f"{cfg.language_name} / {model_name} / {args.split}: "
        f"WER={wer_str}  mean_confidence={conf_str}  "
        f"n={len(per_sample)} (failed={len(failed)})"
    )
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
