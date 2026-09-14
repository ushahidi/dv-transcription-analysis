"""CLI: run Google Cloud Speech-to-Text (Chirp) over a language's local test
set and report WER, in the same shape as whisper/scripts/run_baseline.py and
gemini/scripts/run_baseline.py. Per-sample confidence is not reported here -
see chirp/lib/stt_client.py's transcribe docstring for why.

A single clip failing does not abort the run - it's recorded as a failed
sample with an empty ("") hypothesis, which counts against overall_wer as a
fully-wrong transcript rather than being quietly dropped from scoring. If any
clip fails, the script still writes full results, then exits non-zero.

BLOCKED ON GCP SETUP, not on unfinished code - the script itself is complete
and ready to run; what's missing is a GCP project with billing enabled, the
Speech-to-Text API turned on, and a credential:
GOOGLE_APPLICATION_CREDENTIALS (a service-account JSON key) OR
`gcloud auth application-default login` (no JSON file needed either way),
plus GOOGLE_CLOUD_PROJECT set. See chirp/README.md for the full setup.

Usage (once GCP setup is done):
    # PowerShell
    $env:GOOGLE_CLOUD_PROJECT = "your-project-id"
    $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\path\\to\\service-account.json"
    # bash
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

    python chirp/scripts/run_baseline.py --language kiswahili --split eval
    python chirp/scripts/run_baseline.py --language kiswahili --split eval --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also add the repo root so `from data.lib...` below can find the shared
# data/ package (load_manifest lives there now, not duplicated per engine -
# see data/lib/dataset_utils.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from data.lib.dataset_utils import load_manifest
from lib.config import load_config
from lib.metrics import compute_wer
from lib.stt_client import make_client, project_id, transcribe


def _positive_int(value: str) -> int:
    """argparse type for --limit: must be a whole number >= 1 - see
    whisper/scripts/run_baseline.py's `_positive_int` for why (0 silently
    means "no limit", negative silently becomes DataFrame.head(-n))."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"--limit must be >= 1, got {value}")
    return ivalue


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
        help="Cloud STT model id (e.g. chirp_3, chirp_2). Defaults to stt.model in "
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
        type=_positive_int,
        default=None,
        help="Only evaluate the first N rows of the split (e.g. --limit 100 out of a "
        "698-clip 'eval' split) - for a quicker/cheaper run without downloading a "
        "separate smaller split. Must be >= 1.",
    )
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_name = args.model or cfg.stt_model
    project = project_id()

    manifest = load_manifest(cfg, args.split)
    if args.limit:
        manifest = manifest.head(args.limit)
    client = make_client(cfg.gcp_location)

    per_sample = []
    for i, row in manifest.iterrows():
        audio_path = cfg.wav_dir / row["file_name"]
        print(f"[{i + 1}/{len(manifest)}] {row['file_name']}...", end=" ", flush=True)

        # A single clip failing outright (e.g. Cloud STT staying under
        # sustained 5xx/rate-limit errors past transcribe's own retry/backoff
        # budget) shouldn't take down the whole run - see
        # gemini/scripts/run_baseline.py and wav2vec2/scripts/run_baseline.py,
        # which already do this. Record it as a failed sample and keep going,
        # rather than losing every clip that already succeeded.
        try:
            result = transcribe(
                client,
                audio_path,
                project,
                cfg.gcp_location,
                model_name,
                cfg.stt_language_codes,
            )
            hypothesis = result["hypothesis"] or ""
            confidence = result["confidence"]
            error = None
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
            hypothesis = ""
            confidence = None
            error = str(exc)

        # This clip's own WER can independently fail to be computable (e.g.
        # its reference is punctuation-only) even when transcription itself
        # succeeded - a scoring problem with this one clip, not an engine
        # failure, so it's tracked separately from `error`.
        try:
            wer = compute_wer([row["transcript"]], [hypothesis])
        except ValueError:
            wer = None

        if error:
            print(f"FAILED: {error}")
        else:
            print(f"wer={wer:.3f}" if wer is not None else "wer=n/a (unscorable reference)")

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

    failed = [r for r in per_sample if r["error"] is not None]

    # Every sample goes into the aggregate, failed ones included with their
    # "" hypothesis - see gemini/scripts/run_baseline.py's module docstring
    # for why dropping failures instead would flatter the WER. Only raises
    # if literally every reference in the whole split is unscorable.
    references = [r["reference"] for r in per_sample]
    hypotheses = [r["hypothesis"] for r in per_sample]
    try:
        overall_wer = compute_wer(references, hypotheses) if per_sample else None
    except ValueError:
        overall_wer = None

    # Cloud STT's `confidence` is currently always None (see
    # chirp/lib/stt_client.py's transcribe docstring) - this averages only
    # over non-None values, same pattern as gemini/scripts/run_baseline.py,
    # so mean_confidence naturally comes out as None rather than a
    # misleading ~0.0 constant.
    confidences = [r["confidence"] for r in per_sample if r["confidence"] is not None]
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

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
