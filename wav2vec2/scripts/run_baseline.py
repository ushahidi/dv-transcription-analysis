"""CLI: run a wav2vec2/CTC checkpoint over a language's local test set and
report WER plus a per-sample confidence proxy, in the same shape as
whisper/scripts/run_baseline.py.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
The wav2vec2 counterpart to whisper/scripts/run_baseline.py - see that
file's docstring for the full "what is WER/confidence" explanation. Same
job, different model architecture (CTC instead of Whisper's autoregressive
generation - see wav2vec2/lib/model_utils.py for what that changes).

Results are saved to wav2vec2/<language>/results/ as JSON + CSV, in the same
shape whisper/, gemini/, and chirp/'s run_baseline.py scripts produce, so all
four are directly diffable/comparable.

Usage:
    python wav2vec2/scripts/run_baseline.py --language kiswahili --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf

from lib.config import load_config
from lib.dataset_utils import load_manifest
from lib.metrics import compute_wer
from lib.model_utils import load_model, load_processor, transcribe


def _read_wav(path: Path, target_sr: int):
    """Read a .wav file from disk into a plain array of numbers - see
    whisper/scripts/run_baseline.py's `_read_wav` for the full explanation."""
    audio, sr = sf.read(path)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


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
        help="Hugging Face model name (e.g. thinkKenya/wav2vec2-large-xls-r-300m-sw). "
        "Defaults to model.name in config.yaml.",
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
    model_name = args.model or cfg.model_name

    manifest = load_manifest(cfg, args.split)
    if args.limit:
        manifest = manifest.head(args.limit)

    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model.eval()

    per_sample = []
    for i, row in manifest.iterrows():
        audio_path = cfg.wav_dir / row["file_name"]
        print(f"[{i + 1}/{len(manifest)}] {row['file_name']}...", end=" ", flush=True)

        try:
            audio = _read_wav(audio_path, cfg.sample_rate)
            result = transcribe(model, processor, audio, cfg.sample_rate, cfg.device)
            hypothesis = result["hypothesis"]
            confidence = result["confidence"]
            wer = compute_wer([row["transcript"]], [hypothesis])
            error = None
            print(f"wer={wer:.3f}")
        except Exception as exc:  # noqa: BLE001 - one bad clip shouldn't kill the whole run
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
    mean_confidence = sum(r["confidence"] for r in succeeded) / len(succeeded) if succeeded else None

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
