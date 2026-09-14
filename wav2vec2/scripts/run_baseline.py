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

A single clip failing does not abort the run - it's recorded as a failed
sample with an empty ("") hypothesis, which counts against overall_wer as a
fully-wrong transcript rather than being quietly dropped from scoring. If any
clip fails, the script still writes full results, then exits non-zero.

Usage:
    python wav2vec2/scripts/run_baseline.py --language kiswahili --split eval
    python wav2vec2/scripts/run_baseline.py --language kiswahili --split eval --limit 100
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
import soundfile as sf

from data.lib.dataset_utils import load_manifest
from lib.config import load_config
from lib.metrics import compute_wer
from lib.model_utils import load_model, load_processor, transcribe


def _read_wav(path: Path, target_sr: int):
    """Read a .wav file from disk into a plain array of numbers - see
    whisper/scripts/run_baseline.py's `_read_wav` for the full explanation.

    `soundfile` returns a 2D (frames, channels) array for a multi-channel
    file, not the 1D array the rest of this pipeline (and librosa.resample)
    assumes - resampling that shape directly would resample along the wrong
    axis. FLEURS' clips are mono so this hasn't fired yet, but downmixing to
    mono first (plain average across channels) keeps this correct for any
    stereo file too."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


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
        help="Hugging Face model name (e.g. thinkKenya/wav2vec2-large-xls-r-300m-sw). "
        "Defaults to model.name in config.yaml. WARNING: this only swaps the "
        "checkpoint id - it does NOT change model.type/model.lang_code, which still "
        "come from config.yaml. Passing an MMS checkpoint here while config.yaml is "
        "set up for a plain single-language model (model.type: ctc) will fail loudly "
        "instead of silently skipping adapter selection - see the mismatch check below. "
        "Use --config wav2vec2/<language>/config_mms.yaml to select MMS correctly.",
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
    model_name = args.model or cfg.model_name

    # MMS checkpoints (adapter-based, need model.type: mms + model.lang_code
    # to actually select a language - see wav2vec2/lib/model_utils.py) can be
    # swapped in via --model without touching config.yaml at all, which means
    # cfg.model_type would silently stay "ctc" and load_model/load_processor
    # would silently skip set_target_lang/load_adapter entirely - producing a
    # loaded-but-wrong model with no error. This is a heuristic (checkpoint
    # id contains "mms"), not a certainty, but it's a fail-closed guess: real
    # MMS checkpoints on the Hub are named facebook/mms-*, so this catches
    # the case this review flagged without needing a hardcoded checkpoint list.
    if "mms" in model_name.lower() and cfg.model_type != "mms":
        parser.error(
            f"'{model_name}' looks like an MMS adapter-based checkpoint, but "
            f"{args.language}'s config.yaml has model.type: '{cfg.model_type}' (no "
            f"adapter would be loaded, and the result would silently use whichever "
            f"adapter this checkpoint happens to default to). Use "
            f"--config wav2vec2/{args.language}/config_mms.yaml instead of --model to "
            f"select MMS correctly."
        )

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

        # A single clip failing outright shouldn't take down the whole run -
        # see the module docstring. Deliberately NOT excluded from scoring
        # below: an empty ("") hypothesis is scored as fully wrong, not
        # dropped.
        try:
            audio = _read_wav(audio_path, cfg.sample_rate)
            result = transcribe(model, processor, audio, cfg.sample_rate, cfg.device)
            hypothesis = result["hypothesis"] or ""
            confidence = result["confidence"]
            error = None
        except Exception as exc:  # noqa: BLE001 - one bad clip shouldn't kill the whole run
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

    # Not every clip is guaranteed a confidence value - average only over the
    # ones that have one, same pattern as gemini/chirp's run_baseline.py (a
    # plain sum(...)/len(...) over every sample would TypeError the moment
    # any confidence is None, e.g. after a failed clip).
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
