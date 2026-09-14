"""CLI: run a pretrained (or fine-tuned) Whisper checkpoint over a language's local
test set and report WER plus a per-sample confidence proxy.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
This is the "how accurate is this model?" script. It takes every audio clip in
a language's test set (downloaded earlier by prepare_dataset.py), asks a
Whisper model to transcribe each one, and compares that guess against the
known-correct transcript. It reports:

  - WER (Word Error Rate) for each clip and overall - the standard "how many
    words did it get wrong" accuracy measure.
  - A confidence score for each clip - how sure the model seemed of its own
    answer, which is useful even when we DON'T know the correct answer (e.g.
    on brand new, real-world audio).

Results are saved to whisper/<language>/results/ as both a JSON file (easy for
other programs to read) and a CSV file (easy to open in Excel).

This same script works for a plain pretrained model (e.g. "openai/whisper-tiny")
or for a model this project has already fine-tuned (by passing its folder path
to --model), so it can be used both "before" and "after" fine-tuning to see
whether accuracy actually improved.

A single clip failing (a bad file, a transient error) does not abort the run -
it's recorded as a failed sample with an empty ("") hypothesis, which counts
against overall_wer as a fully-wrong transcript rather than being quietly
dropped from scoring (dropping it would flatter the WER by only scoring the
"easy" clips that happened to succeed). If any clip fails, the script still
writes full results, then exits non-zero at the end - see main()'s tail.

Usage:
    python whisper/scripts/run_baseline.py --language english --model openai/whisper-tiny
    python whisper/scripts/run_baseline.py --language kiswahili --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Let this script import from the sibling whisper/lib/ folder regardless of
# which directory it's actually launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf
import torch

from lib.config import load_config
from lib.dataset_utils import load_manifest
from lib.metrics import compute_wer, sequence_confidence
from lib.model_utils import load_model, load_processor


def _read_wav(path: Path, target_sr: int):
    """Read a .wav file from disk into a plain array of numbers (the sound
    wave). If, for some reason, the file's sample rate doesn't match what
    Whisper expects (16kHz), stretch/compress it (resample) so it does.
    """
    audio, sr = sf.read(path)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def _positive_int(value: str) -> int:
    """argparse type for --limit: must be a whole number >= 1.

    Without this, `if args.limit:` treats 0 as "no limit" (confusing but
    survivable) and a negative number silently becomes `DataFrame.head(-n)` -
    pandas' "everything except the last n rows", which is a very surprising
    thing for --limit to mean and would look like a data bug, not a CLI
    mistake.
    """
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"--limit must be >= 1, got {value}")
    return ivalue


def main() -> None:
    # --- Read the command-line options ---
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
        help="Hugging Face model name (e.g. openai/whisper-tiny) or a local folder path "
        "to a fine-tuned model. Defaults to model.name in config.yaml.",
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=["train", "validation", "test", "eval"],
        help="Which downloaded split to evaluate against. 'eval' (the default) is the "
        "shared, larger FLEURS test+validation set in ../data/ - see data/README.md. "
        "NOTE: 'eval' includes this project's own fine-tuning validation split - after "
        "fine-tuning a model on this repo's train/validation data, re-evaluate it with "
        "--split test instead, or 'eval' will leak validation data into its own score.",
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

    # The manifest is the spreadsheet listing every clip in this split and its
    # correct transcript - produced earlier by prepare_dataset.py (for
    # train/validation) or data/scripts/prepare_dataset.py (for eval/test) -
    # see lib/dataset_utils.py's load_manifest for which one it'll point you
    # at if it's missing.
    manifest = load_manifest(cfg, args.split)
    if args.limit:
        manifest = manifest.head(args.limit)

    # Load the model and its processor once, before the loop, since loading is
    # slow and we want to reuse the same loaded model for every clip.
    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model.eval()  # tell the model we're only using it to make predictions, not to train it

    per_sample = []  # will hold one result dictionary per audio clip
    for i, row in manifest.iterrows():
        print(f"[{i + 1}/{len(manifest)}] {row['file_name']}...", end=" ", flush=True)

        # A single clip failing outright (a corrupt file, an OOM, whatever)
        # shouldn't take down the whole run and lose every already-completed
        # transcription - see gemini/chirp/wav2vec2's run_baseline.py, which
        # already do this. Record it as a failed sample and keep going.
        # Deliberately NOT excluded from scoring below: an empty ("")
        # hypothesis is scored as fully wrong, not dropped - see the module
        # docstring for why dropping failures would flatter the WER.
        try:
            audio = _read_wav(cfg.wav_dir / row["file_name"], cfg.sample_rate)

            inputs = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(cfg.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_features,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=225,  # a safety cap on how long a transcript can be
                )

            hypothesis = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            confidence = sequence_confidence(model, outputs.sequences, outputs.scores)[0]
            error = None
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
            hypothesis = ""
            confidence = None
            error = str(exc)

        # This clip's own WER can independently fail to be computable (e.g.
        # its reference is punctuation-only) even when transcription itself
        # succeeded - that's a scoring problem with this one clip, not an
        # engine failure, so it's tracked separately from `error`.
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
                "reference": row["transcript"],   # the correct, known transcript
                "hypothesis": hypothesis,          # what the model guessed ("" if it failed)
                "wer": wer,
                "confidence": confidence,
                "error": error,
            }
        )

    failed = [r for r in per_sample if r["error"] is not None]

    # --- Summarize across every clip we just tested ---
    # Every sample goes into the aggregate, failed ones included with their
    # "" hypothesis - see the module docstring for why. Only if literally
    # every reference in the whole split turns out unscorable does this
    # raise; that's a pathological input, not a normal outcome, so it's
    # caught here rather than crashing the whole script without writing
    # whatever results DO exist.
    references = [r["reference"] for r in per_sample]
    hypotheses = [r["hypothesis"] for r in per_sample]
    try:
        overall_wer = compute_wer(references, hypotheses) if per_sample else None
    except ValueError:
        overall_wer = None

    # Not every clip is guaranteed a confidence value - average only over the
    # ones that have one, rather than treating a missing value as 0 and
    # skewing the mean down.
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
        "samples": per_sample,  # the full per-clip detail, for closer inspection
    }

    # --- Save the results to disk in two formats ---
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.split("/")[-1]
    json_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.json"  # full detail, machine-readable
    csv_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.csv"    # per-clip table, spreadsheet-friendly
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

    # Non-zero exit when anything failed, so this is safe to depend on in
    # scripts/CI without also reading the JSON just to notice a partial run -
    # the results are still written above either way.
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
