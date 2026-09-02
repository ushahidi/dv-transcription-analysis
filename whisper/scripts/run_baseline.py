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

Usage:
    python whisper/scripts/run_baseline.py --language english --model openai/whisper-tiny
    python whisper/scripts/run_baseline.py --language kiswahili --split validation
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

    # The manifest is the spreadsheet listing every clip in this split and its
    # correct transcript - produced earlier by prepare_dataset.py.
    manifest = load_manifest(cfg, args.split)
    if args.limit:
        manifest = manifest.head(args.limit)

    # Load the model and its processor once, before the loop, since loading is
    # slow and we want to reuse the same loaded model for every clip.
    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model.eval()  # tell the model we're only using it to make predictions, not to train it

    per_sample = []  # will hold one result dictionary per audio clip
    for _, row in manifest.iterrows():
        # Load this clip's actual sound wave from disk.
        audio = _read_wav(cfg.wav_dir / row["file_name"], cfg.sample_rate)

        # Convert the raw sound wave into the numeric format Whisper expects
        # (a spectrogram-like representation), and move it onto whichever
        # device we're running on (CPU here, potentially GPU elsewhere).
        inputs = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(cfg.device)

        # Ask the model to actually transcribe this clip. `output_scores=True`
        # and `return_dict_in_generate=True` make it also hand back internal
        # probability information we use below to compute a confidence score.
        # `torch.no_grad()` tells PyTorch we're not training, so it can skip
        # some bookkeeping it would otherwise do to support learning - this
        # makes transcription noticeably faster.
        with torch.no_grad():
            outputs = model.generate(
                input_features,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=225,  # a safety cap on how long a transcript can be
            )

        # Convert the model's numeric output tokens back into readable text.
        hypothesis = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # How confident was the model in this particular transcript? (0-1 scale)
        confidence = sequence_confidence(model, outputs.sequences, outputs.scores)[0]

        # How many words did it get wrong, comparing to the known-correct transcript?
        wer = compute_wer([row["transcript"]], [hypothesis])

        per_sample.append(
            {
                "file_name": row["file_name"],
                "reference": row["transcript"],   # the correct, known transcript
                "hypothesis": hypothesis,          # what the model guessed
                "wer": wer,
                "confidence": confidence,
            }
        )

    # --- Summarize across every clip we just tested ---
    references = [r["reference"] for r in per_sample]
    hypotheses = [r["hypothesis"] for r in per_sample]
    overall_wer = compute_wer(references, hypotheses)
    mean_confidence = sum(r["confidence"] for r in per_sample) / len(per_sample) if per_sample else 0.0

    result = {
        "language": cfg.language_name,
        "model": model_name,
        "split": args.split,
        "num_samples": len(per_sample),
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

    print(
        f"{cfg.language_name} / {model_name} / {args.split}: "
        f"WER={overall_wer:.3f}  mean_confidence={mean_confidence:.3f}  n={len(per_sample)}"
    )
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
