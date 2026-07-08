"""CLI: run a pretrained (or fine-tuned) Whisper checkpoint over a language's local
test set and report WER plus a per-sample confidence proxy.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf
import torch

from lib.config import load_config
from lib.dataset_utils import load_manifest
from lib.metrics import compute_wer, sequence_confidence
from lib.model_utils import load_model, load_processor


def _read_wav(path: Path, target_sr: int):
    audio, sr = sf.read(path)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["english", "kiswahili"])
    parser.add_argument("--model", default=None, help="Defaults to model.name in config.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_name = args.model or cfg.model_name

    manifest = load_manifest(cfg, args.split)
    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model.eval()

    per_sample = []
    for _, row in manifest.iterrows():
        audio = _read_wav(cfg.wav_dir / row["file_name"], cfg.sample_rate)
        inputs = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(cfg.device)

        with torch.no_grad():
            outputs = model.generate(
                input_features,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=225,
            )
        hypothesis = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        confidence = sequence_confidence(model, outputs.sequences, outputs.scores)[0]
        wer = compute_wer([row["transcript"]], [hypothesis])

        per_sample.append(
            {
                "file_name": row["file_name"],
                "reference": row["transcript"],
                "hypothesis": hypothesis,
                "wer": wer,
                "confidence": confidence,
            }
        )

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
        "samples": per_sample,
    }

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.split("/")[-1]
    json_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.json"
    csv_path = cfg.results_dir / f"baseline_{safe_model}_{args.split}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(per_sample).to_csv(csv_path, index=False)

    print(
        f"{cfg.language_name} / {model_name} / {args.split}: "
        f"WER={overall_wer:.3f}  mean_confidence={mean_confidence:.3f}  n={len(per_sample)}"
    )
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
