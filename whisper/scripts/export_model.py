"""CLI: sanity-check a fine-tuned Whisper checkpoint (reload + transcribe one sample)
and export it to CTranslate2 (int8-quantized) for fast CPU inference via faster-whisper.

Usage:
    python whisper/scripts/export_model.py --language english \\
        --model-dir whisper/english/models/whisper-tiny-finetuned
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from lib.config import load_config
from lib.dataset_utils import load_manifest


def _sanity_check(model_dir: Path, cfg) -> None:
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.to(cfg.device)
    model.eval()

    manifest = load_manifest(cfg, "test")
    sample = manifest.iloc[0]
    audio, _ = sf.read(cfg.wav_dir / sample["file_name"])
    inputs = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features.to(cfg.device), max_new_tokens=225)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Sanity check transcription ({sample['file_name']}): {text!r}")


def _find_ct2_converter() -> str:
    """Locate the ct2-transformers-converter console script. Calling `python.exe`
    directly (rather than an activated venv shell) doesn't put the venv's Scripts/
    dir on PATH, so check next to the current interpreter first."""
    script_name = "ct2-transformers-converter.exe" if sys.platform == "win32" else "ct2-transformers-converter"
    candidate = Path(sys.executable).parent / script_name
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ct2-transformers-converter")
    if found:
        return found
    raise FileNotFoundError(
        "ct2-transformers-converter not found next to the current Python interpreter or on PATH - "
        "make sure the `ctranslate2` package is installed in this environment."
    )


def _export_ctranslate2(model_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _find_ct2_converter(),
        "--model",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--quantization",
        "int8",
        "--force",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["english", "kiswahili"])
    parser.add_argument("--model-dir", required=True, help="Path to a fine-tuned HF model dir (from finetune.py).")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--skip-ct2",
        action="store_true",
        help="Only run the reload/transcribe sanity check, skip the CTranslate2 export.",
    )
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} does not exist - run finetune.py first.")

    _sanity_check(model_dir, cfg)

    if not args.skip_ct2:
        ct2_dir = model_dir.parent / f"{model_dir.name}-ct2"
        _export_ctranslate2(model_dir, ct2_dir)
        print(f"CTranslate2 export: {ct2_dir}")


if __name__ == "__main__":
    main()
