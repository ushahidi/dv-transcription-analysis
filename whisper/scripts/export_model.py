"""CLI: sanity-check a fine-tuned Whisper checkpoint (reload + transcribe one sample)
and export it to CTranslate2 (int8-quantized) for fast CPU inference via faster-whisper.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
After fine-tuning a model (whisper/scripts/finetune.py), we want two things:

  1. A quick sanity check that the saved model actually works - reload it from
     disk exactly as any other program would, and confirm it can still
     transcribe a sample clip sensibly.
  2. A converted copy of the model in a different, more deployment-friendly
     format (CTranslate2), which runs noticeably faster for real-world use -
     particularly for CPU-only inference - than the original training format.
     Think of this like exporting a document from an editable working format
     into a smaller, faster-to-open format for actually using it.

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

# Let this script import from the sibling whisper/lib/ folder regardless of
# which directory it's actually launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from lib.config import load_config
from lib.dataset_utils import load_manifest


def _sanity_check(model_dir: Path, cfg) -> None:
    """Reload the fine-tuned model completely fresh from disk (exactly as any
    other program using it later would) and transcribe one clip from the test
    set, just to prove the saved files are complete and usable - not checking
    accuracy here, just "does this even work at all"."""
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.to(cfg.device)
    model.eval()

    manifest = load_manifest(cfg, "test")
    sample = manifest.iloc[0]  # just grab the first test clip, that's enough for a sanity check
    audio, _ = sf.read(cfg.wav_dir / sample["file_name"])
    inputs = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features.to(cfg.device), max_new_tokens=225)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Sanity check transcription ({sample['file_name']}): {text!r}")


def _find_ct2_converter() -> str:
    """Locate the ct2-transformers-converter program on this computer.

    This program gets installed alongside the `ctranslate2` Python package,
    but it lives in a "Scripts" folder that is only automatically available
    if you've formally "activated" the Python virtual environment in your
    terminal. Since we sometimes call Python directly instead (e.g.
    `.venv\\Scripts\\python.exe script.py`), we look for the converter right
    next to whichever Python program is currently running, before falling
    back to a normal system-wide search.
    """
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
    """Convert the model at `model_dir` into the CTranslate2 format and save it
    to `output_dir`, using 8-bit ("int8") number precision instead of the
    usual 32-bit - this makes the exported model noticeably smaller and faster
    to run, at a very small, generally unnoticeable cost to accuracy."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _find_ct2_converter(),
        "--model",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--quantization",
        "int8",
        "--force",  # overwrite output_dir if it already exists from a previous export
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)  # check=True: raise an error here if the conversion tool itself fails


def main() -> None:
    # --- Read the command-line options ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "kiswahili"],
        help="Which language this model was fine-tuned for (used to find its test clip).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to a fine-tuned model folder produced by finetune.py.",
    )
    parser.add_argument("--config", default=None, help="Override path to a config.yaml.")
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
