"""Dataset download shared by data/scripts/prepare_dataset.py.

Same download mechanics every engine folder used to duplicate (see e.g.
whisper/lib/dataset_utils.py's original `_stream_split`/`download_split`,
whose docstrings this borrows) - `decode=False` + manual `soundfile` decoding
to avoid an ffmpeg/torchcodec dependency, streaming rather than a bulk
download so only the samples actually asked for get pulled.

One addition over the old per-engine versions: `download_split` can combine
MULTIPLE underlying Hugging Face splits into one local split. This exists
because FLEURS' own `sw_ke` `test` split is only 487 clips - short of a
500+-clip target - so the local "eval" split used for the large-scale
comparison is `test` (487) + `validation` (211) = 698 clips, combined here
into one manifest/wav set. Both source splits are held-out data (never
`train`), so this doesn't blur the train/eval distinction - it just draws
the eval set from two held-out splits instead of one.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import librosa
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

from .config import DatasetConfig


def _stream_split(cfg: DatasetConfig, hf_split: str, max_samples: Optional[int]) -> Iterator[dict]:
    """Connect to the online dataset and hand back examples one at a time
    from one specific underlying Hugging Face split (e.g. "test"),
    stopping once `max_samples` have been yielded (or exhausting the split,
    if `max_samples` is None)."""
    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=hf_split, streaming=True)
    ds = ds.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sample_rate, decode=False))
    for i, example in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield example


def download_split(
    cfg: DatasetConfig,
    local_split: str,
    source_splits: List[str],
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Download `source_splits` (one or more underlying Hugging Face splits,
    e.g. ["test", "validation"]), concatenate them in the order given, and
    save the result as one local split named `local_split` - a manifest CSV
    plus one .wav file per clip, named `<local_split>_%04d.wav`.

    `max_samples` caps the TOTAL across every source split combined, applied
    greedily in the order `source_splits` is given (all of the first split,
    then the second, etc., until the cap is hit). Leave it as None - the
    default, and what the "eval" split uses - to pull every available clip
    from each source split; the whole point of "eval" is to use everything
    held-out that FLEURS has, not a subset.
    """
    cfg.wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counter = 0
    for hf_split in source_splits:
        remaining = None if max_samples is None else max_samples - counter
        if remaining is not None and remaining <= 0:
            break

        for example in _stream_split(cfg, hf_split, remaining):
            audio_field = example[cfg.audio_column]
            array, sr = sf.read(io.BytesIO(audio_field["bytes"]))

            if sr != cfg.sample_rate:
                array = librosa.resample(array, orig_sr=sr, target_sr=cfg.sample_rate)
                sr = cfg.sample_rate

            file_name = f"{local_split}_{counter:04d}.wav"
            sf.write(cfg.wav_dir / file_name, array, sr)
            duration_sec = len(array) / sr

            rows.append(
                {
                    "file_name": file_name,
                    "transcript": example[cfg.text_column],
                    "duration_sec": round(duration_sec, 3),
                    "split": local_split,
                    "source_dataset": cfg.dataset_id,
                    "source_split": hf_split,
                    "source_id": example.get("id", counter),
                }
            )
            counter += 1

    df = pd.DataFrame(rows)
    cfg.language_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.manifest_path(local_split), index=False)

    _update_dataset_card(cfg, local_split, source_splits, len(df))
    return df


def load_manifest(cfg, split: str) -> pd.DataFrame:
    """Read back the manifest CSV for a split that was already downloaded.

    This is now THE load_manifest every engine folder's run_baseline.py
    calls - gemini/, chirp/, and wav2vec2/ used to each keep an identical
    copy of this exact function (only ever reading, never downloading,
    since data/ took over downloading); this is the one copy, and those
    three were deleted. It works for any `cfg` with a `.manifest_path(split)`
    method, which includes both this file's own `DatasetConfig` and every
    engine's own `PipelineConfig` (whisper/gemini/chirp/wav2vec2 all define
    one) - duck-typed on purpose so this doesn't need to import any of them.
    """
    path = cfg.manifest_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for split '{split}' at {path}. Run "
            f"`python data/scripts/prepare_dataset.py --language {cfg.language} "
            f"--split {split}` first (or whisper/scripts/prepare_dataset.py for "
            f"'train'/'validation' - see data/README.md's \"Prefer one writer\" note)."
        )
    return pd.read_csv(path)


def _update_dataset_card(cfg: DatasetConfig, local_split: str, source_splits: List[str], num_samples: int) -> None:
    path = cfg.dataset_card_path()
    card = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    card["dataset_id"] = cfg.dataset_id
    card["dataset_config"] = cfg.dataset_config
    card["language"] = cfg.language_name
    card["license"] = "CC-BY-4.0"
    card.setdefault("splits", {})
    card["splits"][local_split] = {
        "source_splits": source_splits,
        "num_samples": num_samples,
        "sample_rate": cfg.sample_rate,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
