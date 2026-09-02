# Shared eval data

The one place test audio + transcripts actually get downloaded to, shared by
every engine folder (`whisper/`, `gemini/`, `chirp/`, `wav2vec2/`) - each of
them reads `data/<language>/manifest_<split>.csv` + `data/<language>/wav/` and
scores against it, instead of keeping its own copy.

## Why this exists

Each engine folder used to download its own private copy of the same FLEURS
clips - deliberate while the shared test set was a fixed 20 samples (cheap to
duplicate, and it kept every folder's code fully independent of the others -
see e.g. `gemini/README.md`'s original "Why a separate folder" reasoning).
That stopped being cheap once the eval set grew past 500 clips: four copies
of ~700 audio clips is real, avoidable disk and download time. This folder is
now the one download; every engine's `lib/config.py` points its
`wav_dir`/`manifest_path` here instead (see e.g. `whisper/lib/config.py`'s
`SHARED_DATA_ROOT`).

Only the *data* was consolidated, not the code - each engine folder still has
its own independent config/metrics/model-calling code, per this repo's
established per-engine-folder convention. `whisper/` is the one exception: it
also keeps its own `prepare_dataset.py`/`dataset_utils.py` download logic,
because it alone fine-tunes (`train`/`validation` splits for `finetune.py`) -
something none of the other three engines do. Even so, `whisper/`'s own
downloads now land in this same shared folder (its `data_dir` points here
too), so there's still only ever one copy on disk.

## The `eval` split

FLEURS `sw_ke`'s own `test` split is only 487 clips - short of a 500+-clip
target. `eval` is `test` (487) + `validation` (211) combined = **698 clips**,
both held-out (never `train`), so this doesn't blur the train/eval distinction
- it just draws from two held-out splits instead of one. Every engine's
`run_baseline.py` defaults `--split` to `eval`.

## Layout

```
data/
├── lib/                  # config loading + the actual download logic
├── scripts/
│   └── prepare_dataset.py
├── english/
│   ├── config.yaml       # tracked - dataset_id/dataset_config/language identity
│   ├── wav/              # generated (gitignored)
│   └── manifest_eval.csv # generated (gitignored)
└── kiswahili/             # same structure
```

## Usage

```powershell
# The large-scale eval set used for the engine comparison (698 clips):
python data/scripts/prepare_dataset.py --language kiswahili --split eval

# A single underlying HF split, capped or not - e.g. for whisper/'s own
# fine-tuning (whisper/scripts/prepare_dataset.py does this too, writing to
# the same place):
python data/scripts/prepare_dataset.py --language kiswahili --split train --max-samples 80
```

Manifest CSV columns: `file_name, transcript, duration_sec, split,
source_dataset, source_split, source_id` - `source_split` records which
underlying HF split each row actually came from (`test` or `validation`, for
`eval`), so it's still possible to recover the original 487/211 breakdown
later if that distinction ever matters.

## Adding a new language

Create `data/<language>/config.yaml` with `language_name`, `language_code`,
`dataset_id`, `dataset_config`, `text_column`, `audio_column` (see
`data/kiswahili/config.yaml`), then run `prepare_dataset.py --language
<language> --split eval` and point each engine folder's own
`<engine>/<language>/config.yaml` at it.
