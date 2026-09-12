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
downloads land in this same shared folder (its `data_dir` points here too), so
there's still only ever one copy on disk.

**Prefer one writer per split.** `whisper/scripts/prepare_dataset.py` is
scoped to `train`/`validation` only - it cannot write `test` or `eval` (that
used to be possible, and was a real risk: whisper's own config defaults
`test` to only 20 samples, so an accidental `whisper/scripts/prepare_dataset.py
--split test` would have silently overwritten the full 487-clip `test` pull
with a 20-clip one, out from under `eval`'s next regeneration). This script
(`data/scripts/prepare_dataset.py`) is the only thing that ever writes
`test`/`eval`. It's technically capable of writing `train`/`validation` too
(nothing stops you passing `--split train`), but in practice leave those to
`whisper/scripts/prepare_dataset.py` - don't run both against the same split.

## The `eval` split

FLEURS `sw_ke`'s own `test` split is only 487 clips - short of a 500+-clip
target. `eval` is `test` (487) + `validation` (211) combined = **698 clips**,
both held-out (never `train`), so this doesn't blur the train/eval distinction
- it just draws from two held-out splits instead of one. Every engine's
`run_baseline.py` defaults `--split` to `eval`.

**This is fine for comparing pretrained/API models against each other** (none
of Whisper/Gemini/Chirp/wav2vec2's off-the-shelf checkpoints were fine-tuned
on this repo's `validation` split, so there's no leakage today). **It is NOT
fine for scoring a model this repo has fine-tuned**: `whisper/scripts/finetune.py`
trains against exactly this `validation` split for early stopping/checkpoint
selection, so evaluating that fine-tuned model against `eval` would leak its
own validation data into its reported WER. Use `--split test` (the 487-clip
split alone) to evaluate a fine-tuned model instead - see
`whisper/README.md`'s fine-tuning section.

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

## Setup

```powershell
pip install -r data/requirements.txt
```

None of the engine folders' own `requirements.txt` include what this script
needs (`datasets`/`soundfile`/`librosa`/`pandas`/`pyyaml`) - installing e.g.
`gemini/requirements.txt` alone leaves you missing these at prepare_dataset.py
time, since gemini/chirp only need them for downloading (which they no longer
do themselves), not for evaluation.

## Usage

```powershell
# The large-scale eval set used for the engine comparison (698 clips):
python data/scripts/prepare_dataset.py --language kiswahili --split eval

# A single underlying HF split, capped or not - e.g. to top up something a
# future engine needs that isn't train/validation/test/eval:
python data/scripts/prepare_dataset.py --language kiswahili --split test --max-samples 50
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

## The `run_baseline.py` result contract

Every engine's `run_baseline.py` (`whisper/`, `gemini/`, `chirp/`,
`wav2vec2/`) writes `baseline_<model>_<split>.json` in this exact shape - a
fifth engine should match it, not invent its own field names, or the
comparison across engines breaks silently instead of loudly:

```jsonc
{
  "language": "swahili",
  "model": "gemini-2.5-flash",
  "split": "eval",
  "num_samples": 698,          // every row attempted, success or failure
  "num_failed": 0,             // count of samples with a non-null "error"
  "overall_wer": 0.131,        // null only if every reference was unscorable
  "mean_confidence": 0.94,     // null if no sample had a confidence value
  "date": "2026-...T...Z",     // UTC ISO 8601
  "samples": [
    {
      "file_name": "eval_0000.wav",
      "reference": "...",       // ground truth, verbatim from the manifest
      "hypothesis": "...",      // "" for a failed sample - never null
      "wer": 0.05,              // null only if THIS reference was unscorable
      "confidence": 0.97,       // null if this engine has none for this sample
      "error": null             // null on success, else the failure message
    }
  ]
}
```

Rules that keep engines comparable, not just individually correct:
- A failed sample is `"hypothesis": ""` and counted in `overall_wer` as
  fully wrong - never dropped from scoring (that would flatter a flaky
  engine by only scoring the clips that happened to succeed).
- `wer: null` means something different from a failure: it's this one
  sample's *reference* that was unscorable (e.g. punctuation-only), which
  can happen independently of whether transcription itself succeeded.
- `confidence` is engine-specific and not directly comparable across
  engines (see each engine's README for how it's computed, or whether it's
  always `null` - Chirp's always is, see `chirp/README.md`) - only `overall_wer`
  is meant to be compared head-to-head.
