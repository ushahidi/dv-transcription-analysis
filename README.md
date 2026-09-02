# dv-transcription-analysis

Testing and fine-tuning environment for speech-to-text providers, tuned towards
African languages (starting with Kiswahili, then Chichewa) for Ushahidi's
transcription pipeline.

## Shared eval data

[data/](data/) is the one shared source of test audio + transcripts - every
engine folder below reads the same downloaded clips from here rather than
keeping its own copy (see [data/scripts/prepare_dataset.py](data/scripts/prepare_dataset.py)).
This started as four separate copies of a fixed 20-clip FLEURS test set
(cheap to duplicate); once the eval set grew past 500 clips, one shared copy
replaced them. The current Kiswahili `eval` split is FLEURS `sw_ke`'s `test`
(487) + `validation` (211) splits combined = 698 clips, all held-out (never
`train`):

```powershell
python data/scripts/prepare_dataset.py --language kiswahili --split eval
```

## Engines

Each engine gets its own independent top-level folder - own dependencies, own
model-loading/scoring code, reading that same shared `data/` - so results are
directly comparable without the folders sharing *code*:

- [whisper/README.md](whisper/README.md) - Whisper setup, baseline WER evaluation,
  fine-tuning, and model export workflow.
- [gemini/README.md](gemini/README.md) - Gemini Flash (free Developer API) baseline
  WER evaluation.
- [chirp/README.md](chirp/README.md) - Google Cloud Speech-to-Text (Chirp) baseline
  WER evaluation. Not runnable yet - needs a GCP project + billing set up first.
- [wav2vec2/README.md](wav2vec2/README.md) - open, self-hosted wav2vec2/CTC
  checkpoints (`thinkKenya/wav2vec2-large-xls-r-300m-sw`, Meta's MMS) baseline
  WER evaluation.
