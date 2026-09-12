"""Integration test for the mixed success/failure aggregation logic shared
by every engine's run_baseline.py (gemini's is exercised here as the
representative case - whisper/chirp/wav2vec2 all follow the identical
pattern, see each script's module docstring).

Verifies the actual review finding this guards against: a failed clip used
to be silently dropped from the aggregate WER (scoring "the easy subset"
that happened to succeed), which would flatter a flaky engine's number
relative to one that failed less often. Now a failed clip counts as a fully
wrong ("" hypothesis) transcript in the aggregate, and num_failed is reported
so the two numbers can be told apart - see gemini/scripts/run_baseline.py's
module docstring.

This imports gemini.scripts.run_baseline directly and monkeypatches its
make_client/transcribe - the sole direct import of an engine's scripts/*.py
in this suite (see tests/test_wav2vec2_mms.py's module docstring for why a
second one in the same session would risk a bare `lib` name collision;
there's only ever one here, so it's safe).
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import gemini.scripts.run_baseline as run_baseline


def _write_wav(path, duration_sec: float = 0.3, sr: int = 16000):
    audio = np.zeros(int(duration_sec * sr), dtype="float32")
    sf.write(path, audio, sr)


@pytest.fixture
def fake_shared_data(tmp_path, monkeypatch):
    """Point gemini/lib/config.py's SHARED_DATA_ROOT at a throwaway temp
    directory instead of the repo's real data/, and populate it with a
    tiny 3-clip 'eval' manifest - two clips that will "succeed" and one
    that will "fail", so the test controls the outcome directly rather
    than depending on real downloaded audio or a real API key."""
    data_root = tmp_path / "data"
    kiswahili_dir = data_root / "kiswahili"
    wav_dir = kiswahili_dir / "wav"
    wav_dir.mkdir(parents=True)

    rows = []
    for i, name in enumerate(["ok_one", "fails", "ok_two"]):
        file_name = f"eval_{i:04d}.wav"
        _write_wav(wav_dir / file_name)
        rows.append({"file_name": file_name, "transcript": f"reference for {name}"})
    pd.DataFrame(rows).to_csv(kiswahili_dir / "manifest_eval.csv", index=False)

    # gemini/scripts/run_baseline.py imports load_config via its own
    # sys.path-prepended bare `lib.config` (see its sys.path.insert calls),
    # which is a DIFFERENT module object in sys.modules than the
    # qualified-name `gemini.lib.config` this test could import instead -
    # patching that one would silently do nothing here. `__globals__` on
    # any function pulled from the actual module in use is identity-proof:
    # it's the literal namespace dict PipelineConfig.__init__ (defined in
    # the same file) also reads SHARED_DATA_ROOT from.
    real_load_config = run_baseline.load_config
    monkeypatch.setitem(real_load_config.__globals__, "SHARED_DATA_ROOT", data_root)

    # Also redirect results_dir to this same temp directory - otherwise the
    # test would write baseline_*.json into the real repo's
    # gemini/kiswahili/results/, clobbering any real results already there.
    # load_config's OWN config.yaml/default_config.yaml lookups are left
    # alone (reading them has no side effects), only the output path.
    def _load_config_with_temp_results(language, config_path=None):
        cfg = real_load_config(language, config_path)
        cfg.results_dir = tmp_path / "results"
        return cfg

    monkeypatch.setattr(run_baseline, "load_config", _load_config_with_temp_results)

    return kiswahili_dir


def test_mixed_success_failure_aggregation(fake_shared_data, monkeypatch, tmp_path):
    # Fake transcribe(): succeeds for two clips (exact match, wer=0), raises
    # for the middle one - simulating a real transcription failure (a
    # network error, an exhausted retry budget, whatever).
    def _fake_transcribe(client, audio_path, language_name, model_name):
        if audio_path.name == "eval_0001.wav":  # the "fails" clip, see fake_shared_data
            raise RuntimeError("simulated transcription failure")
        # audio_path is named e.g. eval_0000.wav - match it back to its
        # reference text via the manifest, so this is scored as a perfect
        # transcription for the two clips that "succeed".
        manifest = pd.read_csv(fake_shared_data / "manifest_eval.csv")
        row = manifest[manifest["file_name"] == audio_path.name].iloc[0]
        return {"hypothesis": row["transcript"], "confidence": 0.9}

    monkeypatch.setattr(run_baseline, "make_client", lambda: object())
    monkeypatch.setattr(run_baseline, "transcribe", _fake_transcribe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_baseline.py", "--language", "kiswahili", "--split", "eval"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_baseline.main()
    # Non-zero exit because one of the three clips failed - see the module
    # docstring: a failed run must be detectable from the exit code alone.
    assert exc_info.value.code == 1

    result_path = tmp_path / "results" / "baseline_gemini-2.5-flash_eval.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["num_samples"] == 3
    assert result["num_failed"] == 1

    samples_by_file = {s["file_name"]: s for s in result["samples"]}
    assert samples_by_file["eval_0001.wav"]["error"] is not None
    assert samples_by_file["eval_0001.wav"]["hypothesis"] == ""
    # A failed clip is scored against its real (scoreable) reference with an
    # empty hypothesis - fully wrong (wer=1.0), not None. wer=None is a
    # different, separate case (the reference itself being unscorable -
    # punctuation-only, say), unrelated to whether the engine succeeded.
    assert samples_by_file["eval_0001.wav"]["wer"] == pytest.approx(1.0)

    # The failed clip is NOT dropped from the headline WER - it's included
    # with its "" hypothesis, so overall_wer reflects all three clips, not
    # just the two that happened to succeed. With 2 perfect + 1 fully-wrong
    # (empty) out of 3, overall_wer must be > 0 - a silently-dropped failure
    # would instead report a perfect 0.0 (only ever scoring the easy subset).
    assert result["overall_wer"] > 0.0
    assert result["overall_wer"] == pytest.approx(1 / 3)
