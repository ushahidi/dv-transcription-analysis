"""Tests for compute_wer across all four engines' lib/metrics.py copies.

Covers the review findings this suite exists to lock in:
  - identical strings score 0.0 (sanity check nothing else broke)
  - a punctuation-only/empty reference is skipped when it's one of several
    pairs, not counted and not crashing the batch
  - when EVERY pair is skip-worthy, compute_wer raises rather than silently
    returning 0.0 (which would look like a perfect score for data that was
    never actually scoreable)
  - gemini's jiwer-based normalizer and wav2vec2/whisper's
    transformers-based BasicTextNormalizer disagree on the exact normalized
    string (see test_metrics.py's manual check: "..." -> " " for
    BasicTextNormalizer, "" for jiwer's transforms) but must still agree on
    the *behavior* that matters: an all-empty batch raises either way.
"""

import pytest

import chirp.lib.metrics as chirp_metrics
import gemini.lib.metrics as gemini_metrics
import wav2vec2.lib.metrics as wav2vec2_metrics
import whisper.lib.metrics as whisper_metrics

ALL_METRICS_MODULES = [gemini_metrics, chirp_metrics, wav2vec2_metrics, whisper_metrics]
MODULE_IDS = ["gemini", "chirp", "wav2vec2", "whisper"]


@pytest.mark.parametrize("mod", ALL_METRICS_MODULES, ids=MODULE_IDS)
def test_identical_strings_score_zero(mod):
    assert mod.compute_wer(["hello world"], ["hello world"]) == 0.0


@pytest.mark.parametrize("mod", ALL_METRICS_MODULES, ids=MODULE_IDS)
def test_one_wrong_word_out_of_two(mod):
    assert mod.compute_wer(["hello world"], ["hello there"]) == pytest.approx(0.5)


@pytest.mark.parametrize("mod", ALL_METRICS_MODULES, ids=MODULE_IDS)
def test_punctuation_only_reference_is_skipped_not_counted(mod):
    # One normal pair (correct) + one punctuation-only reference. The bad
    # pair must be dropped, not crash the batch and not get counted as an
    # error against the overall score.
    wer = mod.compute_wer(["hello world", "..."], ["hello world", "some hypothesis"])
    assert wer == 0.0


@pytest.mark.parametrize("mod", ALL_METRICS_MODULES, ids=MODULE_IDS)
def test_all_references_unscorable_raises_not_zero(mod):
    # NOT 0.0 - this is the actual bug being guarded against. A batch where
    # every reference normalizes to nothing has no WER to report; returning
    # 0.0 would look identical to "the model got everything right".
    with pytest.raises(ValueError):
        mod.compute_wer(["..."], ["anything"])
    with pytest.raises(ValueError):
        mod.compute_wer(["", "   ", "!!!"], ["a", "b", "c"])


def test_gemini_and_wav2vec2_agree_on_all_punctuation_batch():
    # Different normalizer implementations - jiwer's own transforms
    # (gemini/chirp) vs transformers' BasicTextNormalizer (whisper/wav2vec2)
    # - produce different normalized STRINGS for punctuation-only text
    # (BasicTextNormalizer maps "..." to " ", jiwer's transforms map it to
    # ""), but both must still treat it as unscorable rather than disagree
    # on whether an all-empty batch is a real 0.0 score.
    with pytest.raises(ValueError):
        gemini_metrics.compute_wer(["!!!"], ["something"])
    with pytest.raises(ValueError):
        wav2vec2_metrics.compute_wer(["!!!"], ["something"])


@pytest.mark.parametrize("mod", [gemini_metrics, chirp_metrics], ids=["gemini", "chirp"])
def test_gemini_chirp_filter_on_normalized_not_raw_reference(mod):
    # The actual bug this guards against: gemini/chirp used to filter on the
    # RAW reference text (`if r and r.strip()`), so "..." (non-empty as raw
    # text) would slip past the filter and only fail once jiwer normalized
    # it internally. Filtering on the normalized text catches this the same
    # way whisper/wav2vec2 already did.
    with pytest.raises(ValueError):
        mod.compute_wer(["..."], ["hello"])
