"""WER scoring shared by gemini/scripts/*.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Same job as whisper/lib/metrics.py's `compute_wer` - measure how many words a
transcript got wrong compared to the known-correct reference. Deliberately
re-implemented here with a different normalizer, though: whisper/lib/metrics.py
uses `transformers.models.whisper.english_normalizer.BasicTextNormalizer`, which
would pull the whole `transformers`/torch stack into this folder just for text
cleanup. Since this folder is meant to stay a light install (no local model, no
torch), normalization here uses jiwer's own built-in transforms instead -
lowercase, strip punctuation, collapse extra whitespace. The effect is
essentially the same as BasicTextNormalizer for this project's purposes, but the
two are NOT guaranteed byte-for-byte identical - keep that in mind if you're
diffing WER numbers between whisper/ and gemini/ down to the third decimal
place; the numbers are comparable, not derived from the exact same code path.
"""

from __future__ import annotations

from typing import Sequence

import jiwer

_normalize = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def normalize(text: str) -> str:
    """Clean up a piece of text (lowercase, strip punctuation/extra spaces)
    before comparing it to another piece of text. Same role as
    whisper/lib/metrics.py's `normalize`, different implementation - see the
    module docstring above."""
    return _normalize(text or "")


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute the overall Word Error Rate across a list of correct transcripts
    (`references`) versus what the model actually produced (`hypotheses`). The
    two lists must line up one-to-one. A single number is returned summarizing
    accuracy across all of them combined, not one number per clip.
    """
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

    # Filter on the NORMALIZED reference, not the raw one - a raw reference
    # like "..." is non-empty and would slip past a raw-text check, only to
    # normalize down to nothing anyway. (Checking the raw text here used to
    # be the actual bug: whisper/lib/metrics.py and wav2vec2/lib/metrics.py
    # already filtered post-normalization; this filtered pre-normalization,
    # which is the inconsistency that mattered.)
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    if not pairs:
        # NOT 0.0 - see whisper/lib/metrics.py's compute_wer for why an empty
        # pair list must not look like a perfect score.
        raise ValueError(
            "compute_wer: every reference normalized to empty text (e.g. "
            "punctuation-only) - there is nothing to score, so no WER can be "
            "computed for this batch."
        )
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))
