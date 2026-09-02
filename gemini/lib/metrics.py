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
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute the overall Word Error Rate across a list of correct transcripts
    (`references`) versus what the model actually produced (`hypotheses`). The
    two lists must line up one-to-one. A single number is returned summarizing
    accuracy across all of them combined, not one number per clip.
    """
    # jiwer.wer can't handle a reference that normalizes down to nothing (e.g.
    # a "transcript" that was just a punctuation mark) - skip those pairs
    # rather than letting the whole calculation crash, same as whisper's
    # compute_wer.
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r and r.strip()]
    if not pairs:
        return 0.0
    refs, hyps = zip(*pairs)
    return jiwer.wer(
        list(refs),
        list(hyps),
        reference_transform=_normalize,
        hypothesis_transform=_normalize,
    )
