"""WER scoring shared by chirp/scripts/*.py.

Identical to gemini/lib/metrics.py (see that file's docstring for why this
uses jiwer's built-in normalizer instead of transformers' BasicTextNormalizer)
- duplicated rather than imported so chirp/ stays independent of gemini/.
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
    return _normalize(text or "")


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

    # Filter on the NORMALIZED reference - see gemini/lib/metrics.py's
    # compute_wer for why checking the raw text here was the actual bug.
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
