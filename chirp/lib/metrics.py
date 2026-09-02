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
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
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
