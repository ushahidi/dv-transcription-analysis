"""WER scoring shared by wav2vec2/scripts/*.py.

Identical to whisper/lib/metrics.py's `compute_wer` and `normalize`,
including the same transformers-based `BasicTextNormalizer`. Unlike
gemini/lib/metrics.py and chirp/lib/metrics.py (which deliberately avoid a
transformers dependency, since neither of those folders otherwise needs
torch/transformers at all), this folder already requires torch/transformers
to load the wav2vec2 model itself - so there's no dependency-weight reason to
use a different normalizer here, and using the exact same one as whisper/
keeps WER numbers genuinely apples-to-apples with the Whisper baselines this
folder exists to compare against.
"""

from __future__ import annotations

from typing import List, Sequence

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer(text)


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

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
