"""WER scoring and confidence extraction shared by run_baseline.py and finetune.py."""

from __future__ import annotations

from typing import List, Sequence

import jiwer
import torch
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# BasicTextNormalizer (not EnglishTextNormalizer) is language-agnostic - it lowercases,
# strips punctuation, and collapses whitespace without English-specific spelling/number
# rules, so it is safe to apply to both English and Kiswahili references/hypotheses.
_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer(text)


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER after basic text normalization."""
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]
    # jiwer can't align an empty reference - drop pairs where the reference normalizes
    # to nothing (e.g. a source transcript that was just punctuation).
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    if not pairs:
        return 0.0
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


def sequence_confidence(model, sequences: torch.Tensor, scores) -> List[float]:
    """Derive a [0, 1] confidence proxy per generated sequence from
    model.generate(..., output_scores=True, return_dict_in_generate=True).scores.

    `sequences_scores` (the obvious shortcut) is only populated by HF's generate()
    for beam search - for the greedy decoding used here it's absent, so this instead
    calls `model.compute_transition_scores` to get each generated token's
    log-probability, averages over the real (non-padding) tokens of each sequence,
    and exponentiates.

    This is a SEQUENCE-level confidence (averaged over every generated token), not a
    per-word confidence flag. Word-level confidence would align transition scores to
    word boundaries instead - noted in whisper/README.md as a future enhancement if
    per-word flags become a real requirement.
    """
    transition_scores = model.compute_transition_scores(sequences, scores, normalize_logits=True)
    valid = transition_scores != float("-inf")
    lengths = valid.sum(dim=-1).clamp(min=1)
    avg_log_prob = transition_scores.masked_fill(~valid, 0.0).sum(dim=-1) / lengths
    return torch.exp(avg_log_prob).tolist()
