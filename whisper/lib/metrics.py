"""WER scoring and confidence extraction shared by run_baseline.py and finetune.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Once Whisper has produced a text guess ("hypothesis") for what was said in an
audio clip, we need a way to measure how good that guess is compared to the
correct transcript ("reference"). This file answers two separate questions:

  1. "How many words did the model get wrong?" -> Word Error Rate (WER).
     This is the industry-standard accuracy number for speech recognition.
     WER = 0.0 means a perfect transcript; WER = 1.0 (or 100%) means, roughly,
     as many mistakes as there are words; it can even go above 1.0 if the
     model inserts lots of extra wrong words.

  2. "How sure was the model of its own answer?" -> a confidence score between
     0 and 1, based on how strongly the model believed each word/token it
     generated. This is useful for flagging low-confidence transcripts for a
     human to double check, separately from WER (which we can only compute
     when we already know the correct answer - confidence works even on brand
     new audio with no known correct answer).
"""

from __future__ import annotations

from typing import List, Sequence

import jiwer
import torch
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Before comparing two pieces of text, it helps to "normalize" them first -
# lowercase everything, strip out punctuation, and collapse extra spaces - so
# that trivial differences (a capital letter, a stray comma) don't get counted
# as mistakes. BasicTextNormalizer (NOT EnglishTextNormalizer) is used here
# specifically because it is language-agnostic: it does the generic cleanup
# above without English-specific rules (like spelling out numbers), so it's
# safe to use on both English and Kiswahili text.
_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    """Clean up a piece of text (lowercase, strip punctuation/extra spaces)
    before comparing it to another piece of text."""
    return _normalizer(text)


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute the overall Word Error Rate across a list of correct transcripts
    (`references`) versus what the model actually produced (`hypotheses`).

    The two lists must line up one-to-one (references[i] is the correct
    transcript for hypotheses[i]). A single number is returned summarizing
    accuracy across *all* of them combined, not one number per clip.
    """
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

    # The WER math can't handle a reference that normalizes down to nothing at
    # all (e.g. a "transcript" that was just a punctuation mark) - so we skip
    # any such pairs rather than letting the whole calculation crash.
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    if not pairs:
        # NOT 0.0: an empty pair list means there was nothing to score, not a
        # perfect score. Returning 0.0 here would look identical to "the
        # model transcribed everything correctly" - silently misleading in
        # exactly the cases (e.g. a --limit run against one punctuation-only
        # reference) where the WER means nothing at all.
        raise ValueError(
            "compute_wer: every reference normalized to empty text (e.g. "
            "punctuation-only) - there is nothing to score, so no WER can be "
            "computed for this batch."
        )
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


def sequence_confidence(model, sequences: torch.Tensor, scores) -> List[float]:
    """Work out, for each transcript the model generated, roughly how
    confident it was in that answer - a number from 0 (not at all sure) to 1
    (very sure).

    HOW THIS WORKS UNDER THE HOOD (only matters if you're modifying this code):
    When Whisper generates text, for every single word-piece it produces it
    also has an internal probability of "how likely is this the right
    word-piece". `model.compute_transition_scores` retrieves those
    probabilities (as log-probabilities, a mathematically convenient form) for
    each generated sequence. We then average those probabilities across every
    real word-piece in the sequence (ignoring padding added to make a batch of
    different-length sequences line up) and convert back from log form to a
    plain 0-1 probability with `torch.exp`.

    This confidence is a SEQUENCE-level number - one score for the *entire*
    transcript, averaged over every word-piece in it - not a per-word
    confidence flag. A future improvement, if per-word flags are needed, would
    be to line these same per-token probabilities up against word boundaries
    instead of averaging them all together - noted in whisper/README.md.
    """
    transition_scores = model.compute_transition_scores(sequences, scores, normalize_logits=True)

    # Shorter sequences get padded with -infinity so every row in the batch has
    # the same length - `valid` marks which positions are real generated
    # tokens (as opposed to that padding), so we only average over real ones.
    valid = transition_scores != float("-inf")
    lengths = valid.sum(dim=-1).clamp(min=1)  # clamp(min=1) avoids a divide-by-zero
    avg_log_prob = transition_scores.masked_fill(~valid, 0.0).sum(dim=-1) / lengths

    # Convert from log-probability back to an ordinary 0-1 probability/confidence.
    return torch.exp(avg_log_prob).tolist()
