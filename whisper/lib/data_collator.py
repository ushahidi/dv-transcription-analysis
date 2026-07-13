"""Data collator for Whisper seq2seq fine-tuning - the standard recipe from
huggingface.co/blog/fine-tune-whisper, used as-is by whisper/scripts/finetune.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
When training a model, examples are grouped into small "batches" (e.g. 2 or 16
clips at a time) so the model can learn from several examples at once. But
different audio clips and different transcripts naturally come in different
lengths - one sentence might be 5 words, another 20. Before they can be
stacked together into a single batch, the shorter ones need to be "padded"
(filled with placeholder values) so every item in the batch has the same
length.

This file defines exactly how that padding happens for Whisper - one padding
scheme for the audio features, and a different, careful one for the text
labels (since padding on the text side needs a special "ignore this" marker so
the model isn't taught that the padding is real content to predict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Callable object that takes a list of individually-prepared training
    examples and combines them into one padded batch, ready to feed to the model.

    `processor` is passed in once when this collator is created and reused for
    every batch during training (see whisper/scripts/finetune.py).
    """

    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # --- Pad the audio side ---
        # Every example already has its audio converted into "input_features"
        # (a numeric representation of the sound - done earlier in
        # finetune.py). Whisper always expects a fixed-length audio input, so
        # this padding is mostly a formality, but we still go through the
        # processor's own padding logic to keep everything consistent.
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # --- Pad the text side ---
        # Each example's transcript has already been converted into a list of
        # numeric "tokens" (the tokenizer's internal vocabulary IDs), stored
        # under "labels". Shorter transcripts get padded with the tokenizer's
        # normal padding token here...
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # ...but then we swap every padding position to -100. -100 is a special
        # value that tells the training code "don't count this position when
        # computing how wrong the model's guess was" - i.e. ignore the padding
        # entirely rather than teaching the model that clips should always end
        # with a string of padding tokens.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The tokenizer sometimes automatically adds a special "start of
        # sequence" token at the very beginning of the labels. The training
        # code adds its own start token separately when it feeds these labels
        # into the model, so if we left the tokenizer's copy in, every example
        # would end up with that start token twice. This removes the extra one.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
