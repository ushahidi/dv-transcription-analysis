"""Data collator for Whisper seq2seq fine-tuning - the standard recipe from
huggingface.co/blog/fine-tune-whisper, used as-is by whisper/scripts/finetune.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The tokenizer may have prepended a BOS token that the trainer re-adds at
        # generation time - strip it here so it isn't duplicated.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
