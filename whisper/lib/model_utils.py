"""Model/processor loading shared by run_baseline.py, finetune.py, export_model.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Loading a Whisper model correctly involves a few fiddly steps that are easy to
get subtly wrong (and easy to copy-paste incorrectly between scripts), so they
live here once instead of being repeated in every script:

  - Loading the "processor" (the piece that turns raw audio into the numbers
    the model expects, and turns the model's numeric output back into text).
  - Loading the model itself and telling it which language and task
    ("transcribe", as opposed to "translate to English") to expect.
  - Optionally wrapping the model with a lightweight fine-tuning technique
    called LoRA/PEFT, for faster experimentation.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .config import PipelineConfig


def load_processor(cfg: PipelineConfig, model_name: Optional[str] = None) -> WhisperProcessor:
    """Load the "processor" for a given Whisper model.

    The processor bundles together two things: a feature extractor (converts a
    raw sound wave into the spectrogram-like numeric format Whisper reads) and
    a tokenizer (converts between text and the numeric "tokens" the model
    actually works with internally). Telling it the language and task up front
    means it will produce the correct special starting tokens for that
    language/task automatically.
    """
    name = model_name or cfg.model_name
    return WhisperProcessor.from_pretrained(name, language=cfg.language_name, task="transcribe")


def load_model(cfg: PipelineConfig, model_name: Optional[str] = None) -> WhisperForConditionalGeneration:
    """Download (or load from local cache) a Whisper model and configure it to
    transcribe in the right language.

    Whisper is a "multilingual" model - the same weights can transcribe many
    languages, or even translate into English, depending on how it's told to
    behave for a given run. We explicitly set that behaviour here rather than
    relying on defaults, so every script in this project behaves consistently.
    """
    name = model_name or cfg.model_name

    # Some checkpoints (e.g. openai/whisper-large-v3) have their weights stored
    # on the Hub in fp16. Loading without an explicit torch_dtype keeps them in
    # that stored precision, which then mismatches the fp32 audio features
    # whenever we're not actually running fp16 (e.g. this project's CPU runs),
    # crashing with "Input type (float) and bias type (Half) should be the
    # same" on the very first conv layer. Pin the dtype explicitly instead of
    # trusting whatever precision the checkpoint happens to be stored in.
    torch_dtype = torch.float16 if cfg.fp16 else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch_dtype)

    # Tell the model which language to expect and that we want a literal
    # transcript (not a translation into English). `forced_decoder_ids` is an
    # older, now-discouraged way of doing this same thing - if it's left set,
    # it can silently override the settings above, so we explicitly clear it.
    model.generation_config.language = cfg.language_name
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Move the model's numbers onto whichever device we resolved earlier
    # (cfg.device is "cpu" on this machine, or "cuda" on a machine with a
    # compatible graphics card, e.g. Google Colab).
    model.to(cfg.device)
    return model


def maybe_wrap_peft(model, cfg: PipelineConfig):
    """Optionally wrap the model with a "LoRA" adapter for lighter-weight
    fine-tuning, if the config/`--use-peft` flag asked for it. If not, the
    model is returned completely unchanged.

    WHY THIS EXISTS: normal fine-tuning updates every single number in the
    entire model, which is slow and memory-hungry. LoRA instead freezes the
    original model and adds a small number of new, trainable "adapter"
    weights alongside a couple of its attention layers (q_proj/v_proj - the
    parts of the model responsible for deciding which parts of the audio to
    "pay attention to"). Only those small adapter weights get trained, which
    is much faster and lighter, at some cost to how much the model can learn -
    a reasonable tradeoff for quick experiments on a CPU, and still useful on
    a GPU for faster iteration.
    """
    if not cfg.train.get("use_peft", False):
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=32,  # how large the new adapter weights are - bigger can learn more, but is slower
        lora_alpha=64,  # scaling factor applied to the adapter's effect on the model
        target_modules=["q_proj", "v_proj"],  # which parts of the model get an adapter
        lora_dropout=0.05,  # randomly ignores 5% of adapter connections during training, to avoid overfitting
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # prints how few parameters are actually being trained, for visibility
    return model
