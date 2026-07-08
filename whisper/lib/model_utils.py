"""Model/processor loading shared by run_baseline.py, finetune.py, export_model.py."""

from __future__ import annotations

from typing import Optional

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .config import PipelineConfig


def load_processor(cfg: PipelineConfig, model_name: Optional[str] = None) -> WhisperProcessor:
    name = model_name or cfg.model_name
    return WhisperProcessor.from_pretrained(name, language=cfg.language_name, task="transcribe")


def load_model(cfg: PipelineConfig, model_name: Optional[str] = None) -> WhisperForConditionalGeneration:
    name = model_name or cfg.model_name
    model = WhisperForConditionalGeneration.from_pretrained(name)
    # Modern generation_config-based language/task steering, per the HF fine-tuning
    # recipe - forced_decoder_ids is the legacy mechanism and must be nulled out to
    # avoid it silently overriding generation_config.
    model.generation_config.language = cfg.language_name
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(cfg.device)
    return model


def maybe_wrap_peft(model, cfg: PipelineConfig):
    """Wrap `model` with a LoRA adapter targeting Whisper's attention q/v projections
    when cfg.train['use_peft'] is set. Returns `model` unmodified otherwise."""
    if not cfg.train.get("use_peft", False):
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
