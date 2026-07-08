"""CLI: fine-tune a Whisper checkpoint on a language's local train/validation set.

Locally (CPU), this is a smoke test - it proves the training loop runs, loss trends
down, and a checkpoint saves/reloads. It is not a production-quality model. Real
fine-tuning happens on a GPU (e.g. Google Colab via whisper/colab_finetune.ipynb),
using default_config.yaml's GPU-scale hyperparameters instead of a language's
CPU-smoke-test config.yaml overrides - the script itself does not change.

Usage:
    python whisper/scripts/finetune.py --language english --model openai/whisper-tiny
    python whisper/scripts/finetune.py --language kiswahili --use-peft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import soundfile as sf
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from lib.config import load_config
from lib.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from lib.dataset_utils import manifest_to_dataset
from lib.model_utils import load_model, load_processor, maybe_wrap_peft

wer_metric = evaluate.load("wer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["english", "kiswahili"])
    parser.add_argument("--model", default=None, help="Defaults to model.name in config.yaml.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-peft", action="store_true", help="Wrap the model with a LoRA adapter.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_name = args.model or cfg.model_name
    if args.use_peft:
        cfg.train["use_peft"] = True

    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model = maybe_wrap_peft(model, cfg)

    def prepare(batch):
        audio_array, sr = sf.read(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    train_ds = manifest_to_dataset(cfg, "train").map(prepare, remove_columns=["audio_path", "transcript"])
    eval_ds = manifest_to_dataset(cfg, "validation").map(prepare, remove_columns=["audio_path", "transcript"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    train_cfg = cfg.train
    output_dir = cfg.output_dir(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        fp16=cfg.fp16,
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 2),
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        generation_max_length=train_cfg.get("generation_max_length", 225),
        logging_steps=train_cfg.get("logging_steps", 25),
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned model to {output_dir}")


if __name__ == "__main__":
    main()
