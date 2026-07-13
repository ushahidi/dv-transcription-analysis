"""CLI: fine-tune a Whisper checkpoint on a language's local train/validation set.

WHAT THIS SCRIPT DOES, IN PLAIN TERMS:
"Fine-tuning" means taking a Whisper model that already knows how to transcribe
speech in general, and further training it specifically on our own example
clips, so it gets better at whatever it was struggling with (e.g. Kiswahili,
which the baseline test showed the plain pretrained model is very bad at).

The process is: show the model one of our audio clips, let it guess the
transcript, compare that guess to the real transcript, and nudge the model's
internal numbers slightly so it would guess a bit better next time. Repeat this
many times (each repeat is called a "step") across all our training clips.

Locally (on this CPU-only machine), this is a smoke test only - it proves the
training loop runs correctly, the loss (the "how wrong was the guess" number)
trends downward over the smoke-test's handful of steps, and a checkpoint (a
saved copy of the partially-trained model) saves to disk and can be reloaded.
It is NOT a production-quality model - there simply isn't enough time or data
in a CPU smoke test to properly train it.

Real fine-tuning happens on a machine with a graphics card (GPU), e.g. Google
Colab via whisper/colab_finetune.ipynb, using default_config.yaml's GPU-scale
settings (many more steps, bigger batches) instead of a language's
CPU-smoke-test config.yaml overrides. Importantly, this script itself does not
change between the two runs - only the config values do.

Usage:
    python whisper/scripts/finetune.py --language english --model openai/whisper-tiny
    python whisper/scripts/finetune.py --language kiswahili --use-peft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let this script import from the sibling whisper/lib/ folder regardless of
# which directory it's actually launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import soundfile as sf
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from lib.config import load_config
from lib.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from lib.dataset_utils import manifest_to_dataset
from lib.model_utils import load_model, load_processor, maybe_wrap_peft

# `evaluate` is a small Hugging Face library specifically for computing
# standard metrics like WER in a way that matches how everyone else in the
# field computes it, so results are comparable across projects. We load the
# WER calculator once here, rather than inside the training loop, since
# loading it involves a one-time download the first time it's used.
wer_metric = evaluate.load("wer")


def main() -> None:
    # --- Read the command-line options ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "kiswahili"],
        help="Which language's train/validation data to fine-tune on.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base Hugging Face model to start fine-tuning from (e.g. openai/whisper-tiny). "
        "Defaults to model.name in config.yaml.",
    )
    parser.add_argument("--config", default=None, help="Override path to a config.yaml.")
    parser.add_argument(
        "--use-peft",
        action="store_true",
        help="Wrap the model with a LoRA adapter for faster, lighter-weight fine-tuning "
        "(trains far fewer numbers than a full fine-tune - see lib/model_utils.py).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a previously saved checkpoint folder to continue training from "
        "(useful if a long GPU run gets interrupted, e.g. on Colab).",
    )
    args = parser.parse_args()

    cfg = load_config(args.language, args.config)
    model_name = args.model or cfg.model_name
    if args.use_peft:
        cfg.train["use_peft"] = True

    # Load the processor (converts audio <-> numbers, text <-> tokens) and the
    # base model we're about to fine-tune.
    processor = load_processor(cfg, model_name)
    model = load_model(cfg, model_name)
    model = maybe_wrap_peft(model, cfg)  # optionally switch to the lighter-weight LoRA training mode

    def prepare(batch):
        """Convert one raw training example (an audio file path + its correct
        transcript text) into the exact numeric format the model needs to
        learn from: a numeric "input_features" representation of the sound,
        and a numeric "labels" representation of the correct text."""
        audio_array, sr = sf.read(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    # Load the manifests for the "train" and "validation" splits (downloaded
    # earlier by prepare_dataset.py) and run every example through `prepare`
    # above so they're ready for training. `remove_columns` drops the original
    # raw columns once we no longer need them, to save memory.
    train_ds = manifest_to_dataset(cfg, "train").map(prepare, remove_columns=["audio_path", "transcript"])
    eval_ds = manifest_to_dataset(cfg, "validation").map(prepare, remove_columns=["audio_path", "transcript"])

    # The data collator's job is to combine several individually-prepared
    # examples into one padded "batch" during training - see
    # whisper/lib/data_collator.py for exactly how the padding works.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        """Called automatically during training (every `eval_steps` steps) to
        check how the model is doing on the validation set - i.e. examples it
        is NOT being trained on directly, used as an honest progress check."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        # Undo the -100 "ignore this" padding marker (see data_collator.py)
        # before turning the numeric labels back into readable text.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # WER is normally shown as a percentage (0-100) rather than a fraction
        # (0-1) in training logs, hence multiplying by 100 here.
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    train_cfg = cfg.train
    output_dir = cfg.output_dir(model_name)

    # `Seq2SeqTrainingArguments` is Hugging Face's big list of "how should
    # training behave" settings. Every value here is read from config.yaml
    # (via cfg.train) rather than hard-coded, so the CPU smoke-test config and
    # a future GPU config can set completely different values without this
    # script needing to change at all.
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),  # where checkpoints get saved during training
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],  # how many clips per training step
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg["learning_rate"],  # how big a nudge each step makes to the model's numbers
        warmup_steps=train_cfg["warmup_steps"],  # ramp the learning rate up gradually at the very start
        max_steps=train_cfg["max_steps"],  # total number of training steps to run
        fp16=cfg.fp16,  # "half precision" speed-up - only ever true when actually running on a GPU
        eval_strategy=train_cfg.get("eval_strategy", "steps"),  # check validation accuracy every N steps
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg.get("save_strategy", "steps"),  # save a checkpoint every N steps
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 2),  # keep only the N most recent checkpoints (saves disk space)
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        generation_max_length=train_cfg.get("generation_max_length", 225),
        logging_steps=train_cfg.get("logging_steps", 25),  # how often to print progress
        report_to=["none"],  # don't try to log to any external tracking service (e.g. Weights & Biases)
        load_best_model_at_end=True,  # at the very end, use whichever checkpoint scored best, not just the last one
        metric_for_best_model="wer",  # "best" is judged by validation WER...
        greater_is_better=False,  # ...and lower WER is better (fewer mistakes)
    )

    # `Seq2SeqTrainer` is Hugging Face's ready-made training loop - it takes
    # everything we've set up above (the model, the data, the padding logic,
    # the accuracy-checking function, and all the settings) and actually runs
    # the training steps.
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # This line is where the actual training happens - it can take anywhere
    # from a couple of minutes (a small CPU smoke test) to hours (a real GPU
    # fine-tuning run), depending on the settings above.
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the final (best) model and its processor to disk, so later scripts
    # (run_baseline.py, export_model.py) can load it back by folder path.
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned model to {output_dir}")


if __name__ == "__main__":
    main()
