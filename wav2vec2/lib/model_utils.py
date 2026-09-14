"""Model loading + CTC transcription shared by wav2vec2/scripts/run_baseline.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
The wav2vec2 counterpart to whisper/lib/model_utils.py - but the underlying
model architecture is fundamentally different, so the "run one clip through
the model" logic can't be shared. Whisper is an encoder-decoder model that
autoregressively *generates* a token sequence (`model.generate(...)`);
wav2vec2/CTC models are encoder-only - one forward pass produces a probability
distribution over the vocabulary for every audio frame, and the transcript is
recovered by collapsing that frame-by-frame sequence (CTC decoding: repeated
tokens merge into one, then the reserved "blank" token is dropped). There's no
`.generate()` call and no `language`/`task` config to set - a CTC checkpoint
is fine-tuned for one specific language already (see wav2vec2/kiswahili/config.yaml).

One exception to "fine-tuned for one specific language already": adapter-based
multilingual checkpoints like Meta's MMS (`facebook/mms-1b-all`) are still
Wav2Vec2ForCTC underneath, but ship one shared backbone plus a separate small
"adapter" per language - the target language has to be selected before the
first forward pass (`processor.tokenizer.set_target_lang(...)` +
`model.load_adapter(...)`), or it'll transcribe using whatever adapter loaded
by default. See `load_processor`/`load_model` below - `cfg.model_type` (set
via `model.type: mms` in config.yaml) is what switches this on.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCTC, AutoProcessor

from .config import PipelineConfig


def load_processor(cfg: PipelineConfig, model_name: Optional[str] = None) -> AutoProcessor:
    """Load the processor (feature extractor + CTC tokenizer) for a given
    wav2vec2 checkpoint. For an adapter-based checkpoint (model.type: mms),
    also points the tokenizer at the target language's vocabulary - without
    this, decoding would use whichever language's vocab happened to load by
    default, not the one this run is actually testing."""
    name = model_name or cfg.model_name
    processor = AutoProcessor.from_pretrained(name)
    if cfg.model_type == "mms":
        if not cfg.mms_lang_code:
            raise ValueError(
                f"model.type is 'mms' but model.lang_code is not set in this "
                f"language's config.yaml - MMS needs a target language code "
                f"(e.g. 'swh' for Swahili) to select the right adapter."
            )
        processor.tokenizer.set_target_lang(cfg.mms_lang_code)
    return processor


def load_model(cfg: PipelineConfig, model_name: Optional[str] = None) -> AutoModelForCTC:
    """Download (or load from local cache) a wav2vec2/CTC model. For an
    adapter-based checkpoint (model.type: mms), also loads the target
    language's adapter weights on top of the shared backbone."""
    name = model_name or cfg.model_name
    torch_dtype = torch.float16 if cfg.fp16 else torch.float32
    # `dtype=` (not the older `torch_dtype=`, deprecated as of this repo's
    # pinned transformers==5.13.0) - see whisper/lib/model_utils.py's
    # `load_model` for the same dtype-pinning rationale (avoids an
    # fp16/fp32 mismatch crash on CPU).
    model = AutoModelForCTC.from_pretrained(name, dtype=torch_dtype)
    if cfg.model_type == "mms":
        if not cfg.mms_lang_code:
            raise ValueError(
                f"model.type is 'mms' but model.lang_code is not set in this "
                f"language's config.yaml - MMS needs a target language code "
                f"(e.g. 'swh' for Swahili) to select the right adapter."
            )
        model.load_adapter(cfg.mms_lang_code)
    model.to(cfg.device)
    return model


def transcribe(model: AutoModelForCTC, processor: AutoProcessor, audio, sample_rate: int, device: str) -> dict:
    """Run one clip's audio (already loaded as a numpy array, see
    wav2vec2/scripts/run_baseline.py's `_read_wav`) through the model and
    return its transcript plus a confidence proxy.

    Returns a dict with:
      - "hypothesis": the decoded transcript text.
      - "confidence": a 0-1 proxy - the mean max-softmax probability across
        the frames CTC decoding actually kept (i.e. excluding frames whose
        highest-probability class was the CTC blank token, which would
        otherwise inflate the average with confident-but-uninformative
        "nothing here" predictions). Not the same computation as Whisper's
        `sequence_confidence` (whisper/lib/metrics.py) or Gemini's
        `avg_logprobs` proxy (gemini/lib/gemini_client.py) - CTC models don't
        expose the same kind of per-step generation probability - but serves
        the same purpose: a rough "how sure was the model" signal.
    """
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits  # shape: (1, num_frames, vocab_size)

    probs = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(logits, dim=-1)

    hypothesis = processor.batch_decode(predicted_ids)[0]

    blank_id = model.config.pad_token_id  # transformers' CTC convention: pad_token_id doubles as the blank class
    frame_ids = predicted_ids[0]
    frame_confidences = probs[0].max(dim=-1).values
    if blank_id is None:
        # No blank id to exclude on this checkpoint - falling back to
        # `frame_ids != blank_id` here would silently compare against None
        # and be all-True, making the "exclude blank frames" step a no-op
        # without any indication it happened. Averaging over every frame
        # (blanks included) is a known-degraded fallback, not a crash - but
        # explicit about why, rather than a silent difference in behavior.
        confidence = frame_confidences.mean().item()
    else:
        kept = frame_ids != blank_id
        if kept.any():
            confidence = frame_confidences[kept].mean().item()
        else:
            # Every frame decoded to blank - i.e. the model produced no
            # speech at all for this clip. Rather than silently averaging
            # over nothing (which torch.mean would NaN on), report zero
            # confidence explicitly.
            confidence = 0.0

    return {"hypothesis": hypothesis, "confidence": confidence}
