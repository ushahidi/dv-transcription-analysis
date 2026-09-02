"""Gemini Developer API wrapper used by gemini/scripts/run_baseline.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
Whisper is a model we load and run ourselves; Gemini is a model we call over
the network. This file is the one place that knows how to send a single audio
clip to Gemini and get back a transcript - everything else in this folder
(the CLI script, the manifest loading, the WER scoring) stays the same
regardless of which engine produced the transcript.

Auth: reads the API key from the GEMINI_API_KEY environment variable (a free
key from https://aistudio.google.com/apikey - no Google Cloud project or
billing account needed, unlike chirp/). `google.genai.Client()` picks this env
var up automatically; we still check for it explicitly first so a missing key
fails with a clear message instead of a confusing SDK error.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

# The free Developer API tier is rate-limited (roughly 5-15 requests/minute
# depending on the model - see https://ai.google.dev/gemini-api/docs/rate-limits).
# A 20-clip test set sent back-to-back will trip that, so every call is spaced
# out by at least this many seconds, on top of the retry/backoff below for
# whenever a 429 slips through anyway.
MIN_SECONDS_BETWEEN_CALLS = 4.5

# How the model is told what to do. Deliberately blunt ("ONLY the transcript")
# because instruction-tuned models left to their own devices will sometimes
# add commentary ("Here is the transcript:") or translate instead of
# transcribing - either would silently wreck the WER score.
_PROMPT_TEMPLATE = (
    "Transcribe this audio verbatim in {language_name}. "
    "Output ONLY the exact words spoken, with no translation, no commentary, "
    "and no formatting - just the transcript text."
)


def make_client() -> genai.Client:
    """Build the Gemini client, failing fast with a clear error if
    GEMINI_API_KEY isn't set (rather than letting the SDK raise something
    more cryptic later, on the first real call)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key from "
            "https://aistudio.google.com/apikey and set it, e.g. "
            "(PowerShell) $env:GEMINI_API_KEY = '...'"
        )
    return genai.Client(api_key=api_key)


def transcribe(
    client: genai.Client,
    audio_path: Path,
    language_name: str,
    model_name: str,
    max_retries: int = 5,
) -> dict:
    """Send one .wav clip to Gemini and return its transcript.

    Returns a dict with:
      - "hypothesis": the transcript text Gemini produced.
      - "confidence": a 0-1 proxy derived from `avg_logprobs` when the model
        response includes it, else None. Unlike Whisper's `sequence_confidence`
        (whisper/lib/metrics.py) or Cloud STT's native per-result confidence
        (chirp/lib/stt_client.py), Gemini's generateContent API does not
        reliably expose token-level probabilities for every model, so this is
        best-effort - a missing confidence is expected, not a bug.
    """
    audio_bytes = audio_path.read_bytes()
    prompt = _PROMPT_TEMPLATE.format(language_name=language_name)

    delay = MIN_SECONDS_BETWEEN_CALLS
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        time.sleep(delay if attempt == 0 else delay * (2**attempt))
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                ],
            )
            break
        except (ClientError, ServerError) as exc:
            # 429 (rate limit) and any 5xx (Gemini reporting itself
            # overloaded/unavailable - "high demand", a real, observed failure
            # mode on the free tier) are both transient - worth a backoff and
            # retry. Any other ClientError (bad key, bad request, ...) isn't
            # something a retry will fix, so re-raise immediately.
            retryable = getattr(exc, "code", None) == 429 or isinstance(exc, ServerError)
            if not retryable:
                raise
            last_error = exc
    else:
        raise RuntimeError(
            f"Gemini stayed rate-limited/unavailable for {max_retries} retries in a row "
            f"transcribing {audio_path.name}"
        ) from last_error

    hypothesis = (response.text or "").strip()

    confidence = None
    candidates = getattr(response, "candidates", None)
    if candidates:
        avg_logprobs = getattr(candidates[0], "avg_logprobs", None)
        if avg_logprobs is not None:
            import math

            confidence = math.exp(avg_logprobs)

    return {"hypothesis": hypothesis, "confidence": confidence}
