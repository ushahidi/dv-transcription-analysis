"""Cloud Speech-to-Text v2 (Chirp) wrapper used by chirp/scripts/run_baseline.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
The Chirp equivalent of gemini/lib/gemini_client.py - the one place that knows
how to send a single audio clip to Google Cloud Speech-to-Text and get back a
transcript. Everything else in this folder (the CLI script, manifest loading,
WER scoring) stays the same regardless of which engine produced the transcript.

Status: blocked on GCP setup + ADC, not on unfinished code (see chirp/README.md).
This file is complete and ready to run - what's actually missing is
account-level setup nobody else can do on your behalf: a GCP project with
billing enabled, the Speech-to-Text API turned on, and a credential (ADC).
It can't be exercised end-to-end from here until that setup exists on your
account.

Auth: uses Application Default Credentials, the SDK's standard mechanism - set
the GOOGLE_APPLICATION_CREDENTIALS environment variable to a service-account
JSON key path, or run `gcloud auth application-default login`. The project id
comes from the GOOGLE_CLOUD_PROJECT environment variable (deliberately not
checked into config.yaml - it's account-specific).

Uses the *implicit* default recognizer (`.../recognizers/_`) rather than
creating a named Recognizer resource up front - the model and language are
passed inline on every request instead, so there's nothing to provision before
the first call beyond the project/API being enabled.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.cloud import speech_v2

# Transient errors worth retrying: 429 (rate limit) plus the 5xx/deadline
# errors Cloud STT routinely raises under load. None of the latter subclass
# ResourceExhausted, so each needs to be named explicitly - same rationale as
# gemini/lib/gemini_client.py retrying ServerError (any 5xx) alongside 429.
_RETRYABLE_ERRORS = (ResourceExhausted, ServiceUnavailable, InternalServerError, DeadlineExceeded)

# Same rationale as gemini/lib/gemini_client.py's MIN_SECONDS_BETWEEN_CALLS:
# space calls out proactively so a 20-clip run doesn't lean entirely on
# retry/backoff to survive Cloud STT's default per-minute quota.
MIN_SECONDS_BETWEEN_CALLS = 1.0


def make_client(location: str) -> speech_v2.SpeechClient:
    """Build the Cloud Speech-to-Text v2 client. `location` picks a regional
    endpoint (e.g. "us-central1") rather than the global one, matching where
    chirp/<language>/config.yaml's gcp.location says Chirp is being called
    from - Chirp model availability is region-specific (see
    chirp/README.md)."""
    client_options = {"api_endpoint": f"{location}-speech.googleapis.com"}
    return speech_v2.SpeechClient(client_options=client_options)


def project_id() -> str:
    """Read the GCP project id from GOOGLE_CLOUD_PROJECT, failing fast with a
    clear error if it's unset (same pattern as gemini_client.make_client's
    GEMINI_API_KEY check)."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Set it to your GCP project id, e.g. "
            "(PowerShell) $env:GOOGLE_CLOUD_PROJECT = 'your-project-id'. "
            "See chirp/README.md for the full GCP setup this engine needs."
        )
    return project


def transcribe(
    client: speech_v2.SpeechClient,
    audio_path: Path,
    project: str,
    location: str,
    model: str,
    language_codes: List[str],
    max_retries: int = 5,
) -> dict:
    """Send one .wav clip to Cloud Speech-to-Text (Chirp) and return its
    transcript.

    Returns a dict with:
      - "hypothesis": the transcript text (the top alternative of the first
        result; empty string if Chirp returned no speech).
      - "confidence": always None. Cloud STT v2's Chirp models do not
        reliably populate `SpeechRecognitionAlternative.confidence` for
        plain `recognize()` calls - it comes back as a constant 0.0, which
        would read as a real (terrible) confidence signal if surfaced as-is.
        Rather than emit a misleading number, this is left unset here;
        `chirp/scripts/run_baseline.py` skips it the same way
        `gemini/scripts/run_baseline.py` skips Gemini's occasionally-missing
        `avg_logprobs`.
    """
    audio_bytes = audio_path.read_bytes()
    recognizer = f"projects/{project}/locations/{location}/recognizers/_"
    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model=model,
    )

    delay = MIN_SECONDS_BETWEEN_CALLS
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        time.sleep(delay if attempt == 0 else delay * (2**attempt))
        try:
            response = client.recognize(
                recognizer=recognizer, config=config, content=audio_bytes
            )
            break
        except _RETRYABLE_ERRORS as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"Cloud STT stayed rate-limited/unavailable for {max_retries} retries in a "
            f"row transcribing {audio_path.name}"
        ) from last_error

    # Cloud STT v2 returns a SEQUENCE of results, not one - typically one per
    # detected speech segment within the clip. Taking only results[0] would
    # silently truncate the hypothesis to just the first segment for any
    # clip with more than one, understating accuracy for reasons that have
    # nothing to do with the model - join every result's top alternative
    # instead.
    hypothesis = " ".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    ).strip()
    return {"hypothesis": hypothesis, "confidence": None}
