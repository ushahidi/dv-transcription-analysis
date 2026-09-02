"""Cloud Speech-to-Text v2 (Chirp) wrapper used by chirp/scripts/run_baseline.py.

WHAT THIS FILE DOES, IN PLAIN TERMS:
The Chirp equivalent of gemini/lib/gemini_client.py - the one place that knows
how to send a single audio clip to Google Cloud Speech-to-Text and get back a
transcript. Everything else in this folder (the CLI script, manifest loading,
WER scoring) stays the same regardless of which engine produced the transcript.

NOT RUNNABLE YET: this needs a GCP project with billing enabled, the
Speech-to-Text API turned on, and a credential - none of which exist yet (see
chirp/README.md). This file is written and structured so it's ready the moment
that setup is done; it can't be exercised end-to-end from here in the meantime.

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

from google.api_core.exceptions import ResourceExhausted
from google.cloud import speech_v2

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
      - "confidence": Cloud STT v2's own per-alternative confidence (0-1),
        native to the API response - unlike gemini/lib/gemini_client.py's
        best-effort proxy, this is always present when a result is returned.
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
        except ResourceExhausted as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"Cloud STT rate-limited {max_retries} times in a row transcribing "
            f"{audio_path.name}"
        ) from last_error

    if not response.results or not response.results[0].alternatives:
        return {"hypothesis": "", "confidence": 0.0}

    top = response.results[0].alternatives[0]
    return {"hypothesis": top.transcript, "confidence": top.confidence}
