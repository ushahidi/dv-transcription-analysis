"""Tests for chirp/lib/stt_client.py's retry behavior and result-joining.

Uses a fake client (no real GCP project/credentials needed) and patches
time.sleep so retry/backoff delays don't slow the test down. Covers:
  - each of the four transient errors (429/503/500/504) is retried
  - a non-transient error is NOT retried and propagates on the first attempt
  - exhausting all retries raises a clear RuntimeError
  - multiple SpeechRecognitionResults in one response are joined into one
    hypothesis, not truncated to just the first (the actual bug this
    guards against - Cloud STT v2 returns a SEQUENCE of results, one per
    detected speech segment, and taking only results[0] would silently
    drop every segment after the first)
  - an empty results list produces an empty ("") hypothesis, not a crash
  - confidence is always None (Chirp doesn't reliably populate it - see
    the module docstring in chirp/lib/stt_client.py)
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)

import chirp.lib.stt_client as stt_client


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"not real audio, transcribe() never decodes it")
    return p


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(stt_client.time, "sleep", lambda _seconds: None)


def _fake_result(transcript: str):
    alternative = MagicMock()
    alternative.transcript = transcript
    result = MagicMock()
    result.alternatives = [alternative]
    return result


def _fake_response(*transcripts: str):
    response = MagicMock()
    response.results = [_fake_result(t) for t in transcripts]
    return response


@pytest.mark.parametrize(
    "exc",
    [
        ResourceExhausted("rate limited"),
        ServiceUnavailable("high demand"),
        InternalServerError("internal error"),
        DeadlineExceeded("deadline exceeded"),
    ],
    ids=["429_ResourceExhausted", "503_ServiceUnavailable", "500_InternalServerError", "504_DeadlineExceeded"],
)
def test_retries_each_transient_error_then_succeeds(audio_path, exc):
    client = MagicMock()
    client.recognize.side_effect = [exc, _fake_response("hello world")]
    result = stt_client.transcribe(client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"])
    assert result["hypothesis"] == "hello world"
    assert client.recognize.call_count == 2


def test_non_transient_error_propagates_immediately(audio_path):
    client = MagicMock()
    client.recognize.side_effect = ValueError("something unrelated broke")
    with pytest.raises(ValueError):
        stt_client.transcribe(client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"])
    assert client.recognize.call_count == 1


def test_exhausting_all_retries_raises_runtime_error(audio_path):
    client = MagicMock()
    client.recognize.side_effect = ServiceUnavailable("still down")
    with pytest.raises(RuntimeError):
        stt_client.transcribe(
            client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"], max_retries=3
        )
    assert client.recognize.call_count == 3


def test_multiple_results_are_joined_not_truncated(audio_path):
    client = MagicMock()
    client.recognize.return_value = _fake_response("first segment", "second segment", "third segment")
    result = stt_client.transcribe(client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"])
    assert result["hypothesis"] == "first segment second segment third segment"


def test_empty_results_gives_empty_hypothesis(audio_path):
    client = MagicMock()
    client.recognize.return_value = _fake_response()  # no results at all
    result = stt_client.transcribe(client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"])
    assert result["hypothesis"] == ""


def test_confidence_is_always_none(audio_path):
    client = MagicMock()
    client.recognize.return_value = _fake_response("hello")
    result = stt_client.transcribe(client, audio_path, "my-project", "us-central1", "chirp_3", ["sw-KE"])
    assert result["confidence"] is None
