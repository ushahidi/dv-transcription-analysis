"""Tests for gemini/lib/gemini_client.py's retry behavior.

Uses a fake client (no real network/API key needed) and patches time.sleep
so the retry/backoff delays don't actually slow the test down. Covers:
  - a 429 (rate limit) is retried
  - a 5xx ServerError is retried (this is the fix - it used to only catch
    429, and Gemini's real observed failure mode this session was a 503)
  - a non-retryable ClientError (e.g. 400) propagates immediately, on the
    first attempt, with no retry
  - exhausting all retries raises a clear RuntimeError
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.genai.errors import ClientError, ServerError

import gemini.lib.gemini_client as gemini_client


class _FakeResponse:
    """Minimal stand-in for genai.types.GenerateContentResponse - just
    enough surface for gemini_client.transcribe to read `.text` and
    `.candidates[0].avg_logprobs`."""

    def __init__(self, text: str, avg_logprobs=None):
        self.text = text
        candidate = MagicMock()
        candidate.avg_logprobs = avg_logprobs
        self.candidates = [candidate]


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"not real audio, transcribe() never decodes it")
    return p


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retry/backoff sleeps for real seconds - patch them away so these
    tests run instantly regardless of gemini_client's actual delay values."""
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _seconds: None)


def _client_raising_then_succeeding(*exceptions, final_response):
    """Build a fake client whose generate_content raises each of
    `exceptions` in turn, then returns `final_response` on the call after."""
    client = MagicMock()
    side_effects = list(exceptions) + [final_response]
    client.models.generate_content.side_effect = side_effects
    return client


def test_retries_429_then_succeeds(audio_path):
    client = _client_raising_then_succeeding(
        ClientError(429, {"error": {"message": "rate limited"}}),
        ClientError(429, {"error": {"message": "rate limited"}}),
        final_response=_FakeResponse("hello world"),
    )
    result = gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash")
    assert result["hypothesis"] == "hello world"
    assert client.models.generate_content.call_count == 3


def test_retries_5xx_server_error_then_succeeds(audio_path):
    # This is the actual bug this test guards against: the retry loop used
    # to only catch ClientError (429), so a ServerError (5xx) - Gemini's
    # real observed "high demand" 503 - would propagate on the very first
    # attempt instead of being retried.
    client = _client_raising_then_succeeding(
        ServerError(503, {"error": {"message": "high demand"}}),
        final_response=_FakeResponse("hello world"),
    )
    result = gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash")
    assert result["hypothesis"] == "hello world"
    assert client.models.generate_content.call_count == 2


def test_non_retryable_client_error_propagates_immediately(audio_path):
    client = MagicMock()
    client.models.generate_content.side_effect = ClientError(400, {"error": {"message": "bad request"}})
    with pytest.raises(ClientError):
        gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash")
    # Only one attempt - a 400 is not something retrying will ever fix.
    assert client.models.generate_content.call_count == 1


def test_exhausting_all_retries_raises_runtime_error(audio_path):
    client = MagicMock()
    client.models.generate_content.side_effect = ServerError(503, {"error": {"message": "still down"}})
    with pytest.raises(RuntimeError):
        gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash", max_retries=3)
    assert client.models.generate_content.call_count == 3


def test_confidence_falls_back_to_none_without_avg_logprobs(audio_path):
    client = _client_raising_then_succeeding(final_response=_FakeResponse("hi"))
    result = gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash")
    assert result["confidence"] is None


def test_confidence_derived_from_avg_logprobs_when_present(audio_path):
    import math

    client = _client_raising_then_succeeding(final_response=_FakeResponse("hi", avg_logprobs=-0.1))
    result = gemini_client.transcribe(client, audio_path, "swahili", "gemini-2.5-flash")
    assert result["confidence"] == pytest.approx(math.exp(-0.1))
