"""Tests for the FastAPI web UI (transcription mocked, no network)."""

from __future__ import annotations

import io

import pytest
from starlette.testclient import TestClient

from groq_transcribe import web

_RESULT = {
    "text": "hello world",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
}


@pytest.fixture
def client():
    return TestClient(web.app)


def test_index_renders_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'action="/transcribe"' in resp.text


def test_transcribe_renders_transcript(mocker, client):
    mocker.patch.object(web, "transcribe", return_value=_RESULT)
    files = {"audio": ("clip.mp3", io.BytesIO(b"fake"), "audio/mpeg")}

    resp = client.post("/transcribe", files=files, data={"language": "en"})

    assert resp.status_code == 200
    assert "hello world" in resp.text
    assert "clip.mp3" in resp.text


def test_transcribe_with_timestamps(mocker, client):
    mocker.patch.object(web, "transcribe", return_value=_RESULT)
    files = {"audio": ("clip.mp3", io.BytesIO(b"fake"), "audio/mpeg")}

    resp = client.post(
        "/transcribe", files=files, data={"language": "en", "timestamps": "true"}
    )

    assert resp.status_code == 200
    assert "[0.00s - 1.00s] hello world" in resp.text


def test_transcribe_shows_error(mocker, client):
    mocker.patch.object(web, "transcribe", side_effect=ValueError("GROQ_API_KEY is not set"))
    files = {"audio": ("clip.mp3", io.BytesIO(b"fake"), "audio/mpeg")}

    resp = client.post("/transcribe", files=files, data={"language": "en"})

    assert resp.status_code == 200
    assert "GROQ_API_KEY is not set" in resp.text
