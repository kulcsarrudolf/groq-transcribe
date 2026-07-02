"""Tests for the Groq transcription engine (no network calls)."""

from __future__ import annotations

import subprocess

import pytest

from groq_transcribe import config, engine


class _FakeResponse:
    def __init__(self, text: str, segments: list[dict]):
        self.text = text
        self.segments = segments


class _FakeTranscriptions:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self.audio = type("Audio", (), {"transcriptions": _FakeTranscriptions(response)})()


def test_transcribe_maps_response_to_result(mocker, tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"tiny")
    response = _FakeResponse(
        text="hello world",
        segments=[{"start": 0.0, "end": 1.5, "text": "hello world"}],
    )
    mocker.patch.object(engine, "_build_client", return_value=_FakeClient(response))

    result = engine.transcribe(audio, "en")

    assert result["text"] == "hello world"
    assert result["segments"] == [{"start": 0.0, "end": 1.5, "text": "hello world"}]


def test_transcribe_passes_model_and_language(mocker, tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"tiny")
    fake = _FakeClient(_FakeResponse("hi", []))
    mocker.patch.object(engine, "_build_client", return_value=fake)

    engine.transcribe(audio, "HU")

    call = fake.audio.transcriptions.calls[0]
    assert call["model"] == config.GROQ_WHISPER_MODEL
    assert call["language"] == "hu"  # normalized to lowercase
    assert call["response_format"] == "verbose_json"


def test_transcribe_defaults_language(mocker, tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"tiny")
    fake = _FakeClient(_FakeResponse("hi", []))
    mocker.patch.object(engine, "_build_client", return_value=fake)

    engine.transcribe(audio, None)

    assert fake.audio.transcriptions.calls[0]["language"] == config.DEFAULT_LANGUAGE


def test_ensure_under_limit_returns_original_when_small(tmp_path):
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"x" * 10)
    assert engine._ensure_under_limit(audio) == audio


def test_ensure_under_limit_compresses_when_large(mocker, tmp_path):
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    mocker.patch.object(config, "GROQ_MAX_UPLOAD_BYTES", 10)
    sentinel = tmp_path / "compressed.mp3"
    compress = mocker.patch.object(engine, "_compress_to_mp3", return_value=sentinel)

    assert engine._ensure_under_limit(audio) == sentinel
    compress.assert_called_once_with(audio)


def test_compress_to_mp3_builds_ffmpeg_command(mocker, tmp_path):
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    run = mocker.patch.object(engine.subprocess, "run")

    out = engine._compress_to_mp3(audio)

    assert out.suffix == ".mp3"
    cmd = run.call_args.args[0]
    assert cmd[0] == "ffmpeg"
    assert config.COMPRESS_SAMPLE_RATE in cmd
    assert config.COMPRESS_BITRATE in cmd
    out.unlink(missing_ok=True)


def test_compress_to_mp3_missing_ffmpeg_raises_runtime_error(mocker, tmp_path):
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    mocker.patch.object(engine.subprocess, "run", side_effect=FileNotFoundError())

    with pytest.raises(RuntimeError, match="ffmpeg"):
        engine._compress_to_mp3(audio)


def test_compress_to_mp3_ffmpeg_failure_raises_runtime_error(mocker, tmp_path):
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    err = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"boom")
    mocker.patch.object(engine.subprocess, "run", side_effect=err)

    with pytest.raises(RuntimeError, match="boom"):
        engine._compress_to_mp3(audio)


def test_compressed_temp_file_is_cleaned_up(mocker, tmp_path):
    """When a compressed file is sent, it must be deleted afterwards."""
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    compressed = tmp_path / "tmp-compressed.mp3"
    compressed.write_bytes(b"small")
    mocker.patch.object(engine, "_ensure_under_limit", return_value=compressed)
    mocker.patch.object(engine, "_build_client", return_value=_FakeClient(_FakeResponse("hi", [])))

    engine.transcribe(audio, "en")

    assert not compressed.exists()


def test_get_api_key_requires_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        config.get_api_key()


def test_get_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  secret-key  ")
    assert config.get_api_key() == "secret-key"
