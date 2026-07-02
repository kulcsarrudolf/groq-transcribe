"""Groq-hosted Whisper transcription engine (cloud, very fast).

Ported from the igemag-ai project. Sends an audio file to Groq's Whisper API
and returns a normalized :class:`~groq_transcribe.transcript.TranscriptionResult`.
Files larger than Groq's upload limit are transparently re-encoded to a smaller
MP3 with ffmpeg before upload.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from . import config
from .transcript import Segment, TranscriptionResult

logger = logging.getLogger(__name__)


def transcribe(audio_path: str | Path, language: str | None = None) -> TranscriptionResult:
    """Transcribe an audio file via Groq cloud Whisper.

    ``language`` is an ISO-639-1 code (for example ``en``, ``hu``, ``ro``). When
    omitted, :data:`config.DEFAULT_LANGUAGE` is used.
    """
    language = (language or config.DEFAULT_LANGUAGE).strip().lower()
    audio_path = Path(audio_path)
    client = _build_client()
    send_path = _ensure_under_limit(audio_path)
    try:
        response = _call_groq(client, send_path, language)
    finally:
        if send_path != audio_path:
            send_path.unlink(missing_ok=True)
    return _to_result(response)


def _build_client():
    from groq import Groq

    return Groq(api_key=config.get_api_key())


def _call_groq(client, send_path: Path, language: str):
    logger.info(
        "Sending audio to Groq (model=%s, language=%s, size=%.1fMB)",
        config.GROQ_WHISPER_MODEL,
        language,
        send_path.stat().st_size / 1024 / 1024,
    )
    with open(send_path, "rb") as f:
        return client.audio.transcriptions.create(
            file=(send_path.name, f),
            model=config.GROQ_WHISPER_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )


def _to_result(response) -> TranscriptionResult:
    segments: list[Segment] = [
        Segment(start=s["start"], end=s["end"], text=s["text"])
        for s in (response.segments or [])  # type: ignore[attr-defined]
    ]
    return TranscriptionResult(text=response.text, segments=segments)


def _ensure_under_limit(audio_path: Path) -> Path:
    """Return the original path if under the Groq limit, else a compressed MP3."""
    if audio_path.stat().st_size <= config.GROQ_MAX_UPLOAD_BYTES:
        return audio_path
    logger.info(
        "Audio file is %.1fMB, compressing to MP3 for Groq's %dMB limit",
        audio_path.stat().st_size / 1024 / 1024,
        config.GROQ_MAX_UPLOAD_BYTES // (1024 * 1024),
    )
    return _compress_to_mp3(audio_path)


def _compress_to_mp3(audio_path: Path) -> Path:
    # ``mkstemp`` gives a guaranteed-unique path with no lingering file handle.
    fd, tmp_name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ar",
                config.COMPRESS_SAMPLE_RATE,
                "-ac",
                "1",
                "-b:a",
                config.COMPRESS_BITRATE,
                tmp_name,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        Path(tmp_name).unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required to compress audio above the Groq upload limit but was "
            "not found on PATH. Install it (e.g. `brew install ffmpeg`) or provide a "
            "smaller file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        Path(tmp_name).unlink(missing_ok=True)
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to compress audio:\n{stderr}") from exc
    return Path(tmp_name)
