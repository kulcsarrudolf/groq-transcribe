"""Runtime configuration, loaded from environment variables (and a local ``.env``).

Everything the tool needs is read from the environment so it can run purely
locally with no config files beyond an optional ``.env``. Only ``GROQ_API_KEY``
is required; the rest have sensible defaults.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load a local ``.env`` if present. ``override=False`` means real environment
# variables win over the file, which is the least-surprising precedence.
load_dotenv(override=False)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


# Groq Whisper model. ``whisper-large-v3`` is the highest-quality option.
GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Groq's audio upload hard limit is 25 MB. Files above this are transparently
# re-encoded to a smaller MP3 before upload (see ``engine.py``).
GROQ_MAX_UPLOAD_BYTES: int = _int_env("GROQ_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)

# ffmpeg re-encode target when shrinking oversized audio.
COMPRESS_SAMPLE_RATE: str = os.getenv("COMPRESS_SAMPLE_RATE", "16000")
COMPRESS_BITRATE: str = os.getenv("COMPRESS_BITRATE", "48k")

# Default source language (ISO-639-1) used when the caller doesn't specify one.
DEFAULT_LANGUAGE: str = os.getenv("TRANSCRIBE_DEFAULT_LANGUAGE", "en")


def get_api_key() -> str:
    """Return the Groq API key, raising a clear error if it isn't configured."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your environment or a local .env file. "
            "Get a key at https://console.groq.com/keys"
        )
    return key
