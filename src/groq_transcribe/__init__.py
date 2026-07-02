"""groq-transcribe: audio-to-transcript via Groq-hosted Whisper.

Public API::

    from groq_transcribe import transcribe, format_segments

    result = transcribe("talk.mp3", language="en")
    print(result["text"])
    for line in format_segments(result["segments"]):
        print(line)
"""

from __future__ import annotations

from .engine import transcribe
from .transcript import Segment, TranscriptionResult, format_segments

__version__ = "0.1.0"

__all__ = [
    "transcribe",
    "format_segments",
    "Segment",
    "TranscriptionResult",
    "__version__",
]
