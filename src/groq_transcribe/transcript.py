"""Transcript data types and formatting helpers."""

from __future__ import annotations

from typing import TypedDict


class Segment(TypedDict):
    """A single timestamped chunk of transcribed audio."""

    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict):
    """Full transcript: the joined text plus its individual segments."""

    text: str
    segments: list[Segment]


def format_segments(segments: list[Segment], offset: float = 0.0) -> list[str]:
    """Format segments into human-readable timestamped lines.

    Returns a list of strings like ``[0.00s - 5.20s] Hello world``. ``offset``
    shifts every timestamp, useful when the audio was trimmed from a longer file.
    """
    lines: list[str] = []
    for seg in segments:
        start = seg["start"] + offset
        end = seg["end"] + offset
        lines.append(f"[{start:.2f}s - {end:.2f}s] {seg['text']}")
    return lines
