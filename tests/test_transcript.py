"""Tests for transcript formatting helpers."""

from __future__ import annotations

from groq_transcribe.transcript import format_segments


def test_format_segments_basic():
    segments = [
        {"start": 0.0, "end": 1.25, "text": "hello"},
        {"start": 1.25, "end": 2.5, "text": "world"},
    ]
    assert format_segments(segments) == [
        "[0.00s - 1.25s] hello",
        "[1.25s - 2.50s] world",
    ]


def test_format_segments_applies_offset():
    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    assert format_segments(segments, offset=10.0) == ["[10.00s - 11.00s] hi"]


def test_format_segments_empty():
    assert format_segments([]) == []
