"""Command-line interface for groq-transcribe.

Two ways to run it:

* With arguments: ``groq-transcribe talk.mp3 -l en`` behaves like a normal
  flag-driven tool.
* With no positional argument: it drops into an interactive prompt that asks
  for the audio file and language, then transcribes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .engine import transcribe
from .transcript import format_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groq-transcribe",
        description="Transcribe an MP3/M4A/WAV/OGG file with Groq Whisper.",
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        nargs="?",
        help="Path to the audio file. Omit to run interactively.",
    )
    parser.add_argument(
        "-l",
        "--language",
        help=f"Source language as an ISO-639-1 code (default: {config.DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Transcript output file. Default: <audio-stem>.<language>.txt",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Write timestamped segment lines instead of plain transcript text.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the transcript to stdout instead of writing a file.",
    )
    return parser


def _prompt(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{message}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def _resolve_interactively(args: argparse.Namespace) -> None:
    """Fill in missing audio path / language by prompting the user."""
    if args.audio_path is None:
        while True:
            raw = _prompt("Audio file to transcribe")
            if not raw:
                print("An audio file is required.", file=sys.stderr)
                continue
            candidate = Path(raw).expanduser()
            if candidate.is_file():
                args.audio_path = candidate
                break
            print(f"File not found: {candidate}", file=sys.stderr)
    if not args.language:
        args.language = _prompt("Source language (ISO-639-1)", config.DEFAULT_LANGUAGE)


def _build_transcript(audio_path: Path, language: str, timestamps: bool) -> str:
    result = transcribe(audio_path, language)
    if timestamps:
        return "\n".join(format_segments(result["segments"])) + "\n"
    return result["text"].strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # No audio path given -> interactive mode.
    interactive = args.audio_path is None
    if interactive:
        _resolve_interactively(args)

    language = (args.language or config.DEFAULT_LANGUAGE).strip().lower()
    audio_path = args.audio_path.expanduser().resolve()
    if not audio_path.is_file():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    try:
        text = _build_transcript(audio_path, language, args.timestamps)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1

    if args.stdout:
        sys.stdout.write(text)
        return 0

    output_path = args.output or audio_path.with_name(f"{audio_path.stem}.{language}.txt")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote transcript: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
