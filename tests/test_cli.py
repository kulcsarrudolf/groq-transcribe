"""Tests for the command-line interface (transcription mocked, no network)."""

from __future__ import annotations

import pytest

from groq_transcribe import cli

_RESULT = {
    "text": "  hello world  ",
    "segments": [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ],
}


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"fake-audio")
    return path


def test_parser_defaults():
    args = cli.build_parser().parse_args(["clip.mp3"])
    assert str(args.audio_path) == "clip.mp3"
    assert args.language is None
    assert args.timestamps is False


def test_main_writes_plain_transcript(mocker, audio_file):
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)

    rc = cli.main([str(audio_file), "-l", "en"])

    assert rc == 0
    out = audio_file.with_name("clip.en.txt")
    assert out.read_text(encoding="utf-8") == "hello world\n"


def test_main_timestamps(mocker, audio_file):
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)

    rc = cli.main([str(audio_file), "-l", "en", "--timestamps"])

    assert rc == 0
    out = audio_file.with_name("clip.en.txt").read_text(encoding="utf-8")
    assert out == "[0.00s - 1.00s] hello\n[1.00s - 2.00s] world\n"


def test_main_custom_output(mocker, audio_file, tmp_path):
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)
    dest = tmp_path / "out" / "result.txt"

    rc = cli.main([str(audio_file), "-o", str(dest)])

    assert rc == 0
    assert dest.read_text(encoding="utf-8") == "hello world\n"


def test_main_stdout(mocker, audio_file, capsys):
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)

    rc = cli.main([str(audio_file), "--stdout"])

    assert rc == 0
    assert capsys.readouterr().out == "hello world\n"


def test_main_missing_file_returns_2(tmp_path):
    rc = cli.main([str(tmp_path / "does-not-exist.mp3")])
    assert rc == 2


def test_main_transcription_error_returns_1(mocker, audio_file):
    mocker.patch.object(cli, "transcribe", side_effect=RuntimeError("nope"))
    rc = cli.main([str(audio_file)])
    assert rc == 1


def test_interactive_fallback(mocker, audio_file):
    """With no positional arg, prompt for the file and language."""
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)
    responses = iter([str(audio_file), "ro"])
    mocker.patch("builtins.input", lambda _prompt: next(responses))

    rc = cli.main([])

    assert rc == 0
    assert audio_file.with_name("clip.ro.txt").exists()


def test_interactive_reprompts_on_missing_file(mocker, audio_file):
    mocker.patch.object(cli, "transcribe", return_value=_RESULT)
    responses = iter(["/no/such/file.mp3", str(audio_file), "en"])
    mocker.patch("builtins.input", lambda _prompt: next(responses))

    rc = cli.main([])

    assert rc == 0
