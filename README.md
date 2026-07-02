# groq-transcribe

Turn an audio file into a text transcript using [Groq](https://groq.com)-hosted
Whisper. Fast, local, and simple: an interactive CLI plus a small web UI.

- **Provider:** Groq Whisper (`whisper-large-v3` by default)
- **Runs locally**, configured entirely through environment variables
- **Auto-handles large files** by re-encoding them with ffmpeg before upload

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Groq API key (create one at https://console.groq.com/keys)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (only needed for audio files
  larger than Groq's 25 MB upload limit, which are compressed automatically)

## Setup

```bash
git clone https://github.com/kulcsarrudolf/groq-transcribe.git
cd groq-transcribe
uv sync

cp .env.example .env
# then edit .env and set GROQ_API_KEY
```

## Configuration

All configuration comes from environment variables. Only `GROQ_API_KEY` is
required; a local `.env` file (git-ignored) is loaded automatically.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | yes | | Your Groq API key. |
| `GROQ_WHISPER_MODEL` | no | `whisper-large-v3` | Whisper model to use. |
| `TRANSCRIBE_DEFAULT_LANGUAGE` | no | `en` | Default source language (ISO-639-1). |
| `GROQ_MAX_UPLOAD_BYTES` | no | `26214400` | Upload limit; larger files are auto-compressed. |
| `COMPRESS_SAMPLE_RATE` | no | `16000` | ffmpeg sample rate when compressing. |
| `COMPRESS_BITRATE` | no | `48k` | ffmpeg audio bitrate when compressing. |

## Command-line usage

Transcribe a file directly:

```bash
uv run groq-transcribe talk.mp3 -l en
# -> writes talk.en.txt
```

Run with no file to go interactive (it prompts for the file and language):

```bash
uv run groq-transcribe
```

Options:

| Flag | Description |
| --- | --- |
| `audio_path` | Path to the audio file. Omit to run interactively. |
| `-l`, `--language` | Source language (ISO-639-1). Defaults to `TRANSCRIBE_DEFAULT_LANGUAGE`. |
| `-o`, `--output` | Output file. Default: `<audio-stem>.<language>.txt`. |
| `--timestamps` | Emit `[start - end]` segment lines instead of plain text. |
| `--stdout` | Print the transcript instead of writing a file. |

## Web UI

Start the local web app and open it in your browser:

```bash
uv run groq-transcribe-web
# then open http://127.0.0.1:8000
```

Upload an audio file, pick a language, and get the transcript back with copy
and download buttons. It binds to `127.0.0.1` only; override with the `HOST`
and `PORT` environment variables if needed.

## Usage as a library

```python
from groq_transcribe import transcribe, format_segments

result = transcribe("talk.mp3", language="en")

print(result["text"])                       # full transcript
for line in format_segments(result["segments"]):
    print(line)                             # [0.00s - 5.20s] ...
```

## License

MIT. See [LICENSE](LICENSE).
