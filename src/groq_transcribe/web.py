"""A minimal local web UI for groq-transcribe.

A single page: upload an audio file, pick a language, get the transcript back.
Runs locally on 127.0.0.1 with ``groq-transcribe-web``. No persistence, no
external services beyond the Groq API call itself.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import config
from .engine import transcribe
from .transcript import format_segments

_TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="groq-transcribe")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"default_language": config.DEFAULT_LANGUAGE},
    )


@app.post("/transcribe", response_class=HTMLResponse)
def do_transcribe(
    request: Request,
    audio: UploadFile,
    language: str = Form(""),
    language_other: str = Form(""),
    timestamps: bool = Form(False),
) -> HTMLResponse:
    if language.strip().lower() == "other":
        language = language_other
    language = (language or config.DEFAULT_LANGUAGE).strip().lower()
    context: dict = {"default_language": language, "filename": audio.filename}

    tmp_path = _save_upload(audio)
    try:
        result = transcribe(tmp_path, language)
        context["transcript"] = (
            "\n".join(format_segments(result["segments"])) if timestamps else result["text"].strip()
        )
    except Exception as exc:  # surfaced to the user, not the server log alone
        context["error"] = str(exc)
    finally:
        tmp_path.unlink(missing_ok=True)

    return templates.TemplateResponse(request, "index.html", context)


def _save_upload(audio: UploadFile) -> Path:
    """Stream an uploaded file to a temp path, preserving its suffix."""
    suffix = Path(audio.filename or "").suffix or ".bin"
    fd, name = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        while chunk := audio.file.read(1024 * 1024):
            out.write(chunk)
    return Path(name)


def main() -> None:
    """Entry point for the ``groq-transcribe-web`` console script."""
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
