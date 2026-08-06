"""Environment loading and pipeline-wide constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Repository root: .../transcribe_pipeline/config.py -> .../
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# Hosted models. `gpt-transcribe` is OpenAI's recommended model for transcribing
# recorded speech; it produces no timestamps, which is why alignment is a
# separate stage. Cleanup only rewrites punctuation, so it does not need the
# flagship tier.
TRANSCRIBE_MODEL = "gpt-transcribe"
CLEANUP_MODEL = "gpt-5.6-terra"
DIARIZATION_MODEL = "precision-2"

PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"

# The alignment stage runs out-of-process: whisperx pins torch and declares
# `requires-python <3.14`, so it cannot be installed alongside this project.
DEFAULT_ALIGN_PYTHON = "3.12"
ALIGN_WORKER = PROJECT_ROOT / "workers" / "align_worker.py"

# Prepared by `scripts/setup_alignment.py`. When present it is used directly,
# which avoids re-resolving whisperx on every run.
ALIGN_VENV = PROJECT_ROOT / ".venv-align"


def align_venv_python() -> Path | None:
    for candidate in (ALIGN_VENV / "bin" / "python", ALIGN_VENV / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def load_environment() -> None:
    """Load `.env` so the script works without `uv run --env-file`.

    Values already present in the real environment always win, which keeps
    `OPENAI_API_KEY=... python transcribe.py ...` working as an override.
    """
    # Walking up from the CWD handles being invoked from a subdirectory; the
    # project root is loaded too so the script works from anywhere.
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered)
    load_dotenv(PROJECT_ROOT / ".env")


def require_env(name: str, *, purpose: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. It is required for {purpose}.\n"
            f"Add it to {PROJECT_ROOT / '.env'} or export it in your shell."
        )
    return value
