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

# Diarization backends. `community` runs pyannote's open Community-1 pipeline
# locally and is the default: it has no per-minute bill and needs no API key.
# `pyannote` is the hosted pyannoteAI `precision-2` API. They are never
# substituted for one another: switching backend changes cost and quality, so a
# missing setup fails rather than falling back.
DIARIZATION_BACKEND_PYANNOTE = "pyannote"
DIARIZATION_BACKEND_COMMUNITY = "community"
DIARIZATION_BACKENDS = (DIARIZATION_BACKEND_PYANNOTE, DIARIZATION_BACKEND_COMMUNITY)
DEFAULT_DIARIZATION_BACKEND = DIARIZATION_BACKEND_COMMUNITY

COMMUNITY_MODEL_ID = "pyannote/speaker-diarization-community-1"
COMMUNITY_DIARIZATION_MODEL = "community-1"

# What each backend records in the output metadata.
DIARIZATION_MODEL_NAMES = {
    DIARIZATION_BACKEND_PYANNOTE: DIARIZATION_MODEL,
    DIARIZATION_BACKEND_COMMUNITY: COMMUNITY_DIARIZATION_MODEL,
}

# The alignment stage runs out-of-process: whisperx pins torch and declares
# `requires-python <3.14`, so it cannot be installed alongside this project.
DEFAULT_ALIGN_PYTHON = "3.12"
ALIGN_WORKER = PROJECT_ROOT / "workers" / "align_worker.py"

# Prepared by `scripts/setup_alignment.py`. When present it is used directly,
# which avoids re-resolving whisperx on every run. `TRANSCRIBE_ALIGN_VENV`
# relocates it, which is how the container image points at an environment built
# outside the project directory so a bind-mounted checkout cannot shadow it.
ALIGN_VENV = PROJECT_ROOT / ".venv-align"

# Local diarization runs out-of-process for the same reason: pyannote.audio pulls
# in torch, which keeps it off this project's Python version. Its environment is
# separate from the aligner's so neither can perturb the other's torch pin.
DEFAULT_DIARIZE_PYTHON = "3.12"
DIARIZE_WORKER = PROJECT_ROOT / "workers" / "diarize_worker.py"

# Prepared by `scripts/setup_diarization.py`; `TRANSCRIBE_DIARIZE_VENV`
# relocates it, as above.
DIARIZE_VENV = PROJECT_ROOT / ".venv-diarize"


def _venv_python(venv: Path) -> Path | None:
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def _venv_location(variable: str, default: Path) -> Path:
    """Where an out-of-process environment lives, honouring its override.

    Read at call time rather than at import so tests and callers can relocate it
    without reimporting the module.
    """
    override = os.environ.get(variable, "").strip()
    return Path(override).expanduser() if override else default


def align_venv() -> Path:
    return _venv_location("TRANSCRIBE_ALIGN_VENV", ALIGN_VENV)


def diarize_venv() -> Path:
    return _venv_location("TRANSCRIBE_DIARIZE_VENV", DIARIZE_VENV)


def align_venv_python() -> Path | None:
    return _venv_python(align_venv())


def diarize_venv_python() -> Path | None:
    return _venv_python(diarize_venv())


def validate_diarization_backend(value: str | None, *, source: str) -> str:
    """Normalize a backend name, failing clearly rather than guessing."""
    normalized = (value or "").strip().lower()
    if normalized not in DIARIZATION_BACKENDS:
        raise SystemExit(
            f"Unknown diarization backend {value!r} from {source}. "
            f"Choose one of: {', '.join(DIARIZATION_BACKENDS)}."
        )
    return normalized


def resolve_diarization_backend(cli_value: str | None = None) -> str:
    """Pick the backend, letting the CLI override `DIARIZATION_BACKEND`."""
    if cli_value:
        return validate_diarization_backend(cli_value, source="--diarization-backend")

    from_env = os.environ.get("DIARIZATION_BACKEND", "").strip()
    if from_env:
        return validate_diarization_backend(from_env, source="DIARIZATION_BACKEND")

    return DEFAULT_DIARIZATION_BACKEND


def diarization_model_name(backend: str) -> str:
    """The model identifier a backend records in the output metadata."""
    return DIARIZATION_MODEL_NAMES[
        validate_diarization_backend(backend, source="the diarization backend")
    ]


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
