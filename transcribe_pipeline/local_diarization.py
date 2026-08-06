"""Drives the isolated local pyannote Community-1 diarization worker.

pyannote.audio pulls in torch and so cannot be installed next to this project,
exactly like WhisperX. It runs as a subprocess in a uv-managed Python 3.12
environment, and communication is over temporary JSON files rather than pipes
because the worker's dependencies write to stdout on some paths.

This backend is never substituted for the hosted one, or vice versa: the two
differ in cost and quality, so a missing local setup raises rather than quietly
falling back to the API.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transcribe_pipeline.config import (
    COMMUNITY_MODEL_ID,
    DEFAULT_DIARIZE_PYTHON,
    DIARIZE_WORKER,
    diarize_venv_python,
)

# The worker emits the same turn shape the pyannoteAI API does, which is what
# lets one parser serve both backends.
from transcribe_pipeline.diarization import parse_turns
from transcribe_pipeline.models import DiarizationResult


class LocalDiarizationError(RuntimeError):
    pass


def default_command() -> list[str]:
    """Build the command that runs the worker in an isolated 3.12 environment."""
    override = os.environ.get("TRANSCRIBE_DIARIZE_CMD", "").strip()
    if override:
        return shlex.split(override)

    prepared = diarize_venv_python()
    if prepared is not None:
        return [str(prepared), str(DIARIZE_WORKER)]

    if not shutil.which("uv"):
        raise LocalDiarizationError(
            "uv is required to run the local diarization worker in an isolated "
            f"Python {DEFAULT_DIARIZE_PYTHON} environment. Install uv and run "
            "scripts/setup_diarization.py, or set TRANSCRIBE_DIARIZE_CMD to a "
            "command that runs workers/diarize_worker.py with pyannote.audio "
            "available."
        )

    # --no-project keeps uv from resolving against this project, whose Python
    # requirement would make pyannote.audio uninstallable.
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        DEFAULT_DIARIZE_PYTHON,
        "--with",
        "pyannote.audio",
        str(DIARIZE_WORKER),
    ]


def parse_worker_output(payload: dict) -> DiarizationResult:
    return DiarizationResult(
        turns=parse_turns(payload.get("diarization")),
        exclusive_turns=parse_turns(payload.get("exclusive_diarization")),
    )


def diarize(
    source: Path,
    *,
    num_speakers: int | None = None,
    device: str | None = None,
) -> DiarizationResult:
    if not DIARIZE_WORKER.is_file():
        raise LocalDiarizationError(f"Diarization worker not found at {DIARIZE_WORKER}")

    payload = {
        "audio": str(source),
        "model": COMMUNITY_MODEL_ID,
        "num_speakers": num_speakers,
        # Left absent unless explicitly chosen, so the worker's CPU default holds.
        "device": device,
    }

    command = default_command()
    if diarize_venv_python() is None and not os.environ.get("TRANSCRIBE_DIARIZE_CMD", "").strip():
        print(
            "No prepared diarization environment found; building one on demand. "
            "Run scripts/setup_diarization.py once to avoid this.",
            file=sys.stderr,
        )
    print(f"Diarizing locally with {COMMUNITY_MODEL_ID}...", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="diarize-") as workspace:
        request_path = Path(workspace) / "request.json"
        response_path = Path(workspace) / "response.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")

        completed = subprocess.run(
            [*command, str(request_path), str(response_path)],
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise LocalDiarizationError(
                f"Local diarization worker exited with status {completed.returncode}. "
                "Run scripts/setup_diarization.py to build the environment and "
                "download the model, use --diarization-backend pyannote for the "
                "hosted API, or --no-align to produce a transcript without speaker "
                "attribution."
            )
        if not response_path.is_file():
            raise LocalDiarizationError("Local diarization worker produced no output file")

        payload = json.loads(response_path.read_text(encoding="utf-8"))

    for warning in payload.get("warnings", []):
        print(f"Local diarization warning: {warning}", file=sys.stderr)

    result = parse_worker_output(payload)
    if not result.turns:
        raise LocalDiarizationError("Local diarization succeeded but returned no speaker turns")

    print(
        f"Diarization found {len(result.speakers)} speaker(s) across {len(result.turns)} turns.",
        file=sys.stderr,
    )
    return result
