"""Drives the isolated WhisperX forced-alignment worker.

WhisperX cannot be installed next to this project (it pins torch and declares
`requires-python <3.14`), so it runs as a subprocess in a uv-managed Python 3.12
environment. Communication is over temporary JSON files rather than pipes,
because the worker's dependencies write to stdout on some paths.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_pipeline.config import ALIGN_WORKER, DEFAULT_ALIGN_PYTHON, align_venv_python
from transcribe_pipeline.models import Word


class AlignmentError(RuntimeError):
    pass


@dataclass
class AlignmentRequest:
    id: int
    start: float
    end: float
    text: str


@dataclass
class AlignmentResult:
    words_by_id: dict[int, list[Word]]
    warnings: list[str] = field(default_factory=list)


def default_command() -> list[str]:
    """Build the command that runs the worker in an isolated 3.12 environment."""
    override = os.environ.get("TRANSCRIBE_ALIGN_CMD", "").strip()
    if override:
        return shlex.split(override)

    prepared = align_venv_python()
    if prepared is not None:
        return [str(prepared), str(ALIGN_WORKER)]

    if not shutil.which("uv"):
        raise AlignmentError(
            "uv is required to run the WhisperX alignment worker in an isolated "
            "Python 3.12 environment. Install uv, or set TRANSCRIBE_ALIGN_CMD to a "
            "command that runs workers/align_worker.py with whisperx available."
        )

    # --no-project keeps uv from resolving against this project, which requires
    # Python 3.14 and would make whisperx uninstallable.
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        DEFAULT_ALIGN_PYTHON,
        "--with",
        "whisperx",
        str(ALIGN_WORKER),
    ]


def parse_worker_output(payload: dict) -> AlignmentResult:
    words_by_id: dict[int, list[Word]] = {}
    warnings: list[str] = []

    for item in payload.get("items", []):
        words = [
            Word(
                text=entry["word"],
                start=entry.get("start"),
                end=entry.get("end"),
                score=entry.get("score"),
            )
            for entry in item.get("words", [])
        ]
        words_by_id[int(item["id"])] = words
        warnings.extend(item.get("warnings", []))

    return AlignmentResult(words_by_id=words_by_id, warnings=warnings)


def align(
    audio: Path,
    requests: list[AlignmentRequest],
    *,
    language: str | None = None,
    device: str | None = None,
) -> AlignmentResult:
    if not ALIGN_WORKER.is_file():
        raise AlignmentError(f"Alignment worker not found at {ALIGN_WORKER}")

    payload = {
        "audio": str(audio),
        "language": language or "en",
        "device": device,
        "items": [
            {"id": request.id, "start": request.start, "end": request.end, "text": request.text}
            for request in requests
        ],
    }

    command = default_command()
    if align_venv_python() is None:
        print(
            "No prepared alignment environment found; building one on demand. "
            "Run scripts/setup_alignment.py once to avoid this.",
            file=sys.stderr,
        )
    print("Aligning transcript to audio with WhisperX...", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="align-") as workspace:
        request_path = Path(workspace) / "request.json"
        response_path = Path(workspace) / "response.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")

        completed = subprocess.run(
            [*command, str(request_path), str(response_path)],
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise AlignmentError(
                f"Alignment worker exited with status {completed.returncode}. "
                "Re-run with --no-align to produce a transcript without speaker "
                "attribution, or set TRANSCRIBE_ALIGN_CMD to a working command."
            )
        if not response_path.is_file():
            raise AlignmentError("Alignment worker produced no output file")

        return parse_worker_output(json.loads(response_path.read_text(encoding="utf-8")))
