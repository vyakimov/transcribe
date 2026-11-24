#!/usr/bin/env python3
"""Build the isolated environment local diarization runs in.

pyannote.audio pulls in torch, which does not support this project's Python
version, so it cannot be installed alongside it. This creates `.venv-diarize` on
Python 3.12, installs pyannote.audio there, and downloads Community-1 up front so
the first real transcription does not stall on a 1 GB download.

Community-1 is gated: accept the conditions at
https://hf.co/pyannote/speaker-diarization-community-1 and put a token with
access to it in `.env` as `HF_TOKEN` before running this.

    uv run python scripts/setup_diarization.py [--force]

`TRANSCRIBE_DIARIZE_VENV` relocates the environment. With `--models-only` the
environment is assumed to exist already and only the download runs, which is how
the container image — where the environment is baked into the image and the
weights are not — warms its cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transcribe_pipeline.config import (  # noqa: E402 - needs ROOT on sys.path
    COMMUNITY_MODEL_ID,
    DEFAULT_DIARIZE_PYTHON,
    diarize_venv,
    load_environment,
)

# Honours TRANSCRIBE_DIARIZE_VENV, so this builds the same environment the
# pipeline will later look for — including inside the container image.
DIARIZE_VENV = diarize_venv()

WARMUP = """
import os
import sys
from pyannote.audio import Pipeline

model = sys.argv[1]
print(f"Downloading {model}...", file=sys.stderr)
pipeline = Pipeline.from_pretrained(model, token=os.environ.get("HF_TOKEN") or None)
if pipeline is None:
    raise SystemExit(
        f"{model} could not be loaded. Accept the model conditions at "
        f"https://hf.co/{model} and set HF_TOKEN to a token with access to it."
    )
print("ok")
"""


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(command)}", file=sys.stderr)
    subprocess.run(command, check=True, env=env)


def venv_python() -> Path:
    for candidate in (DIARIZE_VENV / "bin" / "python", DIARIZE_VENV / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No interpreter found in {DIARIZE_VENV}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recreate the environment")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help=(
            "Download the model into an environment that already exists, without "
            "creating or installing into it. This is how the container image warms "
            "its cache: the environment is baked in, only the weights are missing."
        ),
    )
    args = parser.parse_args()

    if not args.models_only and not shutil.which("uv"):
        raise SystemExit("uv is required to build the diarization environment")

    # The token lives in `.env` alongside the API keys, so load it the same way
    # the pipeline does before handing it to the subprocess.
    load_environment()
    if not os.environ.get("HF_TOKEN", "").strip():
        print(
            "HF_TOKEN is not set. It is required to download the gated model; "
            "an already-cached model will still load without it.",
            file=sys.stderr,
        )

    if args.models_only:
        if args.force:
            raise SystemExit("--models-only and --force contradict each other")
    else:
        if args.force and DIARIZE_VENV.exists():
            print(f"Removing {DIARIZE_VENV}...", file=sys.stderr)
            shutil.rmtree(DIARIZE_VENV)

        if not DIARIZE_VENV.exists():
            run(["uv", "venv", "--python", DEFAULT_DIARIZE_PYTHON, str(DIARIZE_VENV)])

    python = venv_python()
    if not args.models_only:
        run(["uv", "pip", "install", "--python", str(python), "pyannote.audio"])
    run([str(python), "-c", WARMUP, COMMUNITY_MODEL_ID], env=dict(os.environ))

    print(f"\nDiarization environment ready at {DIARIZE_VENV}", file=sys.stderr)
    print("Select it with --diarization-backend community.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
