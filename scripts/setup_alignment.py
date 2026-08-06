#!/usr/bin/env python3
"""Build the isolated environment the forced aligner runs in.

WhisperX pins torch and declares `requires-python <3.14`, so it cannot be
installed alongside this project. This creates `.venv-align` on Python 3.12,
installs whisperx there, and downloads the wav2vec2 model and the NLTK sentence
tokenizer up front so the first real transcription does not stall on a download.

    uv run python scripts/setup_alignment.py [--language en] [--force]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv-align"
PYTHON_VERSION = "3.12"

WARMUP = """
import sys
import whisperx
language = sys.argv[1]
print(f"Downloading the wav2vec2 alignment model for {language}...", file=sys.stderr)
whisperx.load_align_model(language_code=language, device="cpu")
print("Downloading the NLTK sentence tokenizer...", file=sys.stderr)
import nltk
nltk.download("punkt_tab", quiet=True)
print("ok")
"""


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", file=sys.stderr)
    subprocess.run(command, check=True)


def venv_python() -> Path:
    for candidate in (VENV / "bin" / "python", VENV / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No interpreter found in {VENV}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en", help="Language to pre-download a model for")
    parser.add_argument("--force", action="store_true", help="Recreate the environment")
    args = parser.parse_args()

    if not shutil.which("uv"):
        raise SystemExit("uv is required to build the alignment environment")

    if args.force and VENV.exists():
        print(f"Removing {VENV}...", file=sys.stderr)
        shutil.rmtree(VENV)

    if not VENV.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, str(VENV)])

    python = venv_python()
    run(["uv", "pip", "install", "--python", str(python), "whisperx"])
    run([str(python), "-c", WARMUP, args.language])

    print(f"\nAlignment environment ready at {VENV}", file=sys.stderr)
    print("transcribe.py will use it automatically.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
