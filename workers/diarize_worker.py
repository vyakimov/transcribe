#!/usr/bin/env python3
"""Isolated local speaker diarization worker (pyannote Community-1).

Runs in its own Python 3.12 environment because pyannote.audio pulls in torch,
which does not support this project's Python version. The whole preprocessed
file is diarized in one call, exactly like the hosted pyannoteAI path, so
speaker labels stay consistent end to end.

Usage: diarize_worker.py <request.json> <response.json>

Request:
    {"audio": str, "model": str, "num_speakers": int|null,
     "device": "cpu"|"cuda"|"mps"|null}

Response:
    {"diarization": [{"speaker": str, "start": float, "end": float}, ...],
     "exclusive_diarization": [...], "warnings": [str, ...]}

The response is deliberately provider-neutral: it carries the same shape the
pyannoteAI API returns, so both backends land in the same `DiarizationResult`.

`HF_TOKEN` is read from the environment rather than the request, so the
credential never reaches a temporary file on disk. It is only needed to download
the gated model; once cached, runs need no token at all.

Audio is decoded here with the ffmpeg CLI and handed to pyannote as a waveform
dictionary rather than as a path. pyannote's own decoder is torchcodec, whose
prebuilt libraries are linked against a specific ffmpeg ABI at a hard-coded
rpath, so it breaks whenever the host's ffmpeg is a different major version or
sits at a different prefix. The rest of this project already drives ffmpeg
directly, so decoding here removes that fragile native dependency instead of
trying to satisfy it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DEVICE = "cpu"

# pyannote's models run at 16 kHz. Decoding straight to that rate hands the
# pipeline exactly what it wants and skips a redundant torchaudio resample.
PYANNOTE_SAMPLE_RATE = 16000


def _loose_pairs(annotation):
    """Fallback for pyannote builds whose output is already (segment, label)."""
    for item in annotation:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            yield item[0], item[-1]


def extract_turns(annotation) -> list[dict]:
    """Flatten a pyannote annotation into ordered, provider-neutral turns.

    Community-1 reports no per-turn confidence, so the field is simply absent
    rather than invented; the attribution stage treats it as unknown.
    """
    if annotation is None:
        return []

    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        pairs = ((segment, label) for segment, _track, label in itertracks(yield_label=True))
    else:
        pairs = _loose_pairs(annotation)

    turns: list[dict] = []
    for segment, label in pairs:
        start, end = float(segment.start), float(segment.end)
        if end <= start:
            continue
        turns.append(
            {"speaker": str(label), "start": round(start, 3), "end": round(end, 3)}
        )

    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    return turns


def decode_audio(path: str):
    """Decode any ffmpeg-readable file to a mono (1, time) float32 tensor.

    Returns the waveform dictionary pyannote accepts in place of a path, which
    is the documented escape hatch from its torchcodec-based decoding.
    """
    import torch

    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "ffmpeg is required to decode audio for local diarization but was "
            "not found on PATH."
        )

    if not Path(path).is_file():
        raise SystemExit(f"Audio file not found: {path}")

    print(f"Decoding {Path(path).name} with ffmpeg...", file=sys.stderr)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(PYANNOTE_SAMPLE_RATE),
            "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(f"ffmpeg could not decode {path}: {detail}")
    if not completed.stdout:
        raise SystemExit(f"ffmpeg produced no audio samples for {path}")

    # frombuffer would alias a read-only buffer, so copy into a writable tensor.
    waveform = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32)
    return {
        "waveform": waveform.reshape(1, -1),
        "sample_rate": PYANNOTE_SAMPLE_RATE,
        "uri": Path(path).stem,
    }


def load_pipeline(model: str, device: str):
    """Load Community-1, translating the gated-model failure into advice."""
    # We decode with ffmpeg and never touch torchcodec, but importing pyannote
    # still probes for it and warns at length when its libraries will not load.
    # That warning reads exactly like a fatal error, so drop this one message.
    # Imported here, not at module scope, because `main` binds `warnings` to a
    # list of the warnings this worker reports back to the caller.
    import warnings

    warnings.filterwarnings(
        "ignore", message=r"\s*torchcodec is not installed correctly"
    )

    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN", "").strip() or None
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as error:  # noqa: BLE001 - the cause is almost always access
        raise SystemExit(
            f"Could not load {model}: {error}\n"
            f"Accept the model conditions at https://hf.co/{model} and set HF_TOKEN, "
            "then run scripts/setup_diarization.py."
        ) from error

    # pyannote returns None rather than raising when the gated repository has not
    # been accepted for this token.
    if pipeline is None:
        raise SystemExit(
            f"{model} could not be loaded. Accept the model conditions at "
            f"https://hf.co/{model}, then set HF_TOKEN to a token with access to it."
        )

    return pipeline.to(torch.device(device))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    # pyannote and its dependencies print progress and warnings to stdout.
    # Results travel through a file, so stdout is retargeted at stderr to keep
    # that output out of the caller's stdout.
    sys.stdout = sys.stderr

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    destination = Path(sys.argv[2])

    model = request.get("model") or "pyannote/speaker-diarization-community-1"
    device = request.get("device") or DEFAULT_DEVICE

    print(f"Loading {model} ({device})...", file=sys.stderr)
    pipeline = load_pipeline(model, device)

    options: dict[str, object] = {}
    num_speakers = request.get("num_speakers")
    if num_speakers:
        options["num_speakers"] = int(num_speakers)

    audio = decode_audio(request["audio"])

    print("Diarizing...", file=sys.stderr)
    output = pipeline(audio, **options)

    warnings: list[str] = []
    diarization = extract_turns(getattr(output, "speaker_diarization", output))
    exclusive = extract_turns(getattr(output, "exclusive_speaker_diarization", None))
    if not exclusive:
        warnings.append(
            "the local diarizer returned no exclusive speaker timeline, so "
            "overlapping words can only be resolved by dominance"
        )

    destination.write_text(
        json.dumps(
            {
                "diarization": diarization,
                "exclusive_diarization": exclusive,
                "warnings": warnings,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
