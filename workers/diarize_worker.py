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
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_DEVICE = "cpu"


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


def load_pipeline(model: str, device: str):
    """Load Community-1, translating the gated-model failure into advice."""
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

    print("Diarizing...", file=sys.stderr)
    output = pipeline(request["audio"], **options)

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
