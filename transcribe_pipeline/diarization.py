"""Hosted speaker diarization with pyannoteAI `precision-2`.

The whole preprocessed file is diarized in one job rather than per chunk, so
speaker labels stay consistent across the entire recording. Regular diarization
is canonical because it preserves overlapping speech; the exclusive variant is
retained only as a convenience view.
"""

from __future__ import annotations

import re
import sys
import time
import uuid
from pathlib import Path

import httpx

from transcribe_pipeline.config import DIARIZATION_MODEL, PYANNOTE_API_BASE
from transcribe_pipeline.models import DiarizationResult, Turn

POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 3600.0
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "cancelled"}
_OBJECT_KEY_SAFE = re.compile(r"[^a-zA-Z0-9\-_.]")


class DiarizationError(RuntimeError):
    pass


def _object_key(source: Path) -> str:
    stem = _OBJECT_KEY_SAFE.sub("-", source.stem)[:60] or "audio"
    return f"transcribe-dev/{stem}-{uuid.uuid4().hex}{source.suffix or '.flac'}"


def _upload_media(client: httpx.Client, source: Path) -> str:
    """Stage the audio in pyannoteAI temporary storage and return its media:// URL."""
    media_url = f"media://{_object_key(source)}"
    response = client.post(f"{PYANNOTE_API_BASE}/media/input", json={"url": media_url})
    _raise_for_status(response, "requesting a media upload location")

    presigned = response.json()["url"]
    with source.open("rb") as handle:
        # The presigned URL is pre-authorized; sending the API key to it would
        # leak the credential to the storage host.
        upload = httpx.put(
            presigned,
            content=handle,
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(600.0),
        )
    _raise_for_status(upload, "uploading audio to pyannoteAI temporary storage")
    return media_url


def _submit_job(client: httpx.Client, media_url: str, *, num_speakers: int | None) -> str:
    payload: dict[str, object] = {
        "url": media_url,
        "model": DIARIZATION_MODEL,
        "confidence": True,
        "exclusive": True,
        "turnLevelConfidence": True,
    }
    if num_speakers is not None:
        payload["numSpeakers"] = num_speakers

    response = client.post(f"{PYANNOTE_API_BASE}/diarize", json=payload)
    _raise_for_status(response, "submitting the diarization job")
    body = response.json()
    if body.get("warning"):
        print(f"pyannoteAI warning: {body['warning']}", file=sys.stderr)
    return body["jobId"]


def _await_job(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        response = client.get(f"{PYANNOTE_API_BASE}/jobs/{job_id}")
        _raise_for_status(response, f"polling diarization job {job_id}")
        body = response.json()
        status = body.get("status")

        if status in TERMINAL_STATUSES:
            if status != "succeeded":
                detail = (body.get("output") or {}).get("error") or status
                raise DiarizationError(f"Diarization job {job_id} did not succeed: {detail}")
            return body.get("output") or {}

        if time.monotonic() > deadline:
            raise DiarizationError(
                f"Diarization job {job_id} still {status} after "
                f"{POLL_TIMEOUT_SECONDS / 60:.0f} minutes"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def _coerce_confidence(value: object, speaker: str) -> float | None:
    """Normalize the per-turn confidence, which may be a scalar or a per-speaker map."""
    if isinstance(value, dict):
        value = value.get(speaker)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    # Confidence is documented as a percentage in some responses and a 0-1
    # fraction in others; store it as a fraction either way.
    return number / 100.0 if number > 1.0 else number


def parse_turns(raw: object) -> tuple[Turn, ...]:
    """Convert a raw pyannoteAI diarization array into ordered `Turn`s."""
    if not isinstance(raw, list):
        return ()

    turns: list[Turn] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            speaker = str(entry["speaker"])
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        turns.append(
            Turn(
                speaker=speaker,
                start=start,
                end=end,
                confidence=_coerce_confidence(entry.get("confidence"), speaker),
            )
        )

    turns.sort(key=lambda turn: (turn.start, turn.end))
    return tuple(turns)


def diarize(source: Path, *, api_key: str, num_speakers: int | None = None) -> DiarizationResult:
    print("Diarizing with pyannoteAI precision-2...", file=sys.stderr)
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(120.0)) as client:
        media_url = _upload_media(client, source)
        job_id = _submit_job(client, media_url, num_speakers=num_speakers)
        print(f"pyannoteAI job {job_id} submitted; waiting...", file=sys.stderr)
        output = _await_job(client, job_id)

    if output.get("warning"):
        print(f"pyannoteAI warning: {output['warning']}", file=sys.stderr)

    result = DiarizationResult(
        turns=parse_turns(output.get("diarization")),
        exclusive_turns=parse_turns(output.get("exclusiveDiarization")),
    )
    if not result.turns:
        raise DiarizationError("Diarization succeeded but returned no speaker turns")

    print(
        f"Diarization found {len(result.speakers)} speaker(s) across {len(result.turns)} turns.",
        file=sys.stderr,
    )
    return result


def _raise_for_status(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip()
    if len(detail) > 500:
        detail = detail[:500] + "..."
    raise DiarizationError(f"pyannoteAI error while {action}: HTTP {response.status_code} {detail}")
