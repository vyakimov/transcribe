"""AssemblyAI transcription and speaker labels.

One AssemblyAI job returns words, word timings, and speaker labels together.
`transcribe` keeps all of it and is the default transcription backend.
`diarize` keeps only the speaker timeline, for use as a diarization backend
under the `gpt` transcription backend, where the words come from
`gpt-transcribe` and the timings from forced alignment.

The whole preprocessed file is sent in one job so speaker labels stay consistent
across the recording. AssemblyAI assigns every instant to a single speaker and
does not report overlapping speech, so the regular and exclusive timelines are
the same.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

from transcribe_pipeline.config import ASSEMBLYAI_API_BASE, ASSEMBLYAI_KEY_VARIABLES
from transcribe_pipeline.diarization import DiarizationError
from transcribe_pipeline.models import DiarizationResult, Turn, Word

POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 3600.0


def api_key_from_environment() -> str:
    """The first AssemblyAI key variable that is set, or an empty string."""
    for name in ASSEMBLYAI_KEY_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _upload(client: httpx.Client, source: Path) -> str:
    with source.open("rb") as handle:
        response = client.post(
            f"{ASSEMBLYAI_API_BASE}/upload",
            content=handle,
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(600.0),
        )
    _raise_for_status(response, "uploading audio")
    return response.json()["upload_url"]


def _submit_job(
    client: httpx.Client,
    audio_url: str,
    *,
    num_speakers: int | None,
    language: str | None,
    speaker_labels: bool = True,
) -> str:
    payload: dict[str, object] = {"audio_url": audio_url, "speaker_labels": speaker_labels}
    if speaker_labels and num_speakers is not None:
        payload["speakers_expected"] = num_speakers
    if language:
        payload["language_code"] = language
    else:
        payload["language_detection"] = True

    response = client.post(f"{ASSEMBLYAI_API_BASE}/transcript", json=payload)
    _raise_for_status(response, "submitting the diarization job")
    return response.json()["id"]


def _await_job(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        response = client.get(f"{ASSEMBLYAI_API_BASE}/transcript/{job_id}")
        _raise_for_status(response, f"polling diarization job {job_id}")
        body = response.json()
        status = body.get("status")

        if status == "completed":
            return body
        if status == "error":
            detail = body.get("error") or status
            raise DiarizationError(f"AssemblyAI job {job_id} did not succeed: {detail}")

        if time.monotonic() > deadline:
            raise DiarizationError(
                f"AssemblyAI job {job_id} still {status} after "
                f"{POLL_TIMEOUT_SECONDS / 60:.0f} minutes"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def parse_turns(raw: object) -> tuple[Turn, ...]:
    """Convert AssemblyAI utterances (or words) into ordered `Turn`s.

    Times arrive in milliseconds and speakers as bare letters; the letters are
    prefixed so the labels read like the other backends' (`SPEAKER_A`).
    """
    if not isinstance(raw, list):
        return ()

    turns: list[Turn] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("speaker"):
            continue
        try:
            start = float(entry["start"]) / 1000.0
            end = float(entry["end"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        confidence = entry.get("confidence")
        turns.append(
            Turn(
                speaker=f"SPEAKER_{entry['speaker']}",
                start=start,
                end=end,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            )
        )

    turns.sort(key=lambda turn: (turn.start, turn.end))
    return tuple(turns)


def _run_job(
    source: Path,
    *,
    api_key: str,
    num_speakers: int | None,
    language: str | None,
    speaker_labels: bool = True,
) -> dict:
    # AssemblyAI takes the bare key, without a `Bearer` prefix.
    headers = {"Authorization": api_key}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(120.0)) as client:
        audio_url = _upload(client, source)
        job_id = _submit_job(
            client,
            audio_url,
            num_speakers=num_speakers,
            language=language,
            speaker_labels=speaker_labels,
        )
        print(f"AssemblyAI job {job_id} submitted; waiting...", file=sys.stderr)
        return _await_job(client, job_id)


def parse_words(raw: object) -> list[Word]:
    """Convert AssemblyAI words into timed, speaker-attributed `Word`s."""
    if not isinstance(raw, list):
        return []

    words: list[Word] = []
    for entry in raw:
        if not isinstance(entry, dict) or not str(entry.get("text") or "").strip():
            continue
        try:
            start: float | None = float(entry["start"]) / 1000.0
            end: float | None = float(entry["end"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            start = end = None
        confidence = entry.get("confidence")
        speaker = entry.get("speaker")
        words.append(
            Word(
                text=str(entry["text"]).strip(),
                start=start,
                end=end,
                score=float(confidence) if isinstance(confidence, (int, float)) else None,
                speaker=f"SPEAKER_{speaker}" if speaker else None,
            )
        )
    return words


def transcribe(
    source: Path,
    *,
    api_key: str,
    num_speakers: int | None = None,
    language: str | None = None,
    speaker_labels: bool = True,
) -> tuple[list[Word], DiarizationResult, dict]:
    """Transcribe with AssemblyAI; return its words, speaker turns, and raw body."""
    print("Transcribing with AssemblyAI...", file=sys.stderr)
    body = _run_job(
        source,
        api_key=api_key,
        num_speakers=num_speakers,
        language=language,
        speaker_labels=speaker_labels,
    )

    words = parse_words(body.get("words"))
    if not words:
        raise DiarizationError("AssemblyAI transcription succeeded but returned no words")

    turns = parse_turns(body.get("utterances")) if speaker_labels else ()
    result = DiarizationResult(turns=turns, exclusive_turns=turns)
    print(
        f"AssemblyAI returned {len(words)} words from {len(result.speakers)} speaker(s).",
        file=sys.stderr,
    )
    return words, result, body


def diarize(
    source: Path,
    *,
    api_key: str,
    num_speakers: int | None = None,
    language: str | None = None,
) -> DiarizationResult:
    print("Diarizing with AssemblyAI speaker labels...", file=sys.stderr)
    body = _run_job(source, api_key=api_key, num_speakers=num_speakers, language=language)

    # Utterances are the speaker turns; words are the fallback if they are absent.
    turns = parse_turns(body.get("utterances")) or parse_turns(body.get("words"))
    if not turns:
        raise DiarizationError("AssemblyAI diarization succeeded but returned no speaker turns")

    result = DiarizationResult(turns=turns, exclusive_turns=turns)
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
    raise DiarizationError(f"AssemblyAI error while {action}: HTTP {response.status_code} {detail}")
