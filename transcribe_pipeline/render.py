"""Render the pipeline result as JSON and as a readable transcript."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone

from transcribe_pipeline.models import Segment, TranscriptDocument

WRAP_WIDTH = 88
INDENT = "    "


def timestamp(seconds: float | None, *, milliseconds: bool = True) -> str:
    if seconds is None:
        return "--:--:--" + (".---" if milliseconds else "")
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}" if milliseconds else base


def render_json(document: TranscriptDocument) -> str:
    payload = {
        "audio": str(document.audio),
        "duration": round(document.duration, 3),
        "language": document.language,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": document.models,
        "speakers": document.speakers,
        "segments": [segment.to_json() for segment in document.segments],
        "warnings": document.warnings,
        # Diagnostic only: overlapping speech is removed here, so this view is
        # never used for attribution.
        "exclusive_diarization": [
            {
                "speaker": turn.speaker,
                "start": turn.start,
                "end": turn.end,
                "confidence": turn.confidence,
            }
            for turn in document.exclusive_turns
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _label(segment: Segment) -> str:
    if segment.speaker:
        return segment.speaker
    if segment.overlap:
        candidates = " / ".join(segment.speaker_candidates) or "unknown"
        return f"[overlapping speech: {candidates}]"
    if not segment.aligned:
        return "[unplaced audio]"
    return "[unattributed]"


def _blocks(segments: list[Segment]) -> list[list[Segment]]:
    """Group consecutive segments that share a speaker into turn blocks.

    Grouping keys off the speaker alone, not the overlap flag. A one-word
    interjection credited to the speaker who was already talking should not
    split their sentence in half; the overlap stays recorded in the JSON.
    """
    blocks: list[list[Segment]] = []
    for segment in segments:
        if blocks and _label(blocks[-1][-1]) == _label(segment):
            blocks[-1].append(segment)
        else:
            blocks.append([segment])
    return blocks


def render_text(document: TranscriptDocument) -> str:
    lines: list[str] = []
    title = f"Transcript — {document.audio.name}"
    lines.append(title)
    lines.append("=" * len(title))

    speakers = ", ".join(document.speakers) if document.speakers else "unknown"
    lines.append(f"Duration:  {timestamp(document.duration, milliseconds=False)}")
    lines.append(f"Speakers:  {speakers}")
    if document.language:
        lines.append(f"Language:  {document.language}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Overlap is not marked inline, so surface it here rather than losing it.
    overlapping = sum(1 for segment in document.segments if segment.overlap)
    if overlapping:
        unresolved = sum(
            1 for segment in document.segments if segment.overlap and not segment.speaker
        )
        note = f"Overlap:   {overlapping} segment(s) with simultaneous speech"
        if unresolved:
            note += f", {unresolved} unresolved"
        lines.append(note)
    lines.append("")

    if not document.segments:
        lines.append("(no speech found)")
        return "\n".join(lines) + "\n"

    for block in _blocks(document.segments):
        start = timestamp(block[0].start, milliseconds=False)
        lines.append(f"[{start}]  {_label(block[0])}")
        body = " ".join(segment.clean_text for segment in block).strip()
        lines.extend(
            textwrap.wrap(
                body,
                width=WRAP_WIDTH,
                initial_indent=INDENT,
                subsequent_indent=INDENT,
            )
            or [INDENT + "(silence)"]
        )
        lines.append("")

    if document.warnings:
        lines.append("Warnings")
        lines.append("-" * len("Warnings"))
        for warning in document.warnings:
            lines.extend(
                textwrap.wrap(warning, width=WRAP_WIDTH, initial_indent="  - ", subsequent_indent="    ")
            )
        lines.append("")

    return "\n".join(lines) + "\n"
