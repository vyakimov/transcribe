"""Opt-in end-to-end run against the real hosted services.

Skipped by default: it spends money, uploads audio, and needs the isolated
WhisperX environment. Enable it with

    TRANSCRIBE_E2E=1 uv run pytest tests/test_end_to_end.py -s

`-s` matters — the crosstalk sections are printed for manual inspection, which
is the part no assertion can replace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transcribe_pipeline.config import load_environment
from transcribe_pipeline.pipeline import Options, run
from transcribe_pipeline.render import render_text, timestamp

FIXTURES = [
    Path("input/itrex.m4a"),
    Path("input/bcg_interview_feedback.m4a"),
]

load_environment()

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TRANSCRIBE_E2E") != "1",
        reason="set TRANSCRIBE_E2E=1 to run against the real services",
    ),
    pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") and os.environ.get("PYANNOTE_API_KEY")),
        reason="both OPENAI_API_KEY and PYANNOTE_API_KEY are required",
    ),
]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.stem)
def test_real_interview_transcribes_and_attributes(fixture: Path):
    if not fixture.is_file():
        pytest.skip(f"fixture not present: {fixture}")

    document, raw_responses = run(Options(audio=fixture, language="en"))

    assert document.segments, "no speech was transcribed"
    assert raw_responses, "raw transcription responses were not retained"
    assert len(document.speakers) >= 2, f"expected a conversation, got {document.speakers}"

    attributed = [segment for segment in document.segments if segment.speaker]
    assert len(attributed) > len(document.segments) * 0.7, (
        "fewer than 70% of segments were attributed to a speaker; "
        f"warnings: {document.warnings}"
    )

    # Every segment keeps its unedited text no matter what cleanup did.
    assert all(segment.raw_text for segment in document.segments)

    print(render_text(document))


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.stem)
def test_crosstalk_sections_are_flagged_for_inspection(fixture: Path):
    """Prints every overlapping region so a human can check the hard cases."""
    if not fixture.is_file():
        pytest.skip(f"fixture not present: {fixture}")

    document, _ = run(Options(audio=fixture, language="en"))
    overlaps = [segment for segment in document.segments if segment.overlap]

    print(f"\n{fixture.name}: {len(overlaps)} overlapping segment(s)")
    for segment in overlaps:
        candidates = " / ".join(segment.speaker_candidates) or "unknown"
        resolved = segment.speaker or "ambiguous"
        print(
            f"  [{timestamp(segment.start)} -> {timestamp(segment.end)}] "
            f"({candidates} -> {resolved}) {segment.clean_text}"
        )

    # Overlap must never be resolved by inventing a speaker outside the candidates.
    for segment in overlaps:
        assert segment.speaker is None or segment.speaker in segment.speaker_candidates
