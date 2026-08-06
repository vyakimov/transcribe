"""JSON and human-readable rendering."""

from __future__ import annotations

import json
from pathlib import Path

from transcribe_pipeline.models import Segment, TranscriptDocument, Turn
from transcribe_pipeline.render import render_json, render_text, timestamp


def segment(raw: str, **overrides) -> Segment:
    defaults = dict(
        start=0.0,
        end=1.0,
        speaker="SPEAKER_00",
        speaker_candidates=[],
        overlap=False,
        raw_text=raw,
        clean_text=raw,
    )
    defaults.update(overrides)
    return Segment(**defaults)


def document(segments: list[Segment], **overrides) -> TranscriptDocument:
    defaults = dict(
        audio=Path("input/interview.m4a"),
        duration=63.5,
        language="en",
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=segments,
        warnings=[],
        models={"transcription": "gpt-transcribe"},
        exclusive_turns=(),
    )
    defaults.update(overrides)
    return TranscriptDocument(**defaults)


class TestTimestamp:
    def test_formats_hours_minutes_seconds_and_milliseconds(self):
        assert timestamp(3723.456) == "01:02:03.456"

    def test_can_omit_milliseconds(self):
        assert timestamp(3723.456, milliseconds=False) == "01:02:03"

    def test_missing_time_renders_as_placeholder(self):
        assert timestamp(None, milliseconds=False) == "--:--:--"


class TestJson:
    def test_segments_keep_raw_and_clean_text(self):
        payload = json.loads(render_json(document([segment("hello", clean_text="Hello.")])))

        assert payload["segments"][0]["raw_text"] == "hello"
        assert payload["segments"][0]["clean_text"] == "Hello."

    def test_overlap_metadata_is_preserved(self):
        payload = json.loads(
            render_json(
                document(
                    [
                        segment(
                            "yes",
                            start=12.4,
                            end=12.8,
                            speaker=None,
                            speaker_candidates=["SPEAKER_00", "SPEAKER_01"],
                            overlap=True,
                        )
                    ]
                )
            )
        )
        entry = payload["segments"][0]

        assert entry["overlap"] is True
        assert entry["speaker"] is None
        assert entry["speaker_candidates"] == ["SPEAKER_00", "SPEAKER_01"]
        assert entry["start"] == 12.4

    def test_warnings_are_included(self):
        payload = json.loads(render_json(document([segment("hi")], warnings=["something odd"])))

        assert payload["warnings"] == ["something odd"]

    def test_exclusive_diarization_is_kept_as_a_diagnostic_view(self):
        payload = json.loads(
            render_json(
                document([segment("hi")], exclusive_turns=(Turn("SPEAKER_00", 0.0, 1.0, 0.9),))
            )
        )

        assert payload["exclusive_diarization"] == [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "confidence": 0.9}
        ]

    def test_output_is_valid_json_with_a_trailing_newline(self):
        rendered = render_json(document([segment("hi")]))

        assert rendered.endswith("\n")
        json.loads(rendered)


class TestReadableText:
    def test_header_reports_duration_and_speakers(self):
        rendered = render_text(document([segment("hello")]))

        assert "interview.m4a" in rendered
        assert "00:01:03" in rendered
        assert "SPEAKER_00, SPEAKER_01" in rendered

    def test_cleaned_text_is_what_gets_rendered(self):
        rendered = render_text(document([segment("hello there", clean_text="Hello there.")]))

        assert "Hello there." in rendered
        assert "hello there" not in rendered

    def test_consecutive_segments_from_one_speaker_share_a_block(self):
        rendered = render_text(
            document(
                [
                    segment("first", start=0.0, end=1.0),
                    segment("second", start=1.0, end=2.0),
                    segment("third", start=2.0, end=3.0, speaker="SPEAKER_01"),
                ]
            )
        )

        headers = [line for line in rendered.splitlines() if line.startswith("[")]

        assert headers == ["[00:00:00]  SPEAKER_00", "[00:00:02]  SPEAKER_01"]
        assert "first second" in rendered

    def test_unresolved_overlapping_speech_is_visibly_marked(self):
        rendered = render_text(
            document(
                [
                    segment(
                        "yes",
                        speaker=None,
                        speaker_candidates=["SPEAKER_00", "SPEAKER_01"],
                        overlap=True,
                    )
                ]
            )
        )

        assert "overlapping speech" in rendered
        assert "SPEAKER_00 / SPEAKER_01" in rendered

    def test_a_resolved_overlap_does_not_split_the_speaker_block(self):
        """A credited interjection must not chop the surrounding sentence in half."""
        rendered = render_text(
            document(
                [
                    segment("I don't", start=0.0, end=1.0),
                    segment(
                        "know",
                        start=1.0,
                        end=1.4,
                        speaker_candidates=["SPEAKER_00", "SPEAKER_01"],
                        overlap=True,
                    ),
                    segment("maybe you had a chance", start=1.4, end=3.0),
                ]
            )
        )
        headers = [line for line in rendered.splitlines() if line.startswith("[")]

        assert headers == ["[00:00:00]  SPEAKER_00"]
        assert "I don't know maybe you had a chance" in rendered

    def test_the_header_reports_the_overlap_count(self):
        rendered = render_text(
            document(
                [
                    segment("hi", overlap=True, speaker_candidates=["SPEAKER_00", "SPEAKER_01"]),
                    segment("there"),
                ]
            )
        )

        assert "Overlap:   1 segment(s) with simultaneous speech" in rendered

    def test_unresolved_overlaps_are_counted_separately(self):
        rendered = render_text(
            document(
                [
                    segment(
                        "hi",
                        overlap=True,
                        speaker=None,
                        speaker_candidates=["SPEAKER_00", "SPEAKER_01"],
                    )
                ]
            )
        )

        assert "1 unresolved" in rendered

    def test_no_overlap_line_when_there_is_no_overlap(self):
        rendered = render_text(document([segment("hi")]))

        assert "Overlap:" not in rendered

    def test_unplaced_audio_is_labelled(self):
        rendered = render_text(
            document([segment("untimed", start=None, end=None, speaker=None, aligned=False)])
        )

        assert "[unplaced audio]" in rendered
        assert "--:--:--" in rendered

    def test_warnings_are_listed_at_the_end(self):
        rendered = render_text(document([segment("hi")], warnings=["3 words unaligned"]))

        assert "Warnings" in rendered
        assert "- 3 words unaligned" in rendered

    def test_long_text_is_wrapped(self):
        rendered = render_text(document([segment(" ".join(["word"] * 120))]))
        body = [line for line in rendered.splitlines() if line.startswith("    ")]

        assert len(body) > 1
        assert all(len(line) <= 90 for line in rendered.splitlines())

    def test_empty_transcript_is_handled(self):
        rendered = render_text(document([]))

        assert "(no speech found)" in rendered
