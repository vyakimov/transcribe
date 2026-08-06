"""The cleanup pass must only ever change punctuation, casing, and whitespace."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from transcribe_pipeline.cleanup import (
    CleanedSegment,
    CleanupResponse,
    clean_segments,
    preserves_words,
)
from transcribe_pipeline.models import Segment


def segment(raw: str, **overrides) -> Segment:
    defaults = dict(
        start=1.0,
        end=2.0,
        speaker="SPEAKER_00",
        speaker_candidates=[],
        overlap=False,
        raw_text=raw,
        clean_text=raw,
    )
    defaults.update(overrides)
    return Segment(**defaults)


class StubClient:
    """Stands in for the OpenAI client at the Responses API boundary."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(output_parsed=self.reply)


def reply_with(*texts: str) -> CleanupResponse:
    return CleanupResponse(
        segments=[CleanedSegment(index=index, clean_text=text) for index, text in enumerate(texts)]
    )


class TestValidation:
    @pytest.mark.parametrize(
        "raw, cleaned",
        [
            ("hello there how are you", "Hello there, how are you?"),
            ("its fine", "It's fine."),
            ("wait  no", "Wait — no!"),
            ("okay", "Okay..."),
        ],
    )
    def test_punctuation_and_casing_changes_are_accepted(self, raw, cleaned):
        assert preserves_words(raw, cleaned) is True

    @pytest.mark.parametrize(
        "raw, cleaned, reason",
        [
            ("i went there", "I went over there.", "added word"),
            ("i really went there", "I went there.", "deleted word"),
            ("i went there", "There I went.", "reordered words"),
            ("im gonna go", "I'm going to go.", "expanded a contraction"),
            ("twenty five people", "25 people.", "rewrote a number"),
            ("the wellbeing of", "The well being of.", "split a word"),
        ],
    )
    def test_word_level_edits_are_rejected(self, raw, cleaned, reason):
        assert preserves_words(raw, cleaned) is False, reason


class TestCleanSegments:
    def test_accepted_cleanup_updates_clean_text_and_keeps_raw_text(self):
        segments = [segment("hello there how are you")]
        client = StubClient(reply_with("Hello there, how are you?"))

        warnings = clean_segments(client, segments)

        assert segments[0].clean_text == "Hello there, how are you?"
        assert segments[0].raw_text == "hello there how are you"
        assert warnings == []

    def test_rejected_cleanup_falls_back_to_raw_text_with_a_warning(self):
        segments = [segment("i went there")]
        client = StubClient(reply_with("I went over there."))

        warnings = clean_segments(client, segments)

        assert segments[0].clean_text == "i went there"
        assert any("rejected" in warning for warning in warnings)

    def test_a_failing_request_leaves_every_segment_untouched(self):
        segments = [segment("first one"), segment("second one")]
        client = StubClient(RuntimeError("connection reset"))

        warnings = clean_segments(client, segments)

        assert [s.clean_text for s in segments] == ["first one", "second one"]
        assert any("connection reset" in warning for warning in warnings)

    def test_unparsable_output_leaves_segments_untouched(self):
        segments = [segment("first one")]
        client = StubClient(None)

        warnings = clean_segments(client, segments)

        assert segments[0].clean_text == "first one"
        assert any("no parsable output" in warning for warning in warnings)

    def test_unknown_indices_are_ignored(self):
        segments = [segment("only one")]
        client = StubClient(
            CleanupResponse(segments=[CleanedSegment(index=7, clean_text="Injected.")])
        )

        warnings = clean_segments(client, segments)

        assert segments[0].clean_text == "only one"
        assert any("unknown segment index" in warning for warning in warnings)

    def test_cleanup_cannot_change_timestamps_speakers_or_overlap(self):
        original = segment(
            "yes",
            start=10.1,
            end=10.5,
            speaker=None,
            speaker_candidates=["SPEAKER_00", "SPEAKER_01"],
            overlap=True,
        )
        client = StubClient(reply_with("Yes."))

        clean_segments(client, [original])

        assert original.start == 10.1 and original.end == 10.5
        assert original.speaker is None
        assert original.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]
        assert original.overlap is True
        assert original.clean_text == "Yes."

    def test_only_segment_text_is_sent_to_the_model(self):
        segments = [segment("hello", speaker="SPEAKER_03", start=99.0)]
        client = StubClient(reply_with("Hello."))

        clean_segments(client, segments)

        payload = client.calls[0]["input"]
        assert "hello" in payload
        assert "SPEAKER_03" not in payload
        assert "99.0" not in payload

    def test_segments_are_batched(self):
        segments = [segment(f"line {index}") for index in range(60)]
        client = StubClient(reply_with())

        clean_segments(client, segments)

        assert len(client.calls) == 3
