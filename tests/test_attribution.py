"""Word-to-speaker attribution, seam handling, and segment grouping."""

from __future__ import annotations

import pytest

from transcribe_pipeline.attribution import (
    attribute_words,
    group_segments,
    keep_owned,
)
from transcribe_pipeline.models import Turn, Word


def word(text: str, start: float | None = None, end: float | None = None) -> Word:
    return Word(text=text, start=start, end=end)


def speech(prefix: str, start: float, count: int, step: float = 0.4) -> list[Word]:
    return [
        word(f"{prefix}{index}", start + index * step, start + index * step + step * 0.9)
        for index in range(count)
    ]


A_ALONE = (Turn("SPEAKER_00", 0.0, 20.0),)
CROSSTALK = (
    Turn("SPEAKER_00", 0.0, 20.0),
    Turn("SPEAKER_01", 10.0, 10.6),
)


def test_single_speaker_words_get_one_speaker_and_no_overlap():
    words = attribute_words(speech("w", 1.0, 5), A_ALONE)

    assert [w.speaker for w in words] == ["SPEAKER_00"] * 5
    assert not any(w.overlap for w in words)
    assert all(w.speaker_candidates == [] for w in words)


def test_crosstalk_interjection_is_flagged_overlapping_and_ambiguous():
    """A short "yes" inside a continuing turn must not be silently credited."""
    words = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK)
    interjection = words[0]

    assert interjection.overlap is True
    assert interjection.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]
    assert interjection.speaker is None


class TestExclusiveTiebreaker:
    """Exclusive diarization names the speaker most likely to be transcribed."""

    EXCLUSIVE = (Turn("SPEAKER_01", 10.0, 10.6),)

    def test_it_resolves_a_word_the_canonical_view_left_ambiguous(self):
        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK, self.EXCLUSIVE)[0]

        assert result.speaker == "SPEAKER_01"

    def test_the_overlap_flag_and_candidates_survive_resolution(self):
        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK, self.EXCLUSIVE)[0]

        assert result.overlap is True
        assert result.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]

    def test_a_speaker_outside_the_candidates_is_refused(self):
        """The tiebreaker may narrow an attribution, never invent one."""
        exclusive = (Turn("SPEAKER_09", 10.0, 10.6),)

        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK, exclusive)[0]

        assert result.speaker is None

    def test_no_exclusive_coverage_leaves_the_word_ambiguous(self):
        exclusive = (Turn("SPEAKER_01", 30.0, 31.0),)

        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK, exclusive)[0]

        assert result.speaker is None

    def test_it_is_not_consulted_when_only_one_speaker_is_active(self):
        exclusive = (Turn("SPEAKER_01", 0.0, 20.0),)

        result = attribute_words([word("solo", 1.0, 1.4)], A_ALONE, exclusive)[0]

        assert result.speaker == "SPEAKER_00"
        assert result.overlap is False

    def test_it_is_not_consulted_when_a_dominant_speaker_already_won(self):
        turns = (Turn("SPEAKER_00", 0.0, 20.0), Turn("SPEAKER_01", 10.0, 10.2))
        exclusive = (Turn("SPEAKER_01", 10.0, 11.0),)

        result = attribute_words([word("clearly", 10.0, 11.0)], turns, exclusive)[0]

        assert result.speaker == "SPEAKER_00"

    def test_the_speaker_with_the_most_exclusive_coverage_wins(self):
        exclusive = (Turn("SPEAKER_00", 10.0, 10.45), Turn("SPEAKER_01", 10.45, 10.6))

        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK, exclusive)[0]

        assert result.speaker == "SPEAKER_00"

    def test_omitting_it_preserves_the_conservative_behaviour(self):
        result = attribute_words([word("yes", 10.1, 10.5)], CROSSTALK)[0]

        assert result.speaker is None


def test_crosstalk_does_not_duplicate_the_interjection():
    """The interjection appears exactly once: it is transcribed with the chunk."""
    words = attribute_words(
        [word("keep", 9.4, 9.8), word("yes", 10.1, 10.5), word("going", 10.9, 11.3)],
        CROSSTALK,
    )

    assert [w.text for w in words] == ["keep", "yes", "going"]
    assert [w.text for w in words].count("yes") == 1
    assert [w.speaker for w in words] == ["SPEAKER_00", None, "SPEAKER_00"]


def test_words_outside_the_overlap_region_are_unaffected():
    before, after = attribute_words([word("before", 9.0, 9.4), word("after", 12.0, 12.4)], CROSSTALK)

    assert (before.speaker, before.overlap) == ("SPEAKER_00", False)
    assert (after.speaker, after.overlap) == ("SPEAKER_00", False)


def test_dominant_speaker_wins_but_overlap_is_still_recorded():
    """A turn clipping the edge of a word should not erase an obvious attribution."""
    turns = (Turn("SPEAKER_00", 0.0, 20.0), Turn("SPEAKER_01", 10.0, 10.2))
    result = attribute_words([word("clearly", 10.0, 11.0)], turns)[0]

    assert result.speaker == "SPEAKER_00"
    assert result.overlap is True
    assert result.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]


def test_simultaneous_words_from_two_speakers_are_both_ambiguous():
    turns = (Turn("SPEAKER_00", 5.0, 6.0), Turn("SPEAKER_01", 5.0, 6.0))
    first, second = attribute_words([word("hello", 5.1, 5.5), word("there", 5.5, 5.9)], turns)

    assert first.speaker is None and second.speaker is None
    assert first.overlap and second.overlap
    assert first.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]


def test_word_in_a_diarization_gap_snaps_to_the_nearest_turn():
    turns = (Turn("SPEAKER_00", 0.0, 5.0), Turn("SPEAKER_01", 8.0, 12.0))
    result = attribute_words([word("mm", 5.2, 5.4)], turns)[0]

    assert result.speaker == "SPEAKER_00"
    assert result.overlap is False


def test_word_far_from_every_turn_stays_unattributed():
    turns = (Turn("SPEAKER_00", 0.0, 5.0),)
    result = attribute_words([word("distant", 30.0, 30.4)], turns)[0]

    assert result.speaker is None


def test_missing_diarization_leaves_every_word_unattributed():
    words = attribute_words(speech("w", 0.0, 4), ())

    assert all(w.speaker is None for w in words)
    assert not any(w.overlap for w in words)
    assert [w.text for w in words] == ["w0", "w1", "w2", "w3"]


def test_unaligned_words_are_retained_and_never_guessed():
    words = attribute_words([word("untimed"), word("also")], A_ALONE)

    assert [w.text for w in words] == ["untimed", "also"]
    assert all(w.speaker is None and not w.aligned for w in words)


class TestSeams:
    def test_words_owned_by_the_neighbouring_chunk_are_dropped(self):
        words = speech("w", 0.0, 10)  # 0.0-4.0s
        kept = keep_owned(words, own_start=1.0, own_end=3.0)

        midpoints = [(float(w.start) + float(w.end)) / 2 for w in kept]
        assert midpoints and all(1.0 <= point < 3.0 for point in midpoints)

    def test_a_seam_word_is_kept_by_exactly_one_side(self):
        words = speech("w", 0.0, 20)
        left = keep_owned(list(words), own_start=0.0, own_end=4.0)
        right = keep_owned(list(words), own_start=4.0, own_end=8.0)

        texts = [w.text for w in left] + [w.text for w in right]
        assert len(texts) == len(set(texts))

    def test_unaligned_words_survive_seam_filtering(self):
        words = [word("untimed"), *speech("w", 0.0, 4)]
        kept = keep_owned(words, own_start=10.0, own_end=20.0)

        assert [w.text for w in kept] == ["untimed"]


class TestGrouping:
    def test_consecutive_same_speaker_words_form_one_segment(self):
        words = attribute_words(speech("w", 1.0, 5), A_ALONE)
        segments = group_segments(words)

        assert len(segments) == 1
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[0].raw_text == "w0 w1 w2 w3 w4"
        assert segments[0].start == pytest.approx(1.0)

    def test_attribution_change_starts_a_new_segment(self):
        words = attribute_words(
            [word("keep", 9.4, 9.8), word("yes", 10.1, 10.5), word("going", 10.9, 11.3)],
            CROSSTALK,
        )
        segments = group_segments(words)

        assert [segment.raw_text for segment in segments] == ["keep", "yes", "going"]
        assert segments[1].overlap is True
        assert segments[1].speaker is None

    def test_a_long_pause_splits_a_segment(self):
        words = attribute_words([word("before", 1.0, 1.4), word("after", 5.0, 5.4)], A_ALONE)
        segments = group_segments(words)

        assert [segment.raw_text for segment in segments] == ["before", "after"]

    def test_unaligned_segments_carry_no_timestamps(self):
        segments = group_segments(attribute_words([word("untimed"), word("run")], A_ALONE))

        assert len(segments) == 1
        assert segments[0].start is None and segments[0].end is None
        assert segments[0].aligned is False
        assert segments[0].raw_text == "untimed run"

    def test_clean_text_starts_as_raw_text(self):
        segments = group_segments(attribute_words(speech("w", 0.0, 3), A_ALONE))

        assert segments[0].clean_text == segments[0].raw_text
