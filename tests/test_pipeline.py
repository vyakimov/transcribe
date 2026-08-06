"""End-to-end wiring, with every hosted boundary stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcribe_pipeline import alignment, asr, cleanup, diarization, media, pipeline
from transcribe_pipeline.models import Chunk, DiarizationResult, MediaInfo, Turn, Word
from transcribe_pipeline.pipeline import Options

# Two chunks joined at a hard boundary: each carries five seconds of the other's
# audio, and only one of them owns the words in that overlap.
CHUNK_ONE = Chunk(path=Path("chunk-1.flac"), audio_start=0.0, own_start=0.0, own_end=20.0)
CHUNK_TWO = Chunk(path=Path("chunk-2.flac"), audio_start=15.0, own_start=20.0, own_end=40.0)

TURNS = (Turn("SPEAKER_00", 0.0, 25.0), Turn("SPEAKER_01", 25.0, 40.0))

# "c" straddles the seam and is transcribed by both chunks.
CHUNK_ONE_WORDS = [Word("a", 1.0, 1.4), Word("b", 19.0, 19.4), Word("c", 21.0, 21.4)]
CHUNK_TWO_WORDS = [Word("b", 19.0, 19.4), Word("c", 21.0, 21.4), Word("d", 30.0, 30.4)]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Replace ffmpeg and every hosted call with deterministic stand-ins."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-pyannote")

    state: dict = {
        "chunks": [CHUNK_ONE, CHUNK_TWO],
        "texts": ["a b c", "b c d"],
        "turns": TURNS,
        "words": {0: CHUNK_ONE_WORDS, 1: CHUNK_TWO_WORDS},
        "align_calls": [],
        "cleanup_calls": 0,
    }

    monkeypatch.setattr(media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        media, "preprocess", lambda source, destination, denoise: destination.write_bytes(b"x")
    )
    monkeypatch.setattr(media, "probe", lambda path: MediaInfo(duration=40.0, size=1000))
    monkeypatch.setattr(media, "extract_chunks", lambda source, workspace: state["chunks"])
    monkeypatch.setattr(pipeline, "OpenAI", lambda api_key: object())

    def fake_transcribe(client, chunks, language=None):
        return [
            asr.ChunkTranscript(chunk=chunk, text=text, raw_response={"text": text})
            for chunk, text in zip(chunks, state["texts"])
        ]

    def fake_diarize(source, *, api_key, num_speakers=None):
        return DiarizationResult(turns=state["turns"])

    def fake_align(audio, requests, *, language=None, device=None):
        state["align_calls"].append(requests)
        words = {request.id: list(state["words"].get(request.id, [])) for request in requests}
        return alignment.AlignmentResult(words_by_id=words, warnings=["synthetic warning"])

    def fake_clean(client, segments):
        state["cleanup_calls"] += 1
        for segment in segments:
            segment.clean_text = segment.raw_text.upper()
        return []

    monkeypatch.setattr(asr, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(diarization, "diarize", fake_diarize)
    monkeypatch.setattr(alignment, "align", fake_align)
    monkeypatch.setattr(cleanup, "clean_segments", fake_clean)
    return state


def options(**overrides) -> Options:
    defaults = dict(audio=Path("input/interview.m4a"), language="en")
    defaults.update(overrides)
    return Options(**defaults)


class TestSeamStitching:
    def test_a_word_spanning_the_seam_appears_exactly_once(self, stubbed):
        document, _ = pipeline.run(options())

        text = " ".join(segment.raw_text for segment in document.segments)
        assert text.split() == ["a", "b", "c", "d"]

    def test_stitched_words_keep_their_speakers(self, stubbed):
        document, _ = pipeline.run(options())

        attributed = [(segment.raw_text, segment.speaker) for segment in document.segments]
        assert attributed == [
            ("a", "SPEAKER_00"),
            ("b", "SPEAKER_00"),
            ("c", "SPEAKER_00"),
            ("d", "SPEAKER_01"),
        ]

    def test_alignment_receives_each_chunk_with_its_own_text_and_offset(self, stubbed):
        pipeline.run(options())
        requests = stubbed["align_calls"][0]

        assert [request.text for request in requests] == ["a b c", "b c d"]
        assert [request.start for request in requests] == [0.0, 15.0]


class TestDocument:
    def test_models_used_are_recorded(self, stubbed):
        document, _ = pipeline.run(options())

        assert document.models["transcription"] == "gpt-transcribe"
        assert document.models["diarization"] == "precision-2"
        assert document.models["cleanup"] == "gpt-5.6-terra"

    def test_speakers_come_from_diarization(self, stubbed):
        document, _ = pipeline.run(options())

        assert document.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_alignment_warnings_reach_the_document(self, stubbed):
        document, _ = pipeline.run(options())

        assert "synthetic warning" in document.warnings

    def test_raw_responses_are_returned_before_cleanup(self, stubbed):
        _, raw = pipeline.run(options())

        assert [entry["response"]["text"] for entry in raw] == ["a b c", "b c d"]

    def test_cleanup_updates_clean_text_only(self, stubbed):
        document, _ = pipeline.run(options())

        assert stubbed["cleanup_calls"] == 1
        for segment in document.segments:
            assert segment.clean_text == segment.raw_text.upper()


class TestSkipFlags:
    def test_no_clean_leaves_clean_text_equal_to_raw_text(self, stubbed):
        document, _ = pipeline.run(options(clean=False))

        assert stubbed["cleanup_calls"] == 0
        for segment in document.segments:
            assert segment.clean_text == segment.raw_text
        assert any("cleanup pass was skipped" in warning for warning in document.warnings)

    def test_no_align_skips_diarization_and_keeps_the_text(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        document, _ = pipeline.run(options(align=False, clean=False))

        text = " ".join(segment.raw_text for segment in document.segments)
        assert text.split() == ["a", "b", "c", "b", "c", "d"]
        assert document.speakers == []
        assert document.models["diarization"] == "none"
        assert any("--no-align" in warning for warning in document.warnings)

    def test_no_align_segments_are_unattributed_and_untimed(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        document, _ = pipeline.run(options(align=False, clean=False))

        assert all(segment.speaker is None for segment in document.segments)
        assert all(segment.start is None for segment in document.segments)


class TestMissingCredentials:
    def test_a_missing_openai_key_fails_with_a_clear_message(self, stubbed, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            pipeline.run(options())

    def test_a_missing_pyannote_key_names_the_variable(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        with pytest.raises(SystemExit, match="PYANNOTE_API_KEY"):
            pipeline.run(options())


class TestDegradedAlignment:
    def test_a_chunk_with_no_aligned_words_keeps_its_text(self, stubbed):
        stubbed["words"] = {0: CHUNK_ONE_WORDS}

        document, _ = pipeline.run(options(clean=False))

        text = " ".join(segment.raw_text for segment in document.segments)
        assert text.split() == ["a", "b", "b", "c", "d"]
        assert any("no aligned words" in warning for warning in document.warnings)

    def test_unattributed_words_are_counted_in_the_warnings(self, stubbed):
        stubbed["turns"] = (Turn("SPEAKER_00", 0.0, 2.0),)

        document, _ = pipeline.run(options(clean=False))

        assert any("could not be attributed" in warning for warning in document.warnings)
