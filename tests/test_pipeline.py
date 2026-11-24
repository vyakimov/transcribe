"""End-to-end wiring, with every hosted boundary stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcribe_pipeline import (
    alignment,
    asr,
    assemblyai_diarization,
    cleanup,
    diarization,
    local_diarization,
    media,
    pipeline,
)
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
    monkeypatch.delenv("DIARIZATION_BACKEND", raising=False)
    for name in ("ASSEMBLYAI_API_KEY", "ASSEMBLYAI_KEY"):
        monkeypatch.delenv(name, raising=False)

    state: dict = {
        "chunks": [CHUNK_ONE, CHUNK_TWO],
        "texts": ["a b c", "b c d"],
        "turns": TURNS,
        "words": {0: CHUNK_ONE_WORDS, 1: CHUNK_TWO_WORDS},
        "align_calls": [],
        "hosted_calls": [],
        "local_calls": [],
        "assemblyai_calls": [],
        "assemblyai_transcribe_calls": [],
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
        state["hosted_calls"].append({"api_key": api_key, "num_speakers": num_speakers})
        return DiarizationResult(turns=state["turns"])

    def fake_assemblyai_diarize(source, *, api_key, num_speakers=None, language=None):
        state["assemblyai_calls"].append(
            {"api_key": api_key, "num_speakers": num_speakers, "language": language}
        )
        return DiarizationResult(turns=state["turns"])

    monkeypatch.setattr(assemblyai_diarization, "diarize", fake_assemblyai_diarize)

    def fake_assemblyai_transcribe(
        source, *, api_key, num_speakers=None, language=None, speaker_labels=True
    ):
        state["assemblyai_transcribe_calls"].append(
            {
                "source": source,
                "api_key": api_key,
                "num_speakers": num_speakers,
                "language": language,
                "speaker_labels": speaker_labels,
            }
        )
        speaker = (lambda name: name) if speaker_labels else (lambda name: None)
        words = [
            Word("hello", 0.0, 0.4, speaker=speaker("SPEAKER_A")),
            Word("there", 0.5, 0.9, speaker=speaker("SPEAKER_A")),
            Word("hi", 1.2, 1.5, speaker=speaker("SPEAKER_B")),
        ]
        turns = (Turn("SPEAKER_A", 0.0, 0.9), Turn("SPEAKER_B", 1.2, 1.5)) if speaker_labels else ()
        body = {"speech_model_used": "aai-model", "language_code": "en"}
        return words, DiarizationResult(turns=turns, exclusive_turns=turns), body

    monkeypatch.setattr(assemblyai_diarization, "transcribe", fake_assemblyai_transcribe)

    def fake_local_diarize(source, *, num_speakers=None, device=None):
        state["local_calls"].append({"num_speakers": num_speakers, "device": device})
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
    monkeypatch.setattr(local_diarization, "diarize", fake_local_diarize)
    monkeypatch.setattr(alignment, "align", fake_align)
    monkeypatch.setattr(cleanup, "clean_segments", fake_clean)
    return state


def options(**overrides) -> Options:
    # These tests cover the multi-stage gpt path; TestAssemblyAITranscription
    # covers the single-job default.
    defaults = dict(
        audio=Path("input/interview.m4a"), language="en", transcription_backend="gpt"
    )
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
        assert document.models["diarization"] == "community-1"
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

    def test_no_align_bypasses_every_diarization_backend(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        for backend in ("pyannote", "community"):
            document, _ = pipeline.run(
                options(align=False, clean=False, diarization_backend=backend)
            )

            assert document.models["diarization"] == "none"

        assert stubbed["hosted_calls"] == []
        assert stubbed["local_calls"] == []


class TestDiarizationBackends:
    def test_the_community_backend_is_used_by_default(self, stubbed):
        document, _ = pipeline.run(options())

        assert len(stubbed["local_calls"]) == 1
        assert stubbed["hosted_calls"] == []
        assert document.models["diarization"] == "community-1"

    def test_the_hosted_backend_runs_remotely_and_is_recorded(self, stubbed):
        document, _ = pipeline.run(options(diarization_backend="pyannote"))

        assert len(stubbed["hosted_calls"]) == 1
        assert stubbed["local_calls"] == []
        assert document.models["diarization"] == "precision-2"

    @pytest.mark.parametrize("variable", ["ASSEMBLYAI_API_KEY", "ASSEMBLYAI_KEY"])
    def test_the_assemblyai_backend_runs_remotely_and_is_recorded(
        self, stubbed, monkeypatch, variable
    ):
        monkeypatch.setenv(variable, "test-assemblyai")

        document, _ = pipeline.run(options(diarization_backend="assemblyai", speakers=2))

        assert stubbed["assemblyai_calls"] == [
            {"api_key": "test-assemblyai", "num_speakers": 2, "language": "en"}
        ]
        assert stubbed["hosted_calls"] == [] and stubbed["local_calls"] == []
        assert document.models["diarization"] == "assemblyai-speaker-labels"

    def test_the_community_backend_keeps_the_rest_of_the_pipeline_intact(self, stubbed):
        document, _ = pipeline.run(options(diarization_backend="community"))

        attributed = [(segment.raw_text, segment.speaker) for segment in document.segments]
        assert attributed == [
            ("a", "SPEAKER_00"),
            ("b", "SPEAKER_00"),
            ("c", "SPEAKER_00"),
            ("d", "SPEAKER_01"),
        ]
        assert document.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_a_known_speaker_count_reaches_the_local_backend(self, stubbed):
        pipeline.run(options(diarization_backend="community", speakers=2))

        assert stubbed["local_calls"][0]["num_speakers"] == 2

    def test_the_diarization_device_reaches_the_local_backend(self, stubbed):
        pipeline.run(options(diarization_backend="community", diarization_device="cuda"))

        assert stubbed["local_calls"][0]["device"] == "cuda"

    def test_the_alignment_device_is_not_confused_with_the_diarization_device(self, stubbed):
        pipeline.run(options(diarization_backend="community", device="cpu"))

        assert stubbed["local_calls"][0]["device"] is None

    def test_an_unknown_backend_fails_before_any_work_is_done(self, stubbed):
        with pytest.raises(SystemExit, match="pyannote, community, assemblyai"):
            pipeline.run(options(diarization_backend="whisper"))

        assert stubbed["hosted_calls"] == []
        assert stubbed["local_calls"] == []


class TestMissingCredentials:
    def test_a_missing_openai_key_fails_with_a_clear_message(self, stubbed, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            pipeline.run(options())

    def test_a_missing_pyannote_key_names_the_variable(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        with pytest.raises(SystemExit, match="PYANNOTE_API_KEY"):
            pipeline.run(options(diarization_backend="pyannote"))

    def test_a_missing_assemblyai_key_names_the_variable(self, stubbed):
        with pytest.raises(SystemExit, match="ASSEMBLYAI_API_KEY"):
            pipeline.run(options(diarization_backend="assemblyai"))

        assert stubbed["assemblyai_calls"] == []

    def test_the_community_backend_needs_no_pyannote_key(self, stubbed, monkeypatch):
        monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)

        document, _ = pipeline.run(options(diarization_backend="community"))

        assert document.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_the_community_backend_still_needs_the_openai_key(self, stubbed, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            pipeline.run(options(diarization_backend="community"))


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


class TestAssemblyAITranscription:
    """The default path: one AssemblyAI job supplies words, timings, and speakers."""

    def _options(self, **overrides) -> Options:
        return Options(**{"audio": Path("input/interview.m4a"), **overrides})

    def test_it_is_the_default_and_needs_no_other_stage(self, stubbed, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_KEY", "test-assemblyai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        document, raw = pipeline.run(self._options(speakers=2))

        call = stubbed["assemblyai_transcribe_calls"][0]
        assert call["api_key"] == "test-assemblyai" and call["num_speakers"] == 2
        # The original recording is sent, not a preprocessed copy.
        assert call["source"] == Path("input/interview.m4a")
        assert stubbed["align_calls"] == [] and stubbed["cleanup_calls"] == 0
        assert stubbed["local_calls"] == [] and stubbed["hosted_calls"] == []
        assert [(s.speaker, s.raw_text) for s in document.segments] == [
            ("SPEAKER_A", "hello there"),
            ("SPEAKER_B", "hi"),
        ]
        assert document.speakers == ["SPEAKER_A", "SPEAKER_B"]
        assert document.language == "en"
        assert document.models == {
            "transcription": "aai-model",
            "diarization": "assemblyai-speaker-labels",
            "cleanup": "none",
        }
        assert raw[0]["response"]["speech_model_used"] == "aai-model"

    def test_no_align_skips_the_speaker_labels(self, stubbed, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-assemblyai")

        document, _ = pipeline.run(self._options(align=False))

        assert stubbed["assemblyai_transcribe_calls"][0]["speaker_labels"] is False
        assert document.speakers == []
        assert document.models["diarization"] == "none"
        assert any("--no-align" in warning for warning in document.warnings)

    def test_a_missing_key_names_the_variable(self, stubbed):
        with pytest.raises(SystemExit, match="ASSEMBLYAI_API_KEY"):
            pipeline.run(self._options())

    def test_another_diarizer_cannot_be_combined_with_it(self, stubbed, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-assemblyai")

        with pytest.raises(SystemExit, match="cannot be combined"):
            pipeline.run(self._options(diarization_backend="community"))

        assert stubbed["assemblyai_transcribe_calls"] == []
