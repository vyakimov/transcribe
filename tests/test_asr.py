"""The OpenAI transcription boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from transcribe_pipeline import asr
from transcribe_pipeline.config import TRANSCRIBE_MODEL
from transcribe_pipeline.models import Chunk


class StubResponse:
    def __init__(self, text: str):
        self.text = text

    def model_dump(self) -> dict:
        return {"text": self.text}


class StubClient:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls: list[dict] = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return StubResponse(self.texts.pop(0))


def chunk(tmp_path: Path, name: str) -> Chunk:
    path = tmp_path / name
    path.write_bytes(b"fake audio")
    return Chunk(path=path, audio_start=0.0, own_start=0.0, own_end=10.0)


def test_whole_chunks_are_transcribed_with_gpt_transcribe(tmp_path):
    client = StubClient(["hello there"])

    result = asr.transcribe_chunk(client, chunk(tmp_path, "a.flac"), language=None)

    assert result.text == "hello there"
    assert client.calls[0]["model"] == TRANSCRIBE_MODEL
    assert client.calls[0]["response_format"] == "json"


def test_diarized_output_is_never_requested(tmp_path):
    """Speaker labels come from pyannoteAI, never from the transcription model."""
    client = StubClient(["hello"])

    asr.transcribe_chunk(client, chunk(tmp_path, "a.flac"), language=None)

    assert client.calls[0]["response_format"] != "diarized_json"


def test_language_is_forwarded_when_given(tmp_path):
    client = StubClient(["hello"])

    asr.transcribe_chunk(client, chunk(tmp_path, "a.flac"), language="en")

    assert client.calls[0]["language"] == "en"


def test_language_is_omitted_when_not_given(tmp_path):
    client = StubClient(["hello"])

    asr.transcribe_chunk(client, chunk(tmp_path, "a.flac"), language=None)

    assert "language" not in client.calls[0]


def test_the_raw_response_is_kept_before_any_cleanup(tmp_path):
    client = StubClient(["  hello there  "])

    result = asr.transcribe_chunk(client, chunk(tmp_path, "a.flac"), language=None)

    assert result.raw_response == {"text": "  hello there  "}
    assert result.text == "hello there"


def test_every_chunk_is_transcribed_once(tmp_path):
    chunks = [chunk(tmp_path, "a.flac"), chunk(tmp_path, "b.flac")]
    client = StubClient(["first", "second"])

    results = asr.transcribe_chunks(client, chunks)

    assert [result.text for result in results] == ["first", "second"]
    assert len(client.calls) == 2
