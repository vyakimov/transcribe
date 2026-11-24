"""The AssemblyAI boundary: request shape, polling, and turn parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from transcribe_pipeline import assemblyai_diarization as aai
from transcribe_pipeline.diarization import DiarizationError


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestParseTurns:
    def test_milliseconds_become_seconds_and_speakers_are_prefixed(self):
        turns = aai.parse_turns(
            [
                {"speaker": "B", "start": 5000, "end": 6500, "confidence": 0.9},
                {"speaker": "A", "start": 250, "end": 2300},
            ]
        )

        assert [turn.speaker for turn in turns] == ["SPEAKER_A", "SPEAKER_B"]
        assert turns[0].start == 0.25 and turns[0].end == 2.3
        assert turns[0].confidence is None
        assert turns[1].confidence == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "entry",
        [
            {"speaker": "A", "start": 500, "end": 500},
            {"speaker": "A", "start": "x", "end": 900},
            {"speaker": None, "start": 0, "end": 900},
            {"start": 0, "end": 900},
            "not a dict",
        ],
    )
    def test_malformed_entries_are_skipped(self, entry):
        assert aai.parse_turns([entry]) == ()

    def test_non_list_input_is_empty(self):
        assert aai.parse_turns(None) == ()


class TestSubmitJob:
    def _submit(self, **kwargs) -> dict:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "job-9", "status": "queued"})

        assert aai._submit_job(_client(handler), "https://cdn/x", **kwargs) == "job-9"
        return captured

    def test_speaker_count_and_language_are_forwarded(self):
        captured = self._submit(num_speakers=3, language="en")

        assert captured["speaker_labels"] is True
        assert captured["speakers_expected"] == 3
        assert captured["language_code"] == "en"
        assert "language_detection" not in captured

    def test_unknown_language_and_speaker_count_are_left_to_assemblyai(self):
        captured = self._submit(num_speakers=None, language=None)

        assert "speakers_expected" not in captured
        assert "language_code" not in captured
        assert captured["language_detection"] is True


class TestJobPolling:
    def _poll(self, responses: list[httpx.Response]) -> dict:
        pending = list(responses)
        return aai._await_job(_client(lambda request: pending.pop(0)), "job-1")

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr(aai.time, "sleep", lambda _: None)

    def test_polling_waits_for_completion(self):
        body = self._poll(
            [
                httpx.Response(200, json={"status": "queued"}),
                httpx.Response(200, json={"status": "processing"}),
                httpx.Response(200, json={"status": "completed", "utterances": []}),
            ]
        )

        assert body["status"] == "completed"

    def test_a_failed_job_raises_with_the_reported_error(self):
        with pytest.raises(DiarizationError, match="bad audio"):
            self._poll([httpx.Response(200, json={"status": "error", "error": "bad audio"})])

    def test_an_http_error_raises_with_its_status(self):
        with pytest.raises(DiarizationError, match="HTTP 401"):
            self._poll([httpx.Response(401, json={"error": "unauthorized"})])


class TestDiarize:
    def _run(self, monkeypatch, tmp_path, final: dict):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path.endswith("/upload"):
                return httpx.Response(200, json={"upload_url": "https://cdn/upload/1"})
            if request.method == "POST":
                return httpx.Response(200, json={"id": "job-1", "status": "queued"})
            return httpx.Response(200, json=final)

        real_client = httpx.Client
        monkeypatch.setattr(
            aai.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        )
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"audio-bytes")
        return aai.diarize(audio, api_key="aai-key", num_speakers=2, language="en"), seen

    def test_utterances_become_both_timelines(self, monkeypatch, tmp_path):
        result, seen = self._run(
            monkeypatch,
            tmp_path,
            {
                "status": "completed",
                "utterances": [
                    {"speaker": "A", "start": 0, "end": 1000},
                    {"speaker": "B", "start": 1200, "end": 2000},
                ],
            },
        )

        assert result.speakers == ["SPEAKER_A", "SPEAKER_B"]
        assert result.exclusive_turns == result.turns
        # The bare key, no Bearer prefix, and the raw bytes as the upload body.
        assert all(request.headers["Authorization"] == "aai-key" for request in seen)
        assert seen[0].content == b"audio-bytes"
        assert json.loads(seen[1].content)["audio_url"] == "https://cdn/upload/1"

    def test_words_are_the_fallback_when_utterances_are_absent(self, monkeypatch, tmp_path):
        result, _ = self._run(
            monkeypatch,
            tmp_path,
            {
                "status": "completed",
                "utterances": None,
                "words": [{"speaker": "A", "start": 0, "end": 400, "text": "hi"}],
            },
        )

        assert result.speakers == ["SPEAKER_A"]

    def test_no_turns_is_an_error(self, monkeypatch, tmp_path):
        with pytest.raises(DiarizationError, match="no speaker turns"):
            self._run(monkeypatch, tmp_path, {"status": "completed", "utterances": []})


def test_the_first_key_variable_set_wins(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setenv("ASSEMBLYAI_KEY", "fallback")
    assert aai.api_key_from_environment() == "fallback"

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "canonical")
    assert aai.api_key_from_environment() == "canonical"


class TestTranscribe:
    BODY = {
        "status": "completed",
        "speech_model_used": "aai-model",
        "utterances": [
            {"speaker": "A", "start": 0, "end": 900},
            {"speaker": "B", "start": 1200, "end": 1500},
        ],
        "words": [
            {"text": "Hello", "start": 0, "end": 400, "confidence": 0.9, "speaker": "A"},
            {"text": "there.", "start": 500, "end": 900, "confidence": 0.8, "speaker": "A"},
            {"text": "Hi.", "start": 1200, "end": 1500, "confidence": 0.7, "speaker": "B"},
            {"text": " ", "start": 1500, "end": 1600},
        ],
    }

    def _run(self, monkeypatch, tmp_path, **kwargs):
        submitted: dict = {}

        def fake_run_job(source, **job):
            submitted.update(job)
            return self.BODY

        monkeypatch.setattr(aai, "_run_job", fake_run_job)
        return aai.transcribe(tmp_path / "a.m4a", api_key="k", **kwargs), submitted

    def test_words_keep_their_timings_and_speakers(self, monkeypatch, tmp_path):
        (words, result, body), submitted = self._run(monkeypatch, tmp_path, num_speakers=2)

        assert [(w.text, w.start, w.end, w.speaker) for w in words] == [
            ("Hello", 0.0, 0.4, "SPEAKER_A"),
            ("there.", 0.5, 0.9, "SPEAKER_A"),
            ("Hi.", 1.2, 1.5, "SPEAKER_B"),
        ]
        assert words[0].score == pytest.approx(0.9)
        assert result.speakers == ["SPEAKER_A", "SPEAKER_B"]
        assert body is self.BODY
        assert submitted["speaker_labels"] is True and submitted["num_speakers"] == 2

    def test_speaker_labels_can_be_switched_off(self, monkeypatch, tmp_path):
        (_, result, _), submitted = self._run(monkeypatch, tmp_path, speaker_labels=False)

        assert submitted["speaker_labels"] is False
        assert result.turns == ()

    def test_no_words_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(aai, "_run_job", lambda source, **job: {"status": "completed"})

        with pytest.raises(DiarizationError, match="no words"):
            aai.transcribe(tmp_path / "a.m4a", api_key="k")


def test_speakers_expected_is_not_sent_without_speaker_labels():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "job-9"})

    aai._submit_job(
        _client(handler), "https://cdn/x", num_speakers=2, language="en", speaker_labels=False
    )

    assert captured["speaker_labels"] is False
    assert "speakers_expected" not in captured
