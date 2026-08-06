"""The pyannoteAI boundary: response parsing and job handling."""

from __future__ import annotations

import httpx
import pytest

from transcribe_pipeline import diarization
from transcribe_pipeline.diarization import DiarizationError, parse_turns
from transcribe_pipeline.models import DiarizationResult


class TestParseTurns:
    def test_turns_are_parsed_and_sorted(self):
        turns = parse_turns(
            [
                {"speaker": "SPEAKER_01", "start": 5.0, "end": 6.0},
                {"speaker": "SPEAKER_00", "start": 0.5, "end": 2.3},
            ]
        )

        assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]
        assert turns[0].start == 0.5 and turns[0].end == 2.3

    def test_percentage_confidence_is_normalized_to_a_fraction(self):
        turns = parse_turns([{"speaker": "SPEAKER_00", "start": 0, "end": 1, "confidence": 93}])

        assert turns[0].confidence == pytest.approx(0.93)

    def test_per_speaker_confidence_map_is_resolved(self):
        turns = parse_turns(
            [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 1,
                    "confidence": {"SPEAKER_00": 88, "SPEAKER_01": 12},
                }
            ]
        )

        assert turns[0].confidence == pytest.approx(0.88)

    def test_fractional_confidence_is_left_alone(self):
        turns = parse_turns([{"speaker": "SPEAKER_00", "start": 0, "end": 1, "confidence": 0.95}])

        assert turns[0].confidence == pytest.approx(0.95)

    def test_missing_confidence_is_none(self):
        turns = parse_turns([{"speaker": "SPEAKER_00", "start": 0, "end": 1}])

        assert turns[0].confidence is None

    @pytest.mark.parametrize(
        "entry",
        [
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 5.0},
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 4.0},
            {"speaker": "SPEAKER_00", "start": "x", "end": 4.0},
            {"start": 1.0, "end": 4.0},
            "not a dict",
        ],
    )
    def test_malformed_entries_are_skipped(self, entry):
        assert parse_turns([entry]) == ()

    def test_non_list_input_is_empty(self):
        assert parse_turns(None) == ()
        assert parse_turns({"diarization": []}) == ()


class TestDiarizationResult:
    def test_speakers_are_unique_and_sorted(self):
        result = DiarizationResult(
            turns=parse_turns(
                [
                    {"speaker": "SPEAKER_01", "start": 0, "end": 1},
                    {"speaker": "SPEAKER_00", "start": 1, "end": 2},
                    {"speaker": "SPEAKER_01", "start": 2, "end": 3},
                ]
            )
        )

        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]


class TestJobPolling:
    def _client(self, responses: list[httpx.Response]) -> httpx.Client:
        pending = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            return pending.pop(0)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_polling_waits_for_success(self, monkeypatch):
        monkeypatch.setattr(diarization.time, "sleep", lambda _: None)
        client = self._client(
            [
                httpx.Response(200, json={"status": "running"}),
                httpx.Response(
                    200,
                    json={
                        "status": "succeeded",
                        "output": {"diarization": [{"speaker": "S0", "start": 0, "end": 1}]},
                    },
                ),
            ]
        )

        output = diarization._await_job(client, "job-1")

        assert output["diarization"][0]["speaker"] == "S0"

    def test_a_failed_job_raises_with_the_reported_error(self, monkeypatch):
        monkeypatch.setattr(diarization.time, "sleep", lambda _: None)
        client = self._client(
            [httpx.Response(200, json={"status": "failed", "output": {"error": "bad audio"}})]
        )

        with pytest.raises(DiarizationError, match="bad audio"):
            diarization._await_job(client, "job-1")

    def test_an_http_error_raises(self, monkeypatch):
        monkeypatch.setattr(diarization.time, "sleep", lambda _: None)
        client = self._client([httpx.Response(401, text="unauthorized")])

        with pytest.raises(DiarizationError, match="401"):
            diarization._await_job(client, "job-1")


class TestSubmitJob:
    def test_known_speaker_count_is_forwarded(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"jobId": "job-9", "status": "created"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        job_id = diarization._submit_job(client, "media://x.flac", num_speakers=3)

        assert job_id == "job-9"
        assert captured["numSpeakers"] == 3
        assert captured["model"] == "precision-2"
        assert captured["confidence"] is True
        assert captured["exclusive"] is True

    def test_speaker_count_is_omitted_when_unknown(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"jobId": "job-9", "status": "created"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        diarization._submit_job(client, "media://x.flac", num_speakers=None)

        assert "numSpeakers" not in captured


def test_object_key_is_unique_and_url_safe(tmp_path):
    from pathlib import Path

    source = Path("my recording (final).flac")
    first = diarization._object_key(source)
    second = diarization._object_key(source)

    assert first != second
    assert " " not in first and "(" not in first
    assert first.endswith(".flac")
