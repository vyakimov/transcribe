"""The local Community-1 boundary: the worker's extraction and the driver's parsing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe_pipeline import attribution, local_diarization
from transcribe_pipeline.config import (
    DIARIZATION_BACKEND_COMMUNITY,
    DIARIZATION_BACKEND_PYANNOTE,
    diarization_model_name,
    resolve_diarization_backend,
    validate_diarization_backend,
)
from transcribe_pipeline.local_diarization import (
    LocalDiarizationError,
    default_command,
    parse_worker_output,
)
from transcribe_pipeline.models import Word

WORKER_PATH = Path(__file__).resolve().parent.parent / "workers" / "diarize_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("diarize_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("diarize_worker", module)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def segment(start: float, end: float):
    return SimpleNamespace(start=start, end=end)


class FakeAnnotation:
    """Stands in for pyannote's `Annotation`, whose stable API is `itertracks`."""

    def __init__(self, rows: list[tuple[float, float, str]]):
        self.rows = rows

    def itertracks(self, yield_label: bool = False):
        for index, (start, end, label) in enumerate(self.rows):
            yield (segment(start, end), f"_{index}", label) if yield_label else segment(start, end)


class PairAnnotation:
    """A build that yields (segment, label) pairs and offers no `itertracks`."""

    def __init__(self, rows: list[tuple[float, float, str]]):
        self.rows = rows

    def __iter__(self):
        for start, end, label in self.rows:
            yield segment(start, end), label


# --- worker extraction ----------------------------------------------------------


class TestExtractTurns:
    def test_tracks_become_ordered_turns(self):
        annotation = FakeAnnotation(
            [(5.0, 6.0, "SPEAKER_01"), (0.5, 2.25, "SPEAKER_00")]
        )

        turns = worker.extract_turns(annotation)

        assert turns == [
            {"speaker": "SPEAKER_00", "start": 0.5, "end": 2.25},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 6.0},
        ]

    def test_pair_yielding_annotations_are_supported(self):
        turns = worker.extract_turns(PairAnnotation([(0.0, 1.0, "SPEAKER_00")]))

        assert turns == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]

    def test_empty_segments_are_skipped(self):
        annotation = FakeAnnotation([(1.0, 1.0, "SPEAKER_00"), (2.0, 1.5, "SPEAKER_00")])

        assert worker.extract_turns(annotation) == []

    def test_a_missing_annotation_is_empty(self):
        assert worker.extract_turns(None) == []


# --- driver parsing -------------------------------------------------------------


class TestAudioDecoding:
    """The worker decodes with ffmpeg so pyannote never reaches for torchcodec,
    whose prebuilt libraries are pinned to one ffmpeg ABI."""

    @staticmethod
    def _stub_torch(monkeypatch):
        """A torch stand-in: this environment runs 3.14, where torch cannot install."""

        class FakeTensor:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data),)

            def reshape(self, *shape):
                self.shape = shape
                return self

        fake = SimpleNamespace(
            float32="float32",
            frombuffer=lambda buffer, dtype: FakeTensor(bytes(buffer)),
        )
        monkeypatch.setitem(sys.modules, "torch", fake)

    def test_the_waveform_is_mono_and_at_the_models_sample_rate(self, monkeypatch, tmp_path):
        self._stub_torch(monkeypatch)
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"x")
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout=b"\x00" * 16, stderr=b"")

        monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(worker.subprocess, "run", fake_run)

        result = worker.decode_audio(str(audio))

        assert result["sample_rate"] == worker.PYANNOTE_SAMPLE_RATE
        assert result["waveform"].shape == (1, -1)
        assert result["uri"] == "a"
        assert commands[0][0] == "ffmpeg"
        assert "-ac" in commands[0] and commands[0][commands[0].index("-ac") + 1] == "1"
        assert commands[0][commands[0].index("-ar") + 1] == str(worker.PYANNOTE_SAMPLE_RATE)

    def test_a_missing_ffmpeg_is_named(self, monkeypatch, tmp_path):
        self._stub_torch(monkeypatch)
        monkeypatch.setattr(worker.shutil, "which", lambda name: None)

        with pytest.raises(SystemExit, match="ffmpeg is required"):
            worker.decode_audio(str(tmp_path / "a.flac"))

    def test_a_missing_file_is_named(self, monkeypatch, tmp_path):
        self._stub_torch(monkeypatch)
        monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/ffmpeg")

        with pytest.raises(SystemExit, match="Audio file not found"):
            worker.decode_audio(str(tmp_path / "missing.flac"))

    def test_a_decode_failure_carries_ffmpegs_own_message(self, monkeypatch, tmp_path):
        self._stub_torch(monkeypatch)
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"x")
        monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            worker.subprocess,
            "run",
            lambda command, **kwargs: SimpleNamespace(
                returncode=1, stdout=b"", stderr=b"Invalid data found"
            ),
        )

        with pytest.raises(SystemExit, match="Invalid data found"):
            worker.decode_audio(str(audio))

    def test_silence_is_not_mistaken_for_success(self, monkeypatch, tmp_path):
        self._stub_torch(monkeypatch)
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"x")
        monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            worker.subprocess,
            "run",
            lambda command, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
        )

        with pytest.raises(SystemExit, match="no audio samples"):
            worker.decode_audio(str(audio))


class TestParseWorkerOutput:
    def test_regular_and_exclusive_turns_are_both_kept(self):
        result = parse_worker_output(
            {
                "diarization": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                    {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
                ],
                "exclusive_diarization": [
                    {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0}
                ],
            }
        )

        assert [turn.speaker for turn in result.turns] == ["SPEAKER_00", "SPEAKER_01"]
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
        assert [turn.speaker for turn in result.exclusive_turns] == ["SPEAKER_01"]

    def test_community_reports_no_confidence(self):
        result = parse_worker_output(
            {"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]}
        )

        assert result.turns[0].confidence is None

    def test_a_missing_exclusive_timeline_is_tolerated(self):
        result = parse_worker_output(
            {"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]}
        )

        assert result.exclusive_turns == ()

    def test_an_empty_payload_yields_no_turns(self):
        assert parse_worker_output({}).turns == ()


class TestExclusiveTurnsResolveOverlap:
    """Community-1's exclusive timeline must break ties the same way precision-2's does."""

    def test_a_contested_word_is_resolved_by_the_exclusive_timeline(self):
        result = parse_worker_output(
            {
                "diarization": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.4},
                    {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
                ],
                "exclusive_diarization": [
                    {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0}
                ],
            }
        )

        word = attribution.attribute_word(
            Word("yes", 10.0, 10.4), result.turns, result.exclusive_turns
        )

        assert word.overlap is True
        assert word.speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]
        assert word.speaker == "SPEAKER_01"

    def test_without_an_exclusive_timeline_the_tie_stays_open(self):
        result = parse_worker_output(
            {
                "diarization": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.4},
                    {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
                ]
            }
        )

        word = attribution.attribute_word(
            Word("yes", 10.0, 10.4), result.turns, result.exclusive_turns
        )

        assert word.overlap is True
        assert word.speaker is None


# --- command construction -------------------------------------------------------


class TestDefaultCommand:
    def test_an_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_DIARIZE_CMD", "/usr/bin/python3 -X dev worker.py")

        assert default_command() == ["/usr/bin/python3", "-X", "dev", "worker.py"]

    def test_a_prepared_environment_is_used_directly(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TRANSCRIBE_DIARIZE_CMD", raising=False)
        prepared = tmp_path / "python"
        prepared.write_text("")
        monkeypatch.setattr(local_diarization, "diarize_venv_python", lambda: prepared)

        command = default_command()

        assert command[0] == str(prepared)
        assert command[1].endswith("diarize_worker.py")

    def test_without_a_prepared_environment_uv_builds_one_on_demand(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_DIARIZE_CMD", raising=False)
        monkeypatch.setattr(local_diarization, "diarize_venv_python", lambda: None)
        monkeypatch.setattr(local_diarization.shutil, "which", lambda name: "/usr/local/bin/uv")

        command = default_command()

        assert command[:3] == ["uv", "run", "--no-project"]
        assert "pyannote.audio" in command
        assert "3.12" in command

    def test_a_missing_uv_names_the_setup_script(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_DIARIZE_CMD", raising=False)
        monkeypatch.setattr(local_diarization, "diarize_venv_python", lambda: None)
        monkeypatch.setattr(local_diarization.shutil, "which", lambda name: None)

        with pytest.raises(LocalDiarizationError, match="setup_diarization.py"):
            default_command()


# --- driving the subprocess -----------------------------------------------------


@pytest.fixture
def worker_run(monkeypatch):
    """Capture the request the worker would receive and reply with a canned payload."""
    monkeypatch.setenv("TRANSCRIBE_DIARIZE_CMD", "/bin/false")

    state: dict = {
        "requests": [],
        "returncode": 0,
        "response": {
            "diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}],
            "exclusive_diarization": [],
            "warnings": [],
        },
    }

    def fake_run(command, check=False, text=True):
        request_path, response_path = Path(command[-2]), Path(command[-1])
        state["requests"].append(json.loads(request_path.read_text(encoding="utf-8")))
        if state["returncode"] == 0 and state["response"] is not None:
            response_path.write_text(json.dumps(state["response"]), encoding="utf-8")
        return SimpleNamespace(returncode=state["returncode"])

    monkeypatch.setattr(local_diarization.subprocess, "run", fake_run)
    return state


class TestDiarize:
    def test_a_known_speaker_count_is_forwarded(self, worker_run):
        local_diarization.diarize(Path("audio.flac"), num_speakers=3)

        assert worker_run["requests"][0]["num_speakers"] == 3

    def test_an_unknown_speaker_count_is_left_null(self, worker_run):
        local_diarization.diarize(Path("audio.flac"))

        assert worker_run["requests"][0]["num_speakers"] is None

    def test_an_explicitly_selected_device_is_forwarded(self, worker_run):
        local_diarization.diarize(Path("audio.flac"), device="mps")

        assert worker_run["requests"][0]["device"] == "mps"

    def test_no_device_leaves_the_workers_cpu_default_in_place(self, worker_run):
        local_diarization.diarize(Path("audio.flac"))

        assert worker_run["requests"][0]["device"] is None

    def test_the_community_model_is_requested(self, worker_run):
        local_diarization.diarize(Path("audio.flac"))

        assert worker_run["requests"][0]["model"] == "pyannote/speaker-diarization-community-1"

    def test_turns_come_back_as_a_diarization_result(self, worker_run):
        result = local_diarization.diarize(Path("audio.flac"))

        assert result.speakers == ["SPEAKER_00"]

    def test_worker_warnings_are_surfaced(self, worker_run, capsys):
        worker_run["response"]["warnings"] = ["no exclusive timeline"]

        local_diarization.diarize(Path("audio.flac"))

        assert "no exclusive timeline" in capsys.readouterr().err

    def test_a_failing_worker_points_at_the_alternatives(self, worker_run):
        worker_run["returncode"] = 1

        with pytest.raises(LocalDiarizationError, match="setup_diarization.py"):
            local_diarization.diarize(Path("audio.flac"))

    def test_a_silent_worker_is_an_error(self, worker_run):
        worker_run["response"] = None

        with pytest.raises(LocalDiarizationError, match="no output file"):
            local_diarization.diarize(Path("audio.flac"))

    def test_an_empty_result_is_an_error_rather_than_a_speakerless_transcript(self, worker_run):
        worker_run["response"] = {"diarization": [], "exclusive_diarization": []}

        with pytest.raises(LocalDiarizationError, match="no speaker turns"):
            local_diarization.diarize(Path("audio.flac"))

    def test_a_missing_worker_script_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setattr(local_diarization, "DIARIZE_WORKER", tmp_path / "absent.py")

        with pytest.raises(LocalDiarizationError, match="not found"):
            local_diarization.diarize(Path("audio.flac"))


# --- backend selection ----------------------------------------------------------


class TestBackendSelection:
    def test_the_local_backend_is_the_default(self, monkeypatch):
        monkeypatch.delenv("DIARIZATION_BACKEND", raising=False)

        assert resolve_diarization_backend(None) == DIARIZATION_BACKEND_COMMUNITY

    def test_the_environment_selects_a_backend(self, monkeypatch):
        monkeypatch.setenv("DIARIZATION_BACKEND", "pyannote")

        assert resolve_diarization_backend(None) == DIARIZATION_BACKEND_PYANNOTE

    def test_the_cli_overrides_the_environment(self, monkeypatch):
        monkeypatch.setenv("DIARIZATION_BACKEND", "community")

        assert resolve_diarization_backend("pyannote") == DIARIZATION_BACKEND_PYANNOTE

    def test_names_are_case_insensitive_and_trimmed(self, monkeypatch):
        monkeypatch.setenv("DIARIZATION_BACKEND", "  Community ")

        assert resolve_diarization_backend(None) == DIARIZATION_BACKEND_COMMUNITY

    def test_an_unknown_cli_value_names_the_flag_and_the_choices(self, monkeypatch):
        monkeypatch.delenv("DIARIZATION_BACKEND", raising=False)

        with pytest.raises(SystemExit, match="--diarization-backend"):
            resolve_diarization_backend("whisper")

    def test_an_unknown_environment_value_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("DIARIZATION_BACKEND", "whisper")

        with pytest.raises(SystemExit, match="DIARIZATION_BACKEND"):
            resolve_diarization_backend(None)

    def test_an_empty_environment_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("DIARIZATION_BACKEND", "   ")

        assert resolve_diarization_backend(None) == DIARIZATION_BACKEND_COMMUNITY

    def test_a_bad_backend_is_rejected_wherever_it_came_from(self):
        with pytest.raises(SystemExit, match="pyannote, community"):
            validate_diarization_backend("nope", source="somewhere")


class TestModelNames:
    def test_each_backend_reports_its_own_model(self):
        assert diarization_model_name(DIARIZATION_BACKEND_PYANNOTE) == "precision-2"
        assert diarization_model_name(DIARIZATION_BACKEND_COMMUNITY) == "community-1"

    def test_an_unknown_backend_has_no_model_name(self):
        with pytest.raises(SystemExit):
            diarization_model_name("whisper")
