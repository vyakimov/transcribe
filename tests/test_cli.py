"""The machine-callable contract: one JSON object on stdout, always.

These tests exist because the CLI's callers are programs. A stray print, a
renamed error type, or a schema that shifts between actions breaks them
silently, so each of those is pinned here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import transcribe_cli
from transcribe_cli import (
    AUTHENTICATION_FAILED,
    DEPENDENCY_MISSING,
    INTERNAL_ERROR,
    INVALID_ARGUMENTS,
    RATE_LIMITED,
    RESOURCE_NOT_FOUND,
    TIMEOUT,
    UPSTREAM_ERROR,
    main,
)

ROOT = Path(__file__).resolve().parent.parent
EXECUTABLE = ROOT / "bin" / "transcribe"


@pytest.fixture
def isolated_env(monkeypatch):
    """Stop `.env` from leaking real credentials into credential tests."""
    monkeypatch.setattr(transcribe_cli, "load_environment", lambda: None)
    for name in ("OPENAI_API_KEY", "PYANNOTE_API_KEY", "HF_TOKEN", "DIARIZATION_BACKEND"):
        monkeypatch.delenv(name, raising=False)


def invoke(capfd, argv: list[str]) -> tuple[int, dict, str]:
    """Run the CLI and return its exit code, parsed stdout, and raw stderr."""
    code = main(argv)
    captured = capfd.readouterr()
    return code, json.loads(captured.out), captured.err


# --- envelope -------------------------------------------------------------------


class TestEnvelope:
    def test_success_has_a_stable_shape(self, capfd):
        code, payload, _ = invoke(capfd, ["list-actions"])

        assert code == 0
        assert payload["ok"] is True
        assert payload["action"] == "list-actions"
        assert "result" in payload and "error" not in payload
        assert payload["meta"]["schema_version"] == transcribe_cli.SCHEMA_VERSION

    def test_failure_has_a_stable_shape(self, capfd):
        code, payload, _ = invoke(capfd, ["frobnicate"])

        assert code == transcribe_cli.USAGE_EXIT_CODE
        assert payload["ok"] is False
        assert payload["action"] == "frobnicate"
        assert "error" in payload and "result" not in payload
        assert set(payload["error"]) == {"type", "message", "details"}

    def test_a_malformed_call_and_a_failed_one_exit_differently(self, capfd, tmp_path):
        """The whole point of exit 2: telling "called wrong" from "work failed"."""
        usage, _, _ = invoke(capfd, ["frobnicate"])
        failure, payload, _ = invoke(
            capfd, ["probe", "--audio", str(tmp_path / "absent.m4a")]
        )

        assert usage == transcribe_cli.USAGE_EXIT_CODE
        assert failure == transcribe_cli.FAILURE_EXIT_CODE
        assert payload["error"]["type"] == RESOURCE_NOT_FOUND

    def test_version_reports_tool_and_schema_as_json(self, capfd):
        code, payload, _ = invoke(capfd, ["--version"])

        assert code == 0
        assert payload["ok"] is True
        assert payload["action"] == "version"
        assert payload["result"]["version"] == transcribe_cli.tool_version()
        assert payload["result"]["schema_version"] == transcribe_cli.SCHEMA_VERSION

    def test_version_matches_the_declared_project_version(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            declared = tomllib.load(handle)["project"]["version"]

        assert transcribe_cli.tool_version() == declared
        assert transcribe_cli.tool_version() != transcribe_cli.FALLBACK_TOOL_VERSION

    def test_every_action_reports_the_same_top_level_keys(self, capfd, tmp_path):
        shapes = set()
        for argv in (["list-actions"], ["check-setup"], ["frobnicate"], []):
            _, payload, _ = invoke(capfd, argv)
            shapes.add(frozenset(payload) - {"result", "error"})

        assert shapes == {frozenset({"ok", "action", "meta"})}

    def test_stdout_carries_nothing_but_the_object(self, capfd):
        main(["list-actions"])
        out = capfd.readouterr().out

        assert out.lstrip().startswith("{")
        assert out.rstrip().endswith("}")
        json.loads(out)


# --- self description -----------------------------------------------------------


class TestListActions:
    def test_every_action_is_catalogued(self, capfd):
        _, payload, _ = invoke(capfd, ["list-actions"])
        names = {action["name"] for action in payload["result"]["actions"]}

        assert names == {
            "list-actions",
            "check-setup",
            "probe",
            "transcribe",
            "job-status",
            "job-list",
            "job-cancel",
        }

    def test_actions_declare_whether_they_mutate(self, capfd):
        _, payload, _ = invoke(capfd, ["list-actions"])
        mutating = {a["name"]: a["mutating"] for a in payload["result"]["actions"]}

        assert mutating["transcribe"] is True
        assert mutating["probe"] is False

    def test_parameters_are_described_with_their_constraints(self, capfd):
        _, payload, _ = invoke(capfd, ["list-actions"])
        action = next(a for a in payload["result"]["actions"] if a["name"] == "transcribe")
        params = {p["name"]: p for p in action["parameters"]}

        assert params["audio"]["required"] is True
        assert params["diarization_backend"]["choices"] == ["pyannote", "community"]
        assert params["audio"]["description"]

    def test_the_error_catalog_is_published(self, capfd):
        _, payload, _ = invoke(capfd, ["list-actions"])

        assert INVALID_ARGUMENTS in payload["result"]["error_types"]
        assert DEPENDENCY_MISSING in payload["result"]["error_types"]


# --- input validation -----------------------------------------------------------


class TestValidation:
    def test_an_unknown_action_is_rejected_and_lists_the_known_ones(self, capfd):
        _, payload, _ = invoke(capfd, ["frobnicate"])

        assert payload["error"]["type"] == INVALID_ARGUMENTS
        assert "transcribe" in payload["error"]["details"]["known_actions"]

    def test_no_action_is_rejected(self, capfd):
        code, payload, _ = invoke(capfd, [])

        assert code == transcribe_cli.USAGE_EXIT_CODE
        assert payload["error"]["type"] == INVALID_ARGUMENTS

    def test_an_unknown_flag_is_rejected_rather_than_ignored(self, capfd):
        _, payload, _ = invoke(capfd, ["probe", "--audio", "x.m4a", "--bogus", "1"])

        assert payload["error"]["type"] == INVALID_ARGUMENTS

    def test_an_unknown_backend_is_rejected_by_choices(self, capfd):
        _, payload, _ = invoke(
            capfd, ["transcribe", "--audio", "x.m4a", "--diarization-backend", "whisper"]
        )

        assert payload["error"]["type"] == INVALID_ARGUMENTS
        assert "pyannote" in payload["error"]["message"]

    def test_a_missing_required_parameter_is_rejected(self, capfd):
        _, payload, _ = invoke(capfd, ["probe"])

        assert payload["error"]["type"] == INVALID_ARGUMENTS

    def test_a_nonexistent_audio_file_is_not_found(self, capfd, tmp_path):
        _, payload, _ = invoke(capfd, ["probe", "--audio", str(tmp_path / "absent.m4a")])

        assert payload["error"]["type"] == RESOURCE_NOT_FOUND

    def test_a_nonsense_speaker_count_is_rejected(self, capfd, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(capfd, ["transcribe", "--audio", str(audio), "--speakers", "0"])

        assert payload["error"]["type"] == INVALID_ARGUMENTS

    def test_a_negative_timeout_is_rejected(self, capfd, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(
            capfd, ["transcribe", "--audio", str(audio), "--timeout-seconds", "-1"]
        )

        assert payload["error"]["type"] == INVALID_ARGUMENTS


# --- error classification -------------------------------------------------------


class TestClassify:
    def test_a_hosted_auth_failure_is_authentication_failed(self):
        from transcribe_pipeline.diarization import DiarizationError

        error_type, _, _ = transcribe_cli._classify(
            DiarizationError("pyannoteAI error while polling: HTTP 401 unauthorized")
        )

        assert error_type == AUTHENTICATION_FAILED

    def test_a_hosted_rate_limit_is_rate_limited(self):
        from transcribe_pipeline.diarization import DiarizationError

        error_type, _, _ = transcribe_cli._classify(DiarizationError("HTTP 429 slow down"))

        assert error_type == RATE_LIMITED

    def test_a_stalled_hosted_job_is_a_timeout(self):
        from transcribe_pipeline.diarization import DiarizationError

        error_type, _, _ = transcribe_cli._classify(
            DiarizationError("Diarization job j1 still running after 60 minutes")
        )

        assert error_type == TIMEOUT

    def test_other_hosted_failures_are_upstream(self):
        from transcribe_pipeline.diarization import DiarizationError

        error_type, _, _ = transcribe_cli._classify(DiarizationError("HTTP 503 unavailable"))

        assert error_type == UPSTREAM_ERROR

    def test_a_missing_local_environment_is_a_dependency_problem(self):
        from transcribe_pipeline.local_diarization import LocalDiarizationError

        error_type, _, _ = transcribe_cli._classify(LocalDiarizationError("uv is required"))

        assert error_type == DEPENDENCY_MISSING

    def test_a_missing_aligner_is_a_dependency_problem(self):
        from transcribe_pipeline.alignment import AlignmentError

        error_type, _, _ = transcribe_cli._classify(AlignmentError("worker exited with status 1"))

        assert error_type == DEPENDENCY_MISSING

    def test_a_missing_key_is_authentication_failed(self):
        error_type, _, _ = transcribe_cli._classify(SystemExit("Missing OPENAI_API_KEY. ..."))

        assert error_type == AUTHENTICATION_FAILED

    def test_a_missing_command_is_a_dependency_problem(self):
        error_type, _, _ = transcribe_cli._classify(
            SystemExit("Missing required command(s): ffmpeg")
        )

        assert error_type == DEPENDENCY_MISSING

    def test_a_deadline_is_a_timeout(self):
        error_type, _, _ = transcribe_cli._classify(TimeoutError("transcribe exceeded 10s"))

        assert error_type == TIMEOUT

    def test_an_unexpected_exception_is_internal(self):
        error_type, message, details = transcribe_cli._classify(RuntimeError("boom"))

        assert error_type == INTERNAL_ERROR
        assert message == "boom"
        assert details["exception"] == "RuntimeError"

    def test_the_type_survives_a_message_rewrite(self):
        """Callers branch on type, so it must not depend on the prose."""
        from transcribe_pipeline.local_diarization import LocalDiarizationError

        first, _, _ = transcribe_cli._classify(LocalDiarizationError("uv is required"))
        second, _, _ = transcribe_cli._classify(LocalDiarizationError("something else entirely"))

        assert first == second == DEPENDENCY_MISSING


# --- check-setup ----------------------------------------------------------------


class TestCheckSetup:
    def test_missing_credentials_are_reported_without_their_values(
        self, capfd, isolated_env, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

        _, payload, _ = invoke(capfd, ["check-setup"])
        credentials = payload["result"]["credentials"]

        assert credentials["OPENAI_API_KEY"] is True
        assert credentials["PYANNOTE_API_KEY"] is False
        assert "sk-secret-value" not in json.dumps(payload)

    def test_blockers_name_what_to_fix(self, capfd, isolated_env):
        _, payload, _ = invoke(capfd, ["check-setup"])

        assert payload["result"]["ready"] is False
        assert any("OPENAI_API_KEY" in blocker for blocker in payload["result"]["blockers"])

    def test_each_backend_reports_its_own_readiness(self, capfd, isolated_env, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "diarize_venv_python", lambda: None)

        _, payload, _ = invoke(capfd, ["check-setup"])
        backends = payload["result"]["diarization_backends"]

        assert backends["pyannote"]["ready"] is False
        assert backends["community"]["ready"] is False
        assert backends["community"]["model"] == "community-1"
        assert any("setup_diarization" in b for b in backends["community"]["blockers"])

    def test_a_ready_setup_has_no_blockers(self, capfd, isolated_env, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("PYANNOTE_API_KEY", "pk-x")
        monkeypatch.setattr(transcribe_cli, "align_venv_python", lambda: tmp_path / "python")
        monkeypatch.setattr(transcribe_cli, "diarize_venv_python", lambda: tmp_path / "python")

        _, payload, _ = invoke(capfd, ["check-setup"])

        assert payload["result"]["blockers"] == []
        assert payload["result"]["ready"] is True


# --- dry run --------------------------------------------------------------------


@pytest.fixture
def fake_media(monkeypatch):
    from transcribe_pipeline import media
    from transcribe_pipeline.models import MediaInfo

    monkeypatch.setattr(media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(media, "probe", lambda path: MediaInfo(duration=120.0, size=1_000_000))


class TestDryRun:
    def test_a_dry_run_reports_the_plan_without_running_anything(
        self, capfd, isolated_env, fake_media, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("PYANNOTE_API_KEY", "pk-x")
        called = []
        monkeypatch.setattr(
            "transcribe_pipeline.pipeline.run", lambda options: called.append(options)
        )
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(capfd, ["transcribe", "--audio", str(audio), "--dry-run"])

        assert called == []
        assert payload["result"]["dry_run"] is True
        assert payload["result"]["would_run"] is True
        assert payload["result"]["models"]["diarization"] == "precision-2"

    def test_a_dry_run_names_the_missing_requirements(
        self, capfd, isolated_env, fake_media, tmp_path
    ):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(capfd, ["transcribe", "--audio", str(audio), "--dry-run"])

        assert payload["ok"] is True  # a plan is a successful answer
        assert payload["result"]["would_run"] is False
        assert "OPENAI_API_KEY" in payload["result"]["missing_requirements"]
        assert "PYANNOTE_API_KEY" in payload["result"]["missing_requirements"]

    def test_a_dry_run_on_the_local_backend_wants_the_environment(
        self, capfd, isolated_env, fake_media, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setattr(transcribe_cli, "diarize_venv_python", lambda: None)
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(
            capfd,
            ["transcribe", "--audio", str(audio), "--diarization-backend", "community", "--dry-run"],
        )

        assert payload["result"]["missing_requirements"] == [".venv-diarize"]
        assert payload["result"]["models"]["diarization"] == "community-1"

    def test_no_align_needs_no_diarization_credentials(
        self, capfd, isolated_env, fake_media, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(
            capfd, ["transcribe", "--audio", str(audio), "--no-align", "--dry-run"]
        )

        assert payload["result"]["would_run"] is True
        assert payload["result"]["models"]["diarization"] == "none"
        assert payload["result"]["diarization_backend"] is None


# --- background jobs ------------------------------------------------------------


class TestDeriveProgress:
    def test_an_empty_log_is_queued(self):
        assert transcribe_cli.derive_progress([])["phase"] == "queued"

    def test_each_phase_is_recognised(self):
        cases = {
            "Preprocessing audio with FFmpeg...": "preprocess",
            "Chunking 1170.7s / 10.0MB audio...": "chunk",
            "Transcribing chunk 1/3...": "transcribe",
            "Diarizing with pyannoteAI precision-2...": "diarize",
            "Aligning transcript to audio with WhisperX...": "align",
            "Cleaning segments 1-25 of 319...": "clean",
        }
        for line, expected in cases.items():
            assert transcribe_cli.derive_progress([line])["phase"] == expected, line

    def test_the_latest_line_decides_the_phase(self):
        progress = transcribe_cli.derive_progress(
            [
                "Preprocessing audio with FFmpeg...",
                "Transcribing chunk 1/3...",
                "Aligning region 2/3...",
            ]
        )

        assert progress["phase"] == "align"

    def test_counted_work_becomes_a_fraction_and_a_detail(self):
        progress = transcribe_cli.derive_progress(["Transcribing chunk 2/4..."])

        assert progress["phase_progress"] == pytest.approx(0.5)
        assert progress["phase_detail"] == "chunk 2 of 4"

    def test_cleanup_counts_segments_finished_not_started(self):
        progress = transcribe_cli.derive_progress(["Cleaning segments 26-50 of 100..."])

        assert progress["phase_progress"] == pytest.approx(0.5)
        assert progress["phase_detail"] == "segments 50 of 100"

    def test_progress_increases_along_the_pipeline(self):
        seen = [
            transcribe_cli.derive_progress([line])["progress"]
            for line in (
                "Preprocessing audio with FFmpeg...",
                "Transcribing chunk 1/1...",
                "Diarization found 2 speaker(s) across 5 turns.",
                "Aligning region 1/1...",
                "Cleaning segments 1-5 of 5...",
            )
        ]

        assert seen == sorted(seen)

    def test_a_running_job_never_claims_to_be_finished(self):
        progress = transcribe_cli.derive_progress(["Cleaning segments 1-99 of 99..."])

        assert progress["progress"] < 1.0

    def test_unrecognised_noise_does_not_shift_the_phase(self):
        progress = transcribe_cli.derive_progress(
            ["Transcribing chunk 1/2...", "UserWarning: std(): degrees of freedom"]
        )

        assert progress["phase"] == "transcribe"


@pytest.fixture
def fake_spawn(monkeypatch):
    """Capture what would be launched instead of launching it."""
    launched: dict = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(transcribe_cli.subprocess, "Popen", fake_popen)
    return launched


class TestSpawn:
    def test_detaching_returns_a_job_id_without_running_the_pipeline(
        self, capfd, fake_spawn, tmp_path, monkeypatch
    ):
        ran = []
        monkeypatch.setattr("transcribe_pipeline.pipeline.run", lambda o: ran.append(o))
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(
            capfd,
            ["transcribe", "--audio", str(audio), "--detach", "--jobs-dir", str(tmp_path / "jobs")],
        )

        assert ran == []
        assert payload["result"]["detached"] is True
        assert payload["result"]["state"] == "running"
        assert payload["result"]["job_id"]

    def test_the_child_does_not_inherit_the_detach_flag(self, capfd, fake_spawn, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        invoke(
            capfd,
            [
                "transcribe", "--audio", str(audio), "--language", "en",
                "--detach", "--jobs-dir", str(tmp_path / "jobs"),
            ],
        )

        assert "--detach" not in fake_spawn["command"]
        assert "--language" in fake_spawn["command"]
        assert "transcribe" in fake_spawn["command"]

    def test_the_child_is_detached_from_this_process_group(self, capfd, fake_spawn, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        invoke(
            capfd,
            ["transcribe", "--audio", str(audio), "--detach", "--jobs-dir", str(tmp_path / "jobs")],
        )

        assert fake_spawn["kwargs"]["start_new_session"] is True

    def test_the_job_directory_is_reported_so_the_caller_need_not_guess(
        self, capfd, fake_spawn, tmp_path
    ):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")
        jobs = tmp_path / "jobs"

        _, payload, _ = invoke(
            capfd, ["transcribe", "--audio", str(audio), "--detach", "--jobs-dir", str(jobs)]
        )
        paths = payload["result"]["paths"]

        assert Path(paths["status"]).is_file()
        assert Path(paths["dir"]).parent == jobs

    def test_a_dry_run_stays_synchronous(self, capfd, fake_spawn, isolated_env, fake_media, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"x")

        _, payload, _ = invoke(
            capfd, ["transcribe", "--audio", str(audio), "--detach", "--dry-run"]
        )

        assert fake_spawn == {}
        assert payload["result"]["dry_run"] is True


def write_job(root: Path, job_id: str, *, state="running", pid=1, stderr="", result=None) -> Path:
    directory = root / job_id
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(
        json.dumps({"job_id": job_id, "state": state, "pid": pid, "started_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (directory / "stderr.log").write_text(stderr, encoding="utf-8")
    if result is not None:
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return directory


class TestReadJob:
    def test_a_live_job_reports_its_phase(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        write_job(tmp_path, "j1", stderr="Transcribing chunk 1/2...\n")

        job = transcribe_cli.read_job(tmp_path, "j1")

        assert job["state"] == "running"
        assert job["phase"] == "transcribe"

    def test_a_finished_job_carries_its_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(
            tmp_path, "j2",
            result={"ok": True, "action": "transcribe", "result": {"speakers": ["S0"]}},
        )

        job = transcribe_cli.read_job(tmp_path, "j2")

        assert job["state"] == "succeeded"
        assert job["progress"] == 1.0
        assert job["result"]["speakers"] == ["S0"]

    def test_a_failed_job_carries_the_error_inside_the_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(
            tmp_path, "j3",
            result={
                "ok": False, "action": "transcribe",
                "error": {"type": AUTHENTICATION_FAILED, "message": "bad key", "details": {}},
            },
        )

        job = transcribe_cli.read_job(tmp_path, "j3")

        assert job["state"] == "failed"
        assert job["error"]["type"] == AUTHENTICATION_FAILED

    def test_a_crashed_job_is_failed_with_its_last_output(self, tmp_path, monkeypatch):
        """No envelope and no process means the child died; say so rather than hanging."""
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(tmp_path, "j4", stderr="Segmentation fault\n")

        job = transcribe_cli.read_job(tmp_path, "j4")

        assert job["state"] == "failed"
        assert job["error"]["type"] == INTERNAL_ERROR
        assert "Segmentation fault" in job["error"]["details"]["stderr_tail"]

    def test_a_terminal_state_is_recorded_so_it_is_not_recomputed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(tmp_path, "j5", result={"ok": True, "action": "transcribe", "result": {}})

        transcribe_cli.read_job(tmp_path, "j5")
        persisted = json.loads((tmp_path / "j5" / "status.json").read_text())

        assert persisted["state"] == "succeeded"
        assert persisted["finished_at"]

    def test_a_cancelled_job_keeps_the_phase_it_stopped_in(self, tmp_path):
        write_job(tmp_path, "j6", state="cancelled", stderr="Transcribing chunk 1/4...\n")

        job = transcribe_cli.read_job(tmp_path, "j6")

        assert job["state"] == "cancelled"
        assert job["phase"] == "transcribe"
        assert job["progress"] < 1.0

    def test_an_unknown_job_is_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe_cli.read_job(tmp_path, "absent")


class TestJobActions:
    def test_polling_an_unknown_job_is_resource_not_found(self, capfd, tmp_path):
        _, payload, _ = invoke(
            capfd, ["job-status", "--job-id", "absent", "--jobs-dir", str(tmp_path)]
        )

        assert payload["error"]["type"] == RESOURCE_NOT_FOUND

    def test_a_poll_succeeds_even_when_the_job_failed(self, capfd, tmp_path, monkeypatch):
        """`ok` describes the poll; `result.state` describes the job."""
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(
            tmp_path, "j1",
            result={"ok": False, "action": "transcribe",
                    "error": {"type": UPSTREAM_ERROR, "message": "boom", "details": {}}},
        )

        code, payload, _ = invoke(
            capfd, ["job-status", "--job-id", "j1", "--jobs-dir", str(tmp_path)]
        )

        assert code == 0 and payload["ok"] is True
        assert payload["result"]["state"] == "failed"

    def test_listing_is_newest_first(self, capfd, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        for job_id in ("20260101-000000-aaa", "20260102-000000-bbb"):
            write_job(tmp_path, job_id)

        _, payload, _ = invoke(capfd, ["job-list", "--jobs-dir", str(tmp_path)])

        assert [job["job_id"] for job in payload["result"]["jobs"]] == [
            "20260102-000000-bbb",
            "20260101-000000-aaa",
        ]

    def test_listing_can_filter_by_state(self, capfd, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        write_job(tmp_path, "j1")
        write_job(tmp_path, "j2", state="cancelled")

        _, payload, _ = invoke(
            capfd, ["job-list", "--jobs-dir", str(tmp_path), "--state", "cancelled"]
        )

        assert [job["job_id"] for job in payload["result"]["jobs"]] == ["j2"]

    def test_listing_respects_the_limit(self, capfd, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        for index in range(5):
            write_job(tmp_path, f"job-{index}")

        _, payload, _ = invoke(capfd, ["job-list", "--jobs-dir", str(tmp_path), "--limit", "2"])

        assert len(payload["result"]["jobs"]) == 2

    def test_a_capped_listing_says_so(self, capfd, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        for index in range(5):
            write_job(tmp_path, f"job-{index}")

        _, payload, _ = invoke(capfd, ["job-list", "--jobs-dir", str(tmp_path), "--limit", "2"])

        assert payload["meta"]["truncated"] is True
        assert payload["meta"]["count"] == 2
        assert payload["meta"]["total"] == 5

    def test_a_complete_listing_is_not_flagged_as_truncated(self, capfd, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        for index in range(3):
            write_job(tmp_path, f"job-{index}")

        _, payload, _ = invoke(capfd, ["job-list", "--jobs-dir", str(tmp_path), "--limit", "10"])

        assert payload["meta"]["truncated"] is False
        assert payload["meta"]["count"] == payload["meta"]["total"] == 3

    def test_truncation_counts_only_what_the_state_filter_matched(
        self, capfd, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        for index in range(4):
            write_job(tmp_path, f"running-{index}")
        write_job(tmp_path, "cancelled-1", state="cancelled")

        _, payload, _ = invoke(
            capfd, ["job-list", "--jobs-dir", str(tmp_path), "--state", "cancelled"]
        )

        assert payload["meta"]["total"] == 1
        assert payload["meta"]["truncated"] is False

    def test_an_absent_jobs_directory_lists_nothing(self, capfd, tmp_path):
        _, payload, _ = invoke(capfd, ["job-list", "--jobs-dir", str(tmp_path / "never")])

        assert payload["result"]["jobs"] == []

    def test_a_nonsense_limit_is_rejected(self, capfd, tmp_path):
        _, payload, _ = invoke(
            capfd, ["job-list", "--jobs-dir", str(tmp_path), "--limit", "0"]
        )

        assert payload["error"]["type"] == INVALID_ARGUMENTS

    def test_cancelling_a_finished_job_is_a_no_op_not_a_failure(
        self, capfd, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: False)
        write_job(tmp_path, "j1", result={"ok": True, "action": "transcribe", "result": {}})

        code, payload, _ = invoke(
            capfd, ["job-cancel", "--job-id", "j1", "--jobs-dir", str(tmp_path)]
        )

        assert code == 0
        assert payload["result"]["cancelled"] is False

    def test_the_jobs_directory_can_come_from_the_environment(
        self, capfd, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(transcribe_cli.JOBS_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(transcribe_cli, "_process_alive", lambda pid: True)
        write_job(tmp_path, "j1")

        _, payload, _ = invoke(capfd, ["job-list"])

        assert [job["job_id"] for job in payload["result"]["jobs"]] == ["j1"]


# --- the executable -------------------------------------------------------------


class TestExecutable:
    def test_the_wrapper_needs_no_uv_run(self):
        completed = subprocess.run(
            [str(EXECUTABLE), "list-actions"], capture_output=True, text=True, cwd=ROOT
        )

        assert completed.returncode == 0
        assert json.loads(completed.stdout)["ok"] is True

    def test_the_wrapper_works_from_any_directory(self, tmp_path):
        completed = subprocess.run(
            [str(EXECUTABLE), "list-actions"], capture_output=True, text=True, cwd=tmp_path
        )

        assert completed.returncode == 0
        assert json.loads(completed.stdout)["ok"] is True

    def test_a_failure_still_exits_nonzero_with_json(self):
        completed = subprocess.run(
            [str(EXECUTABLE), "frobnicate"], capture_output=True, text=True, cwd=ROOT
        )

        assert completed.returncode == transcribe_cli.USAGE_EXIT_CODE
        assert json.loads(completed.stdout)["error"]["type"] == INVALID_ARGUMENTS

    def test_the_wrapper_is_executable(self):
        import os

        assert os.access(EXECUTABLE, os.X_OK)
