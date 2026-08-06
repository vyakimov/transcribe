#!/usr/bin/env python3
"""Machine-callable entry point for the transcription pipeline.

Every invocation prints exactly one JSON object to stdout and nothing else.
Progress, model chatter, and subprocess noise go to stderr. Callers can parse
stdout unconditionally, on success and on failure alike.

Envelope:
    {"ok": true,  "action": str, "result": {...}, "meta": {...}}
    {"ok": false, "action": str, "error": {"type": str, "message": str,
                                           "details": {...}}, "meta": {...}}

Exit code is 0 when `ok` is true and 1 when it is false. Run
`bin/transcribe list-actions` for the machine-readable action catalog.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from transcribe_pipeline.config import (
    DEFAULT_OUTPUT_DIR,
    DIARIZATION_BACKEND_COMMUNITY,
    DIARIZATION_BACKEND_PYANNOTE,
    DIARIZATION_BACKENDS,
    PROJECT_ROOT,
    align_venv_python,
    diarization_model_name,
    diarize_venv_python,
    load_environment,
    resolve_diarization_backend,
)

SCHEMA_VERSION = "1.1"
DEFAULT_TIMEOUT_SECONDS = 7200

CLI_PATH = Path(__file__).resolve()
DEFAULT_JOBS_DIR = PROJECT_ROOT / ".transcribe-jobs"
JOBS_DIR_ENV = "TRANSCRIBE_JOBS_DIR"

JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

# Stable, snake_case error types. Callers should branch on these, never on the
# message text, which is free to change.
INVALID_ARGUMENTS = "invalid_arguments"
RESOURCE_NOT_FOUND = "resource_not_found"
AUTHENTICATION_FAILED = "authentication_failed"
PERMISSION_DENIED = "permission_denied"
TIMEOUT = "timeout"
UPSTREAM_ERROR = "upstream_error"
RATE_LIMITED = "rate_limited"
CONFLICT = "conflict"
DEPENDENCY_MISSING = "dependency_missing"
INTERNAL_ERROR = "internal_error"


# --- envelope -------------------------------------------------------------------


def _ok(action: str, result: dict, meta: dict | None = None) -> dict:
    return {"ok": True, "action": action, "result": result, "meta": meta or {}}


def _fail(
    action: str, error_type: str, message: str, details: dict | None = None,
    meta: dict | None = None,
) -> dict:
    return {
        "ok": False,
        "action": action,
        "error": {"type": error_type, "message": message, "details": details or {}},
        "meta": meta or {},
    }


def _emit(payload: dict, stream=None) -> int:
    """Write the one and only JSON object, then report the process exit code."""
    payload.setdefault("meta", {})["schema_version"] = SCHEMA_VERSION
    print(json.dumps(payload, indent=2, ensure_ascii=False), file=stream or sys.stdout)
    return 0 if payload.get("ok") else 1


@contextlib.contextmanager
def _quarantined_stdout():
    """Point fd 1 at fd 2 so nothing but our JSON can reach the caller's stdout.

    The pipeline's workers are subprocesses whose native dependencies (torch,
    ffmpeg, pyannote) write to file descriptor 1 directly. Reassigning the
    descriptor is the only thing that reliably contains them; replacing
    `sys.stdout` alone would not.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


@contextlib.contextmanager
def _deadline(seconds: int, action: str):
    """Enforce a wall-clock limit, so a wedged upstream cannot hang the caller."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _expire(_signum, _frame):
        raise TimeoutError(f"{action} exceeded its {seconds}s timeout")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --- background jobs ------------------------------------------------------------

# Phases in pipeline order, with the share of a run each one typically takes.
# The weights are calibrated from real runs and only feed the coarse `progress`
# estimate; `phase` is the number to trust.
PHASE_WEIGHTS: list[tuple[str, float]] = [
    ("queued", 0.0),
    ("preprocess", 0.05),
    ("chunk", 0.02),
    ("transcribe", 0.25),
    ("diarize", 0.25),
    ("align", 0.33),
    ("clean", 0.10),
]

# Matched against the pipeline's own stderr. Ordered most specific first; the
# last line that matches anything decides the current phase.
PHASE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Cleaning segments (\d+)-(\d+) of (\d+)"), "clean"),
    (re.compile(r"Aligning region (\d+)/(\d+)"), "align"),
    (re.compile(r"Loading wav2vec2 alignment model"), "align"),
    (re.compile(r"Aligning transcript to audio"), "align"),
    (re.compile(r"Diarization found"), "diarize"),
    (re.compile(r"Diarizing"), "diarize"),
    (re.compile(r"Loading pyannote/"), "diarize"),
    (re.compile(r"Transcribing chunk (\d+)/(\d+)"), "transcribe"),
    (re.compile(r"Chunking "), "chunk"),
    (re.compile(r"Preprocessing audio"), "preprocess"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jobs_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get(JOBS_DIR_ENV, "").strip() or DEFAULT_JOBS_DIR).expanduser()


def _job_paths(root: Path, job_id: str) -> dict[str, Path]:
    directory = root / job_id
    return {
        "dir": directory,
        "status": directory / "status.json",
        "stderr": directory / "stderr.log",
        "result": directory / "result.json",
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, but it exists.
        return True
    return True


def _tail(path: Path, limit: int = 200) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()[-limit:]
    except OSError:
        return []


def derive_progress(lines: list[str]) -> dict:
    """Turn the pipeline's stderr into a phase and a coarse completion estimate."""
    phase, detail, fraction = "queued", None, None

    for line in reversed(lines):
        for pattern, name in PHASE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            phase = name
            groups = [int(value) for value in match.groups()] if match.groups() else []
            if name == "clean" and len(groups) == 3 and groups[2]:
                detail = f"segments {groups[1]} of {groups[2]}"
                fraction = min(1.0, groups[1] / groups[2])
            elif len(groups) == 2 and groups[1]:
                unit = "chunk" if name == "transcribe" else "region"
                detail = f"{unit} {groups[0]} of {groups[1]}"
                fraction = min(1.0, groups[0] / groups[1])
            break
        if phase != "queued":
            break

    # Everything before the current phase is done; the current one counts only
    # as far as it has reported.
    completed = 0.0
    for name, weight in PHASE_WEIGHTS:
        if name == phase:
            completed += weight * (fraction if fraction is not None else 0.5)
            break
        completed += weight

    return {
        "phase": phase,
        "phase_detail": detail,
        "phase_progress": round(fraction, 4) if fraction is not None else None,
        "progress": round(min(completed, 0.99), 4),
    }


def _spawn_job(argv: list[str], root: Path) -> dict:
    """Start a detached run and return its identity without waiting for it."""
    job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    paths = _job_paths(root, job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    # --detach is dropped so the child runs the work rather than forking again.
    child_argv = [token for token in argv if token != "--detach"]
    command = [sys.executable, str(CLI_PATH), *child_argv]

    stderr_handle = paths["stderr"].open("wb")
    result_handle = paths["result"].open("wb")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=result_handle,
            stderr=stderr_handle,
            # A new session detaches the child from this process group, so it
            # survives the caller exiting or its terminal closing.
            start_new_session=True,
            cwd=os.getcwd(),
        )
    finally:
        stderr_handle.close()
        result_handle.close()

    status = {
        "job_id": job_id,
        "state": JOB_RUNNING,
        "pid": process.pid,
        "command": command,
        "cwd": os.getcwd(),
        "started_at": _now(),
    }
    paths["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")
    return {"job_id": job_id, "paths": paths, "status": status}


def read_job(root: Path, job_id: str) -> dict:
    """Assemble a job's current state from its directory. Never blocks."""
    paths = _job_paths(root, job_id)
    status = _read_json(paths["status"])
    if status is None:
        raise FileNotFoundError(f"no such job: {job_id}")

    state = status.get("state", JOB_RUNNING)
    lines = _tail(paths["stderr"])
    envelope = _read_json(paths["result"])

    if state == JOB_RUNNING:
        alive = _process_alive(int(status["pid"]))
        if envelope is not None:
            state = JOB_SUCCEEDED if envelope.get("ok") else JOB_FAILED
        elif not alive:
            # The child died without writing an envelope; that is a crash, and
            # the tail of its stderr is the only evidence of why.
            state = JOB_FAILED
            envelope = _fail(
                "transcribe",
                INTERNAL_ERROR,
                "the job exited without producing a result",
                {"stderr_tail": lines[-20:]},
            )
        if state != JOB_RUNNING:
            status["state"] = state
            status["finished_at"] = _now()
            paths["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")

    report = {
        "job_id": job_id,
        "state": state,
        "pid": status.get("pid"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "paths": {name: str(path) for name, path in paths.items()},
        "recent_stderr": lines[-5:],
    }

    if state in (JOB_RUNNING, JOB_CANCELLED):
        # A cancelled job keeps the phase it stopped in; claiming 100% would lie.
        report.update(derive_progress(lines))
    else:
        report.update({"phase": "done", "phase_detail": None, "phase_progress": None,
                       "progress": 1.0})

    if envelope is not None:
        if envelope.get("ok"):
            report["result"] = envelope.get("result")
        else:
            report["error"] = envelope.get("error")
    return report


# --- error classification -------------------------------------------------------


def _classify(error: BaseException) -> tuple[str, str, dict]:
    """Map a pipeline exception onto a stable error type."""
    from transcribe_pipeline.alignment import AlignmentError
    from transcribe_pipeline.diarization import DiarizationError
    from transcribe_pipeline.local_diarization import LocalDiarizationError

    message = str(error).strip() or error.__class__.__name__
    details = {"exception": error.__class__.__name__}

    if isinstance(error, TimeoutError):
        return TIMEOUT, message, details

    try:
        import openai

        if isinstance(error, openai.AuthenticationError):
            return AUTHENTICATION_FAILED, message, details
        if isinstance(error, openai.PermissionDeniedError):
            return PERMISSION_DENIED, message, details
        if isinstance(error, openai.RateLimitError):
            return RATE_LIMITED, message, details
        if isinstance(error, openai.APITimeoutError):
            return TIMEOUT, message, details
        if isinstance(error, openai.APIError):
            return UPSTREAM_ERROR, message, details
    except ImportError:  # pragma: no cover - openai is a hard dependency
        pass

    if isinstance(error, DiarizationError):
        # The hosted diarizer reports transport failures as text; recover the
        # status so callers can retry rate limits without parsing prose.
        if "HTTP 401" in message or "HTTP 403" in message:
            return AUTHENTICATION_FAILED, message, details
        if "HTTP 429" in message:
            return RATE_LIMITED, message, details
        if "still" in message and "minutes" in message:
            return TIMEOUT, message, details
        return UPSTREAM_ERROR, message, details

    if isinstance(error, (LocalDiarizationError, AlignmentError)):
        return DEPENDENCY_MISSING, message, details

    if isinstance(error, SystemExit):
        text = str(error.code) if error.code is not None else ""
        if "Missing required command" in text:
            return DEPENDENCY_MISSING, text, details
        if "API_KEY" in text or "TOKEN" in text:
            return AUTHENTICATION_FAILED, text, details
        if "diarization backend" in text:
            return INVALID_ARGUMENTS, text, details
        return INTERNAL_ERROR, text or "the pipeline exited", details

    if isinstance(error, FileNotFoundError):
        return RESOURCE_NOT_FOUND, message, details

    return INTERNAL_ERROR, message, details


# --- action registry ------------------------------------------------------------


@dataclass
class Action:
    name: str
    summary: str
    handler: Callable[[argparse.Namespace], dict]
    mutating: bool
    configure: Callable[[argparse.ArgumentParser], None]


ACTIONS: dict[str, Action] = {}


def register(name: str, summary: str, *, mutating: bool = False):
    def decorator(handler):
        ACTIONS[name] = Action(
            name=name,
            summary=summary,
            handler=handler,
            mutating=mutating,
            configure=getattr(handler, "_configure", lambda parser: None),
        )
        return handler

    return decorator


def arguments(configure: Callable[[argparse.ArgumentParser], None]):
    def decorator(handler):
        handler._configure = configure
        return handler

    return decorator


def _describe(parser: argparse.ArgumentParser) -> list[dict]:
    described: list[dict] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        described.append(
            {
                "name": action.dest,
                "flags": list(action.option_strings),
                "required": bool(action.required),
                "choices": list(action.choices) if action.choices else None,
                "default": action.default if action.default is not argparse.SUPPRESS else None,
                "repeated": isinstance(action, argparse._StoreAction) and action.nargs == "+",
                "description": action.help,
            }
        )
    return described


# --- actions --------------------------------------------------------------------


@register("list-actions", "Describe every action, its parameters, and its error types.")
def action_list_actions(args: argparse.Namespace) -> dict:
    catalog = []
    for name, action in sorted(ACTIONS.items()):
        parser = argparse.ArgumentParser(prog=name, add_help=False)
        action.configure(parser)
        catalog.append(
            {
                "name": name,
                "summary": action.summary,
                "mutating": action.mutating,
                "parameters": _describe(parser),
            }
        )
    return {
        "actions": catalog,
        "error_types": [
            INVALID_ARGUMENTS,
            RESOURCE_NOT_FOUND,
            AUTHENTICATION_FAILED,
            PERMISSION_DENIED,
            TIMEOUT,
            UPSTREAM_ERROR,
            RATE_LIMITED,
            CONFLICT,
            DEPENDENCY_MISSING,
            INTERNAL_ERROR,
        ],
    }


def _credential(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


@register("check-setup", "Report which tools, environments, and credentials are available.")
def action_check_setup(args: argparse.Namespace) -> dict:
    import shutil

    tools = {name: bool(shutil.which(name)) for name in ("ffmpeg", "ffprobe", "uv")}
    align_python = align_venv_python()
    diarize_python = diarize_venv_python()

    # Never report credential values, only their presence.
    credentials = {
        "OPENAI_API_KEY": _credential("OPENAI_API_KEY"),
        "PYANNOTE_API_KEY": _credential("PYANNOTE_API_KEY"),
        "HF_TOKEN": _credential("HF_TOKEN"),
    }

    backends = {
        DIARIZATION_BACKEND_PYANNOTE: {
            "ready": credentials["PYANNOTE_API_KEY"],
            "model": diarization_model_name(DIARIZATION_BACKEND_PYANNOTE),
            "requires": ["PYANNOTE_API_KEY"],
            "blockers": [] if credentials["PYANNOTE_API_KEY"] else ["PYANNOTE_API_KEY is not set"],
        },
        DIARIZATION_BACKEND_COMMUNITY: {
            "ready": diarize_python is not None,
            "model": diarization_model_name(DIARIZATION_BACKEND_COMMUNITY),
            "requires": [".venv-diarize"],
            "blockers": (
                []
                if diarize_python is not None
                else ["run scripts/setup_diarization.py to build .venv-diarize"]
            ),
        },
    }

    blockers: list[str] = []
    if not tools["ffmpeg"] or not tools["ffprobe"]:
        blockers.append("install ffmpeg (provides ffmpeg and ffprobe)")
    if not credentials["OPENAI_API_KEY"]:
        blockers.append("OPENAI_API_KEY is not set")
    if align_python is None:
        blockers.append("run scripts/setup_alignment.py to build .venv-align")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "tools": tools,
        "credentials": credentials,
        "environments": {
            "align": {"path": str(align_python) if align_python else None,
                      "present": align_python is not None},
            "diarize": {"path": str(diarize_python) if diarize_python else None,
                        "present": diarize_python is not None},
        },
        "diarization_backends": backends,
        "default_diarization_backend": resolve_diarization_backend(None),
        "project_root": str(PROJECT_ROOT),
    }


def _probe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", required=True, help="Path to the audio file to inspect")


@register("probe", "Report an audio file's duration and size without calling any API.")
@arguments(_probe_arguments)
def action_probe(args: argparse.Namespace) -> dict:
    from transcribe_pipeline import media

    audio = Path(args.audio)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file does not exist: {audio}")

    media.require_ffmpeg()
    info = media.probe(audio)
    needs_chunking = (
        info.duration > media.MAX_MODEL_SECONDS or info.size > media.MAX_UPLOAD_BYTES
    )
    return {
        "audio": str(audio),
        "duration_seconds": round(info.duration, 3),
        "size_bytes": info.size,
        "needs_chunking": needs_chunking,
        "limits": {
            "max_upload_bytes": media.MAX_UPLOAD_BYTES,
            "max_model_seconds": media.MAX_MODEL_SECONDS,
        },
    }


def _transcribe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", required=True, help="Path to the audio file to transcribe")
    parser.add_argument("--language", help="ISO-639-1 code, e.g. en; also selects the aligner")
    parser.add_argument(
        "--speakers", type=int, help="Known speaker count, forwarded to diarization"
    )
    parser.add_argument(
        "--diarization-backend",
        choices=list(DIARIZATION_BACKENDS),
        help="pyannote (hosted precision-2, default) or community (local Community-1)",
    )
    parser.add_argument(
        "--diarization-device", help="Device for local diarization: cpu, cuda, or mps"
    )
    parser.add_argument("--align-device", help="Device for forced alignment: cpu or cuda")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the default output paths",
    )
    parser.add_argument("--json-output", help="Explicit path for the JSON transcript")
    parser.add_argument("--text-output", help="Explicit path for the readable transcript")
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Send the original audio unchanged"
    )
    parser.add_argument(
        "--no-denoise", action="store_true", help="Keep filtering, skip noise reduction"
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip diarization and alignment; no speakers or timings are produced",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip the punctuation cleanup pass")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Return the readable transcript inline in result.transcript",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock limit for the whole run; 0 disables it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report the plan without calling any paid API",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Start the run in the background and return a job_id immediately; "
            "poll it with job-status. Recommended for automated callers"
        ),
    )
    parser.add_argument(
        "--jobs-dir",
        help=f"Directory holding job state (default: {DEFAULT_JOBS_DIR}, or ${JOBS_DIR_ENV})",
    )


@register("transcribe", "Transcribe, diarize, align, and clean an audio file.", mutating=True)
@arguments(_transcribe_arguments)
def action_transcribe(args: argparse.Namespace) -> dict:
    from transcribe_pipeline import media
    from transcribe_pipeline.pipeline import Options, run, write_raw_responses
    from transcribe_pipeline.render import render_json, render_text

    audio = Path(args.audio)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file does not exist: {audio}")

    if args.speakers is not None and args.speakers < 1:
        raise ValueError("--speakers must be a positive integer")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout-seconds must not be negative")

    backend = resolve_diarization_backend(args.diarization_backend)

    # A dry run answers in milliseconds, so detaching it would only add a poll.
    if args.detach and not args.dry_run:
        spawned = _spawn_job(getattr(args, "_argv", []), jobs_root(args.jobs_dir))
        return {
            "detached": True,
            "job_id": spawned["job_id"],
            "state": JOB_RUNNING,
            "poll_with": f"job-status --job-id {spawned['job_id']}",
            "paths": {name: str(path) for name, path in spawned["paths"].items()},
            "audio": str(audio),
            "started_at": spawned["status"]["started_at"],
        }

    align = not args.no_align

    output_dir = Path(args.output_dir)
    stem = audio.stem
    json_path = Path(args.json_output) if args.json_output else output_dir / f"{stem}.json"
    text_path = Path(args.text_output) if args.text_output else output_dir / f"{stem}.txt"
    raw_path = output_dir / f"{stem}.raw.json"

    planned = {
        "audio": str(audio),
        "diarization_backend": backend if align else None,
        "models": {
            "transcription": "gpt-transcribe",
            "diarization": diarization_model_name(backend) if align else "none",
            "cleanup": "none" if args.no_clean else "gpt-5.6-terra",
        },
        "outputs": {
            "json": str(json_path),
            "text": str(text_path),
            "raw": str(raw_path),
        },
    }

    if args.dry_run:
        media.require_ffmpeg()
        info = media.probe(audio)
        missing: list[str] = []
        if not _credential("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if align and backend == DIARIZATION_BACKEND_PYANNOTE and not _credential(
            "PYANNOTE_API_KEY"
        ):
            missing.append("PYANNOTE_API_KEY")
        if align and backend == DIARIZATION_BACKEND_COMMUNITY and diarize_venv_python() is None:
            missing.append(".venv-diarize")

        return {
            "dry_run": True,
            "would_run": not missing,
            "missing_requirements": missing,
            "duration_seconds": round(info.duration, 3),
            "size_bytes": info.size,
            "needs_chunking": (
                info.duration > media.MAX_MODEL_SECONDS or info.size > media.MAX_UPLOAD_BYTES
            ),
            **planned,
        }

    options = Options(
        audio=audio,
        language=args.language,
        speakers=args.speakers,
        preprocess=not args.no_preprocess,
        denoise=not args.no_denoise,
        align=align,
        clean=not args.no_clean,
        device=args.align_device,
        diarization_backend=backend,
        diarization_device=args.diarization_device,
    )

    started = time.monotonic()
    with _deadline(args.timeout_seconds, "transcribe"):
        document, raw_responses = run(options)

    for path in (json_path, text_path, raw_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    readable = render_text(document)
    json_path.write_text(render_json(document), encoding="utf-8")
    text_path.write_text(readable, encoding="utf-8")
    write_raw_responses(raw_path, raw_responses)

    segments = document.segments
    attributed = [segment for segment in segments if segment.speaker]
    overlaps = [segment for segment in segments if segment.overlap]

    result = {
        "dry_run": False,
        **planned,
        "duration_seconds": round(document.duration, 3),
        "language": document.language,
        "speakers": document.speakers,
        "summary": {
            "segments": len(segments),
            "attributed_segments": len(attributed),
            "attribution_coverage": (
                round(len(attributed) / len(segments), 4) if segments else 0.0
            ),
            "overlap_segments": len(overlaps),
            "unresolved_overlap_segments": sum(1 for s in overlaps if not s.speaker),
            "unaligned_segments": sum(1 for s in segments if not s.aligned),
        },
        "warnings": document.warnings,
    }
    if args.include_text:
        result["transcript"] = readable
    result["detached"] = False

    return result, {"elapsed_seconds": round(time.monotonic() - started, 2)}


# --- job actions ----------------------------------------------------------------


def _jobs_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jobs-dir",
        help=f"Directory holding job state (default: {DEFAULT_JOBS_DIR}, or ${JOBS_DIR_ENV})",
    )


def _job_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id", required=True, help="Job id returned by transcribe --detach")
    _jobs_dir_argument(parser)


@register("job-status", "Poll a detached run: phase, progress, and its result once finished.")
@arguments(_job_status_arguments)
def action_job_status(args: argparse.Namespace) -> dict:
    return read_job(jobs_root(args.jobs_dir), args.job_id)


def _job_list_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of jobs to return, newest first"
    )
    parser.add_argument(
        "--state",
        choices=[JOB_RUNNING, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED],
        help="Only return jobs in this state",
    )
    _jobs_dir_argument(parser)


@register("job-list", "List detached runs, newest first.")
@arguments(_job_list_arguments)
def action_job_list(args: argparse.Namespace) -> dict:
    if args.limit < 1:
        raise ValueError("--limit must be a positive integer")

    root = jobs_root(args.jobs_dir)
    if not root.is_dir():
        return {"jobs": [], "jobs_dir": str(root)}

    jobs = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        try:
            job = read_job(root, directory.name)
        except FileNotFoundError:
            continue
        if args.state and job["state"] != args.state:
            continue
        jobs.append(
            {key: job[key] for key in ("job_id", "state", "phase", "progress", "started_at")}
        )
        if len(jobs) >= args.limit:
            break

    return {"jobs": jobs, "jobs_dir": str(root)}


def _job_cancel_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id", required=True, help="Job id to stop")
    _jobs_dir_argument(parser)


@register("job-cancel", "Stop a running detached job.", mutating=True)
@arguments(_job_cancel_arguments)
def action_job_cancel(args: argparse.Namespace) -> dict:
    root = jobs_root(args.jobs_dir)
    job = read_job(root, args.job_id)

    if job["state"] != JOB_RUNNING:
        # Cancelling a finished job is a no-op, not an error: a caller racing
        # completion should not have to handle a failure.
        return {"job_id": args.job_id, "state": job["state"], "cancelled": False,
                "reason": "job was already finished"}

    paths = _job_paths(root, args.job_id)
    try:
        # The child leads its own process group, so this reaches its workers too.
        os.killpg(os.getpgid(int(job["pid"])), signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as error:
        raise RuntimeError(f"could not stop job {args.job_id}: {error}") from error

    status = _read_json(paths["status"]) or {}
    status.update({"state": JOB_CANCELLED, "finished_at": _now()})
    paths["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")

    return {"job_id": args.job_id, "state": JOB_CANCELLED, "cancelled": True}


# --- argument parsing -----------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An argparse parser whose failures are JSON, not usage text on stderr."""

    def error(self, message: str):
        raise _ArgumentError(message)

    def exit(self, status: int = 0, message: str | None = None):
        if status:
            raise _ArgumentError(message or "invalid arguments")
        raise SystemExit(status)


class _ArgumentError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="transcribe",
        description="Machine-callable transcription pipeline. stdout is always one JSON object.",
    )
    subparsers = parser.add_subparsers(dest="action", metavar="ACTION")
    for name, action in sorted(ACTIONS.items()):
        subparser = subparsers.add_parser(name, help=action.summary, description=action.summary)
        action.configure(subparser)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_environment()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except _ArgumentError as error:
        return _emit(
            _fail(
                (argv[0] if argv and not argv[0].startswith("-") else "unknown"),
                INVALID_ARGUMENTS,
                str(error),
                {"known_actions": sorted(ACTIONS)},
            )
        )

    if not args.action:
        return _emit(
            _fail(
                "unknown",
                INVALID_ARGUMENTS,
                "no action given; run list-actions for the catalog",
                {"known_actions": sorted(ACTIONS)},
            )
        )

    # `--detach` re-launches this same command minus that flag, so the handler
    # needs the argv exactly as the caller wrote it.
    args._argv = argv
    action = ACTIONS[args.action]

    # Everything the handler and its subprocesses might print is contained here,
    # so the caller's stdout receives the JSON envelope and nothing else.
    try:
        with _quarantined_stdout():
            outcome = action.handler(args)
            meta: dict = {}
            if isinstance(outcome, tuple):
                outcome, meta = outcome
            payload = _ok(args.action, outcome, meta)
    except _ArgumentError as error:
        return _emit(_fail(args.action, INVALID_ARGUMENTS, str(error)))
    except ValueError as error:
        return _emit(_fail(args.action, INVALID_ARGUMENTS, str(error)))
    except KeyboardInterrupt:
        return _emit(_fail(args.action, INTERNAL_ERROR, "interrupted"))
    except BaseException as error:  # noqa: BLE001 - every failure must still be JSON
        error_type, message, details = _classify(error)
        return _emit(_fail(args.action, error_type, message, details))

    return _emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
