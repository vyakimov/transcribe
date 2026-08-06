---
name: transcribe
description: Transcribe audio into a speaker-attributed, timestamped transcript via the repository's `bin/transcribe` JSON CLI. Use for transcription, diarization ("who spoke when"), speaker attribution, interview and meeting recordings, or when asked to check transcription setup or estimate a run before spending money.
version: 1.1
license: same as the containing repository
---

# Transcribe

A machine-callable pipeline that turns an audio file into a speaker-attributed,
timestamped, punctuation-cleaned transcript.

Transcription and diarization run **independently** and are reconciled by forced
alignment. Neither sees the other's output, so overlapping speech cannot
duplicate text or silently swap speakers.

## The one command

```
<repo>/bin/transcribe <action> [--flag value ...]
```

Call that path directly. Do **not** prefix it with `uv run`, `python`, or
`poetry run` — the executable resolves its own interpreter. It is safe to
allowlist as a single absolute path.

## Output contract

`stdout` is always exactly one JSON object, on success and on failure alike.
Progress, model chatter, and subprocess noise go to `stderr` and can be ignored.

```json
{"ok": true,  "action": "probe", "result": { }, "meta": {"schema_version": "1.2"}}
{"ok": false, "action": "probe", "error": {"type": "resource_not_found",
                                           "message": "...", "details": { }},
                                 "meta": {"schema_version": "1.2"}}
```

| Exit code | Meaning |
| --- | --- |
| `0` | Success — `ok` is true |
| `2` | The call was malformed; `error.type` is `invalid_arguments`. Fix the call, never retry it unchanged |
| `1` | Every other failure — the call was well formed, the work failed |

`2` is worth branching on before you parse anything: it means the problem is in
your arguments, not in the audio, the network, or the credentials.

**Parse stdout. Branch on `error.type`, never on `error.message`** — types are
stable, messages are not.

## Actions

Run `bin/transcribe list-actions` for the authoritative, always-current catalog
with every parameter. Summary:

| Action | Mutates | Cost | Purpose |
| --- | --- | --- | --- |
| `list-actions` | no | free | Machine-readable catalog of actions and parameters |
| `check-setup` | no | free | What is installed, which credentials exist, which backends are ready |
| `probe` | no | free | An audio file's duration and size |
| `transcribe` | **yes** | **paid, slow** | The full pipeline |
| `job-status` | no | free | Poll a detached run |
| `job-list` | no | free | List detached runs, newest first |
| `job-cancel` | yes | free | Stop a running detached run |

`--version` is a flag rather than an action, and answers as a normal envelope
with `"action": "version"`:

```json
{"ok": true, "action": "version",
 "result": {"name": "transcribe", "version": "0.1.0", "schema_version": "1.2"}}
```

Check `result.schema_version` before relying on anything this document
describes; the payload shapes here are the `1.x` contract.

### transcribe

**Always pass `--detach`.** Transcription takes minutes — often longer than the
audio — and a synchronous call blocks you for the whole run with no visibility.
`--detach` returns a `job_id` in well under a second and the work continues in a
detached process that survives your shell exiting. Only run synchronously when
the caller explicitly needs a blocking call.

Key parameters (`--audio` is required):

| Flag | Effect |
| --- | --- |
| `--audio PATH` | Input audio file |
| `--language CODE` | ISO-639-1, e.g. `en`. Also selects the alignment model. Pass it when known — it improves accuracy |
| `--speakers N` | Known speaker count, if you have it |
| `--diarization-backend` | `pyannote` (hosted, default) or `community` (local) |
| `--diarization-device` | `cpu` (default), `cuda`, or `mps` — local backend only |
| `--output-dir DIR` | Where the three output files land |
| `--include-text` | Also return the readable transcript inline in `result.transcript` |
| `--no-align` | Skip diarization and alignment: fast and cheap, but **no speakers and no timestamps** |
| `--no-clean` | Skip punctuation cleanup |
| `--detach` | Return a `job_id` immediately and run in the background — **use this** |
| `--dry-run` | Validate and report the plan without calling any paid API |
| `--timeout-seconds N` | Wall-clock limit, default 7200; `0` disables |
| `--jobs-dir DIR` | Where job state lives (default `.transcribe-jobs/`, or `$TRANSCRIBE_JOBS_DIR`) |

Writes three files and reports their paths in `result.outputs`:

- `<name>.json` — the primary output: segments with speakers, overlap flags, raw and cleaned text
- `<name>.txt` — readable timestamped transcript
- `<name>.raw.json` — untouched transcription responses, saved before cleanup

## Recommended workflow

```bash
bin/transcribe check-setup
bin/transcribe transcribe --audio recording.m4a --language en --dry-run
bin/transcribe transcribe --audio recording.m4a --language en --detach
#   -> {"result": {"job_id": "20260806-173647-de4ad931", ...}}
bin/transcribe job-status --job-id 20260806-173647-de4ad931   # repeat
```

1. **`check-setup`** — if `result.ready` is false, `result.blockers` says exactly
   what to fix. Do this first on an unfamiliar machine.
2. **`transcribe --dry-run`** — confirms `would_run`, lists
   `missing_requirements`, and reports duration and output paths. Free.
3. **`transcribe --detach`** — returns `result.job_id` immediately.
4. **`job-status --job-id ...`** — poll until `result.state` is no longer
   `running`. Poll every 15–30 s; a run takes minutes, so faster polling only
   burns turns.

Then read `result.summary` for quality: `attribution_coverage` near `1.0` is
healthy. Check `result.warnings` — they are advisory, not failures.

### Reading job-status

```json
{"ok": true, "action": "job-status", "result": {
  "job_id": "20260806-173647-de4ad931",
  "state": "running",
  "phase": "align", "phase_detail": "region 2 of 3", "progress": 0.79,
  "recent_stderr": ["Aligning region 2/3..."],
  "paths": {"dir": "...", "status": "...", "stderr": "...", "result": "..."}}}
```

**`ok` reports whether the *poll* succeeded; `result.state` reports what happened
to the *job*.** A poll of a failed job returns `ok: true` with
`result.state: "failed"` and the job's own error under `result.error` — carrying
the same stable `type` values listed below. Do not treat `ok: true` as "the
transcription worked."

| `result.state` | Meaning |
| --- | --- |
| `running` | Still working; `phase` and `progress` show where it is |
| `succeeded` | Finished; the full transcribe payload is under `result.result` |
| `failed` | Finished badly; see `result.error.type` |
| `cancelled` | Stopped by `job-cancel` |

`phase` moves through `preprocess → chunk → transcribe → diarize → align →
clean`. `progress` is a coarse 0–1 estimate derived from those phases — useful
for reporting, not precise. It never reaches `1.0` while a job is running.

Job state lives in `.transcribe-jobs/<job_id>/` (`status.json`, `stderr.log`,
`result.json`). Read `stderr.log` directly when a job fails and you need more
than the tail in `recent_stderr`.

### Reading job-list

| Flag | Effect |
| --- | --- |
| `--limit N` | Maximum jobs returned, newest first. Default `20` |
| `--state` | Only return jobs in one state: `running`, `succeeded`, `failed`, or `cancelled` |
| `--jobs-dir DIR` | Where job state lives (default `.transcribe-jobs/`, or `$TRANSCRIBE_JOBS_DIR`) |

`meta` reports whether the cap hid anything: `count` is what you got, `total` is
how many matched, and `truncated` is true when `count < total`. Raise `--limit`
or narrow with `--state` when `truncated` is true — never conclude a job is
absent from a truncated list.

```json
{"ok": true, "action": "job-list",
 "result": {"jobs": [ ], "jobs_dir": "..."},
 "meta": {"count": 20, "total": 34, "truncated": true, "schema_version": "1.2"}}
```

## Choosing a diarization backend

| | `pyannote` (default) | `community` |
| --- | --- | --- |
| Runs on | pyannoteAI's API | this machine |
| Needs | `PYANNOTE_API_KEY` | `.venv-diarize` built once |
| Per-run cost | billed by pyannoteAI | none |
| First-run setup | none | ~1 GB download, gated model |

They are **not** interchanged automatically. If the selected backend is not
ready, the call fails rather than quietly switching — switching would change
cost and quality without saying so.

Prefer `community` when the machine is already set up (`check-setup` reports
`diarization_backends.community.ready`) and cost matters. Prefer `pyannote` when
no local environment exists. On a measured 19.5-minute two-speaker interview the
two agreed on the speaker for 98.5% of speech, both found the same speakers, and
both attributed 100% of segments — so backend choice is usually a cost and setup
decision, not a quality one. Verify on your own audio before committing.

## Error types

| `error.type` | Meaning | What to do |
| --- | --- | --- |
| `invalid_arguments` | Bad, missing, or unknown parameter | Fix the call; `details.known_actions` lists valid actions |
| `resource_not_found` | The audio file does not exist | Check the path |
| `authentication_failed` | Missing or rejected credential | Set the named variable in `.env`; do not retry unchanged |
| `permission_denied` | Authenticated but not allowed | Escalate to the user |
| `timeout` | Wall-clock or upstream timeout | Retry, or raise `--timeout-seconds` |
| `upstream_error` | A provider API failed | Retry once; if it persists, report it |
| `rate_limited` | Quota exceeded | Back off, then retry |
| `conflict` | Incompatible state | Report it |
| `dependency_missing` | A tool or environment is absent | Run the setup step the message names |
| `internal_error` | Unexpected failure | Report it with `details.exception` |

Credentials live in `.env` at the repository root: `OPENAI_API_KEY` always,
`PYANNOTE_API_KEY` for the hosted backend, `HF_TOKEN` only to download the local
model the first time. `check-setup` reports which are present and never prints
their values.

## Rules

- Never invent flags. Confirm them with `list-actions`.
- Run `transcribe` with `--detach` and poll. Do not block on it.
- `transcribe` spends money on every real call. Prefer `--dry-run` first, and do
  not retry a paid run to "see if it works this time." Before launching a second
  job for the same audio, check `job-list` — a run you forgot about may still be
  going, and detached jobs outlive the shell that started them.
- The pipeline is stateless: each call is independent and re-running it
  overwrites the outputs for that audio stem. Use `--output-dir` to keep runs
  side by side.
- Do not parse the human-readable `.txt`. Read the `.json`.
- `--no-align` produces no speakers and no timestamps. Only use it when the user
  asked for plain text.
