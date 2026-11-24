# Hosted transcription with speaker attribution

Transcribes an audio file with OpenAI's `gpt-transcribe`, diarizes it
independently with a local pyannote Community-1 or pyannoteAI `precision-2`, and
reconciles the two with WhisperX's wav2vec2 forced aligner. Transcription and
diarization never see each other's output, so crosstalk cannot duplicate text or
silently reassign it.

## How it works

1. **Preprocess** — a conservative 60 Hz high-pass filter, light stationary-noise
   reduction, and loudness normalization, rendered to mono 24 kHz FLAC.
2. **Chunk** — split only when the 25 MB upload limit or the model's duration
   limit would be exceeded, preferring a detected silence near each balanced
   boundary. Where no silence exists, a five-second overlap is used and each
   word is assigned to exactly one side of the seam.
3. **Transcribe** — `gpt-transcribe` runs once per whole chunk. Isolated
   diarization turns are never transcribed separately: that is what makes
   crosstalk duplicate text and swap speakers.
4. **Diarize** — the whole file is diarized in one job, so speaker labels stay
   consistent end to end. Regular diarization is canonical because it preserves
   overlapping speech; the exclusive view is kept in the JSON and used only to
   break ties the canonical view leaves open. Two interchangeable backends
   produce that pair — see [Diarization backends](#diarization-backends).
5. **Align** — WhisperX's forced aligner (not its ASR) places GPT's words on the
   timeline.
6. **Attribute** — each word is credited to the diarization speaker(s) active
   while it was spoken.
7. **Clean** — a constrained `gpt-5.6-terra` pass fixes punctuation and casing
   only, and is rejected outright if it changes a single word.

## Setup

```bash
brew install ffmpeg                              # macOS; omit if already installed
uv sync
uv run python scripts/setup_alignment.py         # builds the aligner environment
uv run python scripts/setup_diarization.py       # builds the local diarizer (default)
```

Or skip all of it and run [in Docker](#running-in-docker), which pins ffmpeg and
both torch stacks together so a host upgrade cannot break them.

Put both keys in `.env` at the project root:

```
OPENAI_API_KEY=sk-...
PYANNOTE_API_KEY=...
```

`PYANNOTE_API_KEY` is only needed by the hosted diarizer; the default local
backend uses `HF_TOKEN` instead, and neither is needed with `--no-align`. See
[Diarization backends](#diarization-backends).

`.env` is loaded automatically, so `uv run --env-file .env` is no longer needed.
Real environment variables always take precedence over `.env`.

WhisperX is **not** a project dependency. It pins torch and declares
`requires-python <3.14`, so it cannot live alongside this project. The alignment
stage runs `workers/align_worker.py` as a subprocess in a separate Python 3.12
environment. Build it once:

```bash
uv run python scripts/setup_alignment.py            # add --language de for another model
```

That creates `.venv-align` (~950 MB: torch, torchaudio, transformers) and
pre-downloads the 360 MB wav2vec2 model and the NLTK sentence tokenizer, so the
first transcription does not stall. `transcribe.py` picks the environment up
automatically.

If `.venv-align` is absent, the worker falls back to building a throwaway
environment on demand with `uv run --no-project --python 3.12 --with whisperx`.
`TRANSCRIBE_ALIGN_VENV` points at an environment built somewhere else, and
`TRANSCRIBE_ALIGN_CMD` overrides the command entirely if you manage that
environment yourself.

## Diarization backends

Diarization is the only stage with a choice of provider. Both return the same
regular and exclusive timelines, so everything downstream — alignment,
attribution, the JSON schema — is identical either way. The backend used is
recorded in `models.diarization` in the output.

| Backend | Model | Runs | Needs |
| --- | --- | --- | --- |
| `community` (default) | `pyannote/speaker-diarization-community-1` | your machine | one-time model download |
| `pyannote` | `precision-2` | pyannoteAI's API | `PYANNOTE_API_KEY` |

Select it per run or for the whole shell:

```bash
uv run python transcribe.py input/interview.m4a --diarization-backend pyannote
DIARIZATION_BACKEND=pyannote uv run python transcribe.py input/interview.m4a
```

`--diarization-backend` overrides `DIARIZATION_BACKEND`, and an unrecognized
value in either fails immediately. There is **no automatic fallback between
backends**: a missing local environment or a missing API key is an error, because
silently switching would change cost and quality without saying so.

### Setting up the local backend

Community-1 is gated. Accept the conditions on the
[model card](https://huggingface.co/pyannote/speaker-diarization-community-1),
create a token at [hf.co/settings/tokens](https://hf.co/settings/tokens), and add
it to `.env`:

```
HF_TOKEN=hf_...
```

Then build the environment once:

```bash
uv run python scripts/setup_diarization.py
```

That creates `.venv-diarize` (torch and pyannote.audio, on Python 3.12 for the
same reason the aligner is) and pre-downloads the model. `HF_TOKEN` is only
needed to *download* the gated weights; once they are cached, runs work offline
without it. As with alignment, the worker will build a throwaway environment on
demand if `.venv-diarize` is missing; `TRANSCRIBE_DIARIZE_VENV` points at one
built elsewhere, and `TRANSCRIBE_DIARIZE_CMD` overrides the command entirely.

The worker decodes audio by running `ffmpeg` and handing pyannote a waveform
rather than a path. pyannote's own decoder, torchcodec, ships prebuilt libraries
linked to one ffmpeg ABI at a hard-coded rpath, so it stops loading the moment
the host's ffmpeg moves to a different major version — which is a silent failure
that only surfaces when you try to diarize. Driving ffmpeg directly, as the rest
of the pipeline already does, removes that coupling.

Local diarization defaults to CPU. Pass `--diarization-device cuda` or
`--diarization-device mps` to use an accelerator; it is separate from `--device`,
which selects the *alignment* device.

### Which to use

Community-1 is the default. It is documented as substantially better than the
older open pyannote 3.1, and it is a true diarizer with an exclusive speaker
timeline, so it slots into the attribution logic without compromise. Running it
yourself has no recurring provider bill and no API key; hosted Community-1 still
sits behind [pyannote's monthly plans](https://www.pyannote.ai/pricing).

The cost is setup: a one-time multi-gigabyte model download, acceptance of the
model license, and CPU time per run. Choose hosted `precision-2` instead when
you would rather not carry any of that locally.

Published diarization benchmarks vary substantially by domain, so run both
against representative audio before committing to one.

OpenAI's `gpt-4o-transcribe-diarize` is deliberately not offered: it is an
ASR-plus-diarization call, so it would retranscribe the audio rather than
diarize it independently, at roughly $0.006/minute.

## Usage

```bash
uv run python transcribe.py input/interview.m4a
uv run python transcribe.py input/interview.m4a --language en --speakers 2
```

With no `--output`, results are written to `output/` **and** the readable
transcript is printed to stdout:

| File | Contents |
| --- | --- |
| `output/<name>.json` | Primary output: segments with speakers, overlap flags, and both raw and cleaned text |
| `output/<name>.txt` | Readable timestamped transcript |
| `output/<name>.raw.json` | Untouched `gpt-transcribe` responses, saved before any cleanup |

Use `-o` / `--text-output` to place either file explicitly, `--output-dir` to
change the default directory, and `--quiet` to suppress the stdout copy.

### Options

| Flag | Effect |
| --- | --- |
| `--language` | ISO-639-1 code; also selects the alignment model |
| `--speakers` | Known speaker count, passed to diarization |
| `--device` | Force the alignment device (`cpu`, `cuda`) |
| `--diarization-backend` | `community` (local, default) or `pyannote` (hosted); overrides `DIARIZATION_BACKEND` |
| `--diarization-device` | Force the local diarization device (`cpu`, `cuda`, `mps`); `community` only |
| `--no-denoise` | Keep filtering and normalization, skip noise reduction |
| `--no-preprocess` | Send the original audio unchanged |
| `--no-align` | Skip diarization and alignment entirely — no diarization credentials needed, but no speakers or timings either |
| `--no-clean` | Skip the cleanup pass; `clean_text` equals `raw_text` |

Heavy denoising is deliberately avoided: speech-enhancement artifacts can reduce
modern ASR accuracy.

## Machine-callable CLI

`transcribe.py` is the human interface. For scripts and agents, `bin/transcribe`
is a single executable that prints exactly one JSON object to stdout — on success
and on failure alike — and sends all progress to stderr. It resolves its own
interpreter, so callers never need `uv run` and the one absolute path can be
allowlisted.

```bash
bin/transcribe list-actions                       # machine-readable catalog
bin/transcribe check-setup                        # tools, credentials, backends
bin/transcribe probe --audio input/interview.m4a  # duration and size
bin/transcribe transcribe --audio input/interview.m4a --language en --dry-run
bin/transcribe transcribe --audio input/interview.m4a --language en
```

A run takes minutes, so `--detach` starts it in a background process that
survives the calling shell, returns a `job_id` immediately, and is polled:

```bash
bin/transcribe transcribe --audio input/interview.m4a --language en --detach
bin/transcribe job-status --job-id 20260806-173647-de4ad931
bin/transcribe job-list
bin/transcribe job-cancel --job-id 20260806-173647-de4ad931
```

`job-status` reports `state` (`running`, `succeeded`, `failed`, `cancelled`),
the current `phase` with a coarse `progress` estimate derived from the pipeline's
own output, and the full result once finished. Note that `ok` describes the poll
and `result.state` describes the job: polling a failed job succeeds. State lives
in `.transcribe-jobs/<job_id>/` alongside the job's `stderr.log`.

```json
{"ok": true,  "action": "probe", "result": {}, "meta": {"schema_version": "1.0"}}
{"ok": false, "action": "probe", "error": {"type": "resource_not_found",
                                           "message": "...", "details": {}},
                                 "meta": {"schema_version": "1.0"}}
```

Exit status is 0 when `ok` is true and 1 when it is false. Branch on
`error.type` — the types are stable (`invalid_arguments`, `resource_not_found`,
`authentication_failed`, `permission_denied`, `timeout`, `upstream_error`,
`rate_limited`, `conflict`, `dependency_missing`, `internal_error`); the messages
are not. `--dry-run` validates inputs and reports the plan without calling a paid
API.

Agent-facing documentation lives in [`skills/`](skills/) and is not tied to any
one assistant; see [`skills/transcribe/SKILL.md`](skills/transcribe/SKILL.md).

## Output

The readable transcript groups consecutive segments by speaker:

```text
[00:00:00]  SPEAKER_00
    Thanks for joining today. I wanted to walk you through the feedback we
    collected from the panel.

[00:00:11]  SPEAKER_01
    Sounds good.

[00:00:12]  [overlapping speech: SPEAKER_00 / SPEAKER_01]
    yes
```

The JSON keeps every decision explicit:

```json
{
  "segments": [
    {
      "start": 12.4,
      "end": 12.8,
      "speaker": null,
      "speaker_candidates": ["SPEAKER_00", "SPEAKER_01"],
      "overlap": true,
      "raw_text": "yes",
      "clean_text": "Yes",
      "aligned": true
    }
  ],
  "warnings": []
}
```

Attribution rules, applied per word:

1. exactly one speaker active → that speaker, `overlap: false`;
2. several active but one clearly dominant → that speaker, `overlap: true`,
   every candidate retained;
3. several active with no clear winner → fall back to the diarizer's
   **exclusive** timeline, which names the speaker most likely to be transcribed
   at that instant. Its answer is accepted only if it is already one of the
   candidates, so the tiebreaker can narrow an attribution but never invent one;
4. still undecided, no speaker active nearby, or the word could not be
   aligned → `speaker: null` and a warning.

`overlap: true` and `speaker_candidates` survive step 3 — crediting a speaker
never erases the fact that more than one person was talking. The readable
transcript groups by speaker so a one-word interjection does not chop a sentence
in half, and reports the overlap count in its header.

## Running in Docker

The laptop setup has one fragile joint: three Python environments, two torch
stacks, and a system ffmpeg that upgrades on its own schedule. When ffmpeg moves
to a new major version, pyannote's prebuilt decoder stops loading and local
diarization dies. The image pins all of it together.

```bash
docker compose build
docker compose run --rm warmup     # once: downloads the models into a volume
docker compose run --rm transcribe transcribe \
    --audio input/interview.m4a --language en
```

`transcribe` is the image's entrypoint, so the first argument after the service
name is the action — `check-setup`, `transcribe`, `job-status`, and the rest
behave exactly as they do on the host, and stdout is still a single JSON object.

| Path | What it is |
| --- | --- |
| `./input` | mounted read-only; recordings never enter the image |
| `./output` | mounted read-write; transcripts land on the host |
| `.env` | read at runtime; credentials are never baked into a layer |
| `models` volume | model weights and job state, kept across rebuilds |

### What the image pins

| | Version |
| --- | --- |
| ffmpeg | Debian trixie's, installed at build time |
| Project | Python 3.14 |
| Aligner (`/opt/venvs/align`) | Python 3.12, whisperx, CPU torch |
| Diarizer (`/opt/venvs/diarize`) | Python 3.12, pyannote.audio, CPU torch |

The two worker environments build as independent stages, so their torch
downloads run in parallel and neither can perturb the other's pin. They live
outside `/app` and are found through `TRANSCRIBE_ALIGN_VENV` and
`TRANSCRIBE_DIARIZE_VENV`, so bind-mounting a checkout over `/app` for
development cannot shadow them.

Weights are **not** baked in. Community-1 is gated, so shipping it would
redistribute a licensed model and would need a build-time token; `warmup`
downloads it into the `models` volume instead, using the `HF_TOKEN` from `.env`.
Do that once — afterwards the container runs without `HF_TOKEN` at all.

The image is CPU-only: Docker Desktop on macOS exposes no CUDA device, and the
CUDA wheels are several gigabytes each. On a CUDA host, run on the host rather
than in this image, or rebuild the two worker stages without `--torch-backend=cpu`.

To check the image itself, independently of any credentials:

```bash
docker compose run --rm tests
```

## Tests

```bash
uv run pytest
```

The OpenAI, pyannoteAI, Community-1, and WhisperX boundaries are mocked
independently, so the suite runs offline. It covers the synthetic crosstalk case,
chunk seams, unaligned words, missing diarization, backend selection, and every
way the cleanup pass can be rejected.
