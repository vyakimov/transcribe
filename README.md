# Hosted transcription with speaker attribution

Transcribes an audio file with OpenAI's `gpt-transcribe`, diarizes it
independently with pyannoteAI `precision-2`, and reconciles the two with
WhisperX's wav2vec2 forced aligner. Transcription and diarization never see each
other's output, so crosstalk cannot duplicate text or silently reassign it.

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
4. **Diarize** — the whole file goes to pyannoteAI `precision-2` in one job, so
   speaker labels stay consistent end to end. Regular diarization is canonical
   because it preserves overlapping speech; the exclusive view is kept in the
   JSON and used only to break ties the canonical view leaves open.
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
```

Put both keys in `.env` at the project root:

```
OPENAI_API_KEY=sk-...
PYANNOTE_API_KEY=...
```

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
Set `TRANSCRIBE_ALIGN_CMD` to override the command entirely if you manage that
environment yourself.

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
| `--no-denoise` | Keep filtering and normalization, skip noise reduction |
| `--no-preprocess` | Send the original audio unchanged |
| `--no-align` | Skip diarization and alignment entirely — no `PYANNOTE_API_KEY` needed, but no speakers or timings either |
| `--no-clean` | Skip the cleanup pass; `clean_text` equals `raw_text` |

Heavy denoising is deliberately avoided: speech-enhancement artifacts can reduce
modern ASR accuracy.

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
3. several active with no clear winner → fall back to pyannoteAI's **exclusive**
   diarization, which names the speaker most likely to be transcribed at that
   instant. Its answer is accepted only if it is already one of the candidates,
   so the tiebreaker can narrow an attribution but never invent one;
4. still undecided, no speaker active nearby, or the word could not be
   aligned → `speaker: null` and a warning.

`overlap: true` and `speaker_candidates` survive step 3 — crediting a speaker
never erases the fact that more than one person was talking. The readable
transcript groups by speaker so a one-word interjection does not chop a sentence
in half, and reports the overlap count in its header.
Published diarization benchmarks vary substantially by domain, so representative
local audio remains the final quality check.

## Tests

```bash
uv run pytest
```

The OpenAI, pyannoteAI, and WhisperX boundaries are mocked independently, so the
suite runs offline. It covers the synthetic crosstalk case, chunk seams,
unaligned words, missing diarization, and every way the cleanup pass can be
rejected.
