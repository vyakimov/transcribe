# Hosted audio transcription and diarization

This script preprocesses audio with FFmpeg, then uses OpenAI's hosted
`gpt-4o-transcribe-diarize` model. No transcription or diarization model runs
locally.

## Setup

Install FFmpeg and the Python dependency, then provide an API key:

```bash
brew install ffmpeg                 # macOS; omit if already installed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```

Alternatively, when using `uv`, place `OPENAI_API_KEY=...` in `.env` and run
with `uv run --env-file .env`.

## Usage

```bash
python transcribe.py recording.m4a
python transcribe.py recording.m4a --output transcript.txt
```

By default the script:

1. Applies a conservative 60 Hz high-pass filter, light stationary-noise
   reduction, and loudness normalization.
2. Converts the result to lossless, mono 24 kHz FLAC.
3. Chunks when either the 25 MB upload limit or the model's 1,400-second limit
   would be exceeded. Targets are deliberately lower than both limits.
4. Prefers a detected silence near each balanced chunk boundary. If no silence
   is available, it uses a five-second overlap and assigns each timestamped
   segment to only one side of the seam.
5. Uploads each chunk and merges the results onto the original timeline.

Use `--no-denoise` if the source is already clean, or `--no-preprocess` to send
the original audio unchanged. Heavy denoising is intentionally avoided because
speech-enhancement artifacts can reduce modern ASR accuracy.

For independently processed chunks, speaker labels are namespaced (for example,
`chunk_01_A`). The model cannot guarantee that `A` in one API request is the
same person as `A` in another request without known-speaker reference clips.

Example output:

```text
[00:00:00.000 --> 00:00:04.700] chunk_01_A: Thanks for joining.
[00:00:04.700 --> 00:00:08.100] chunk_01_B: Happy to be here.
```
