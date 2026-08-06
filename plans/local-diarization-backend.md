# Add local Community-1 diarization while preserving pyannoteAI

## Summary

- Keep the existing pyannoteAI `precision-2` plumbing and make it the default.
- Add selectable local `pyannote/speaker-diarization-community-1`.
- Select it with `--diarization-backend community` or `DIARIZATION_BACKEND=community`.
- Recommend Community-1 because it is a true diarizer, supports exclusive speaker timelines, and fits the existing independent `gpt-transcribe` architecture. It is currently documented as substantially better than pyannote 3.1. [Model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- Do not add OpenAI diarization in this iteration: `gpt-4o-transcribe-diarize` is an ASR-plus-diarization call, so it would retranscribe the audio and cost approximately $0.006/minute. [OpenAI diarization docs](https://developers.openai.com/api/docs/guides/speech-to-text#speaker-diarization), [pricing](https://developers.openai.com/api/docs/pricing)

## Implementation changes

- Add backend selection to `Options`, the CLI, and configuration. CLI overrides `DIARIZATION_BACKEND`; unknown values fail clearly.
- Leave `transcribe_pipeline/diarization.py` and its pyannoteAI upload, polling, confidence, and exclusive-output handling intact.
- Add a local diarization runner plus `workers/diarize_worker.py`. The worker will:
  - run Community-1 over the complete preprocessed file;
  - forward `num_speakers` when supplied;
  - return both regular and `exclusive_speaker_diarization` turns;
  - emit provider-neutral JSON consumed as `DiarizationResult`.
- Keep pyannote dependencies out of the main runtime by adding an isolated `.venv-diarize` setup script, similar to the existing WhisperX environment. Require `HF_TOKEN` only for downloading the gated model; cached/offline runs can reuse the downloaded model.
- Add optional `--diarization-device`; default to CPU and pass an explicitly selected device to the worker.
- Record `"precision-2"` or `"community-1"` in output metadata without changing the transcript, alignment, attribution, or JSON schemas.
- Document setup, Hugging Face model acceptance, backend selection, and the fact that hosted Community-1 still sits behind pyannote’s monthly plans, while self-hosting has no recurring provider bill. [pyannote pricing](https://www.pyannote.ai/pricing)

## Tests and acceptance

- Preserve all existing pyannoteAI tests unchanged where possible.
- Add tests for:
  - local worker response parsing, including regular and exclusive turns;
  - speaker-count and device forwarding;
  - backend selection and credential requirements;
  - Community-1 metadata;
  - missing worker/setup/model errors;
  - `--no-align` bypassing every diarization backend.
- Verify that Community-1’s exclusive turns continue to resolve overlap cases through the existing attribution logic.
- Run the full offline test suite with mocked provider boundaries.
- Perform one manual smoke test against the existing interview fixture using both backends and compare speaker counts, overlap handling, and attribution coverage.

## Assumptions

- `pyannote` remains the default until explicitly switched.
- There is no automatic fallback: a missing local setup or API key should fail clearly rather than silently changing model, cost, or quality.
- OpenAI’s `gpt-4o-transcribe-diarize` remains a documented future option, but is outside this first implementation.
