# Revised GPT-Transcribe diarization plan

## Summary

Use `gpt-transcribe` as the authoritative transcription model, but do not transcribe isolated diarization turns. That approach is vulnerable to crosstalk duplication and speaker misattribution.

Instead:

1. Transcribe complete upload-sized chunks with `gpt-transcribe`.
2. Run overlap-capable diarization independently with hosted pyannoteAI `precision-2`.
3. Use WhisperX’s wav2vec2 forced-alignment component—not its ASR—to place GPT’s words on the audio timeline.
4. Attribute each aligned word to the overlapping diarization speaker(s).
5. Run a constrained LLM cleanup pass only on the resulting text.

OpenAI documents `gpt-transcribe` as the recommended ordinary transcription model, while timestamped output is unavailable for it. <https://developers.openai.com/api/docs/guides/speech-to-text>

## Key changes

- Keep the existing preprocessing and chunking logic, but call `gpt-transcribe` once per chunk. Save each raw GPT response before any cleanup.
- Submit the same audio to pyannoteAI `precision-2` with confidence output enabled. Request both regular and exclusive diarization, but use regular diarization as canonical because it preserves overlapping speech. The exclusive result is retained only as a convenience/diagnostic view. Exclusive diarization is documented as removing overlapping speech for easier ASR reconciliation. <https://docs.pyannote.ai/tutorials/diarization-asr-merge>
- Use WhisperX alignment with the GPT-produced text as input. WhisperX supports aligning externally supplied transcripts and produces word-level timestamps. <https://github.com/m-bain/whisperX>
- For every aligned word:
  - assign one speaker when only one diarization interval overlaps;
  - set `overlap: true` and retain `speaker_candidates` when multiple speakers are active;
  - use `speaker: null` when attribution is genuinely ambiguous;
  - retain unaligned GPT words with an alignment warning rather than guessing.
- Stitch chunk results by timestamp, resolving only chunk-overlap duplicates deterministically.

The primary JSON output should preserve both raw and cleaned text:

```json
{
  "segments": [
    {
      "start": 12.4,
      "end": 12.8,
      "speaker": "SPEAKER_01",
      "speaker_candidates": ["SPEAKER_00", "SPEAKER_01"],
      "overlap": true,
      "raw_text": "yes",
      "clean_text": "Yes"
    }
  ],
  "warnings": []
}
```

## Cleanup pass

Run a text-only cleanup pass using the current flagship text model, `gpt-5.6-sol`, through the Responses API. <https://developers.openai.com/api/docs/guides/latest-model>

The cleanup model may change only punctuation, capitalization, whitespace, and formatting. It must not:

- add, remove, reorder, or paraphrase words;
- change timestamps, speakers, overlap flags, or candidate speakers;
- merge text across speaker boundaries;
- resolve ambiguous crosstalk by guessing.

Validate the result with token-order checks and JSON schema validation. If validation fails, use the raw text. Keep both `raw_text` and `clean_text` permanently available.

## Interfaces and dependencies

- Add `PYANNOTE_API_KEY` alongside `OPENAI_API_KEY`.
- Keep JSON as the primary `--output` format.
- Add `--text-output` for the readable timestamped rendering.
- Preserve optional language and known-speaker-count arguments.
- Add WhisperX and its alignment dependencies as an isolated alignment worker; use a Python 3.12 environment if they are incompatible with the project’s Python 3.14 runtime.
- Update `transcribe.py`, dependency metadata, README documentation, and tests.

## Tests and acceptance criteria

- Synthetic crosstalk fixture: main speaker continues while another says “yes”; verify the interjection is not duplicated and is marked as overlapping/possibly ambiguous.
- Verify single-speaker words, simultaneous words, unaligned words, chunk seams, and missing diarization.
- Mock OpenAI, pyannoteAI, and WhisperX boundaries independently.
- Verify cleanup rejects word additions, deletions, reordering, and metadata changes.
- Run an opt-in end-to-end test on the existing interview fixtures and manually inspect crosstalk sections.

Unresolvable overlap should remain explicitly flagged rather than being silently “cleaned up.” Published diarization benchmarks vary substantially by domain, so representative local audio remains the final quality check. <https://arxiv.org/abs/2507.16136>
