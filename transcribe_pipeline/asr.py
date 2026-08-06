"""Transcription with `gpt-transcribe`.

Whole upload-sized chunks are transcribed in one request. Transcribing isolated
diarization turns instead would expose the model to crosstalk duplication and
speaker misattribution, so speaker labels are never sourced from here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from openai import OpenAI

from transcribe_pipeline.config import TRANSCRIBE_MODEL
from transcribe_pipeline.models import Chunk


@dataclass(frozen=True)
class ChunkTranscript:
    chunk: Chunk
    text: str
    raw_response: dict


def transcribe_chunk(client: OpenAI, chunk: Chunk, *, language: str | None) -> ChunkTranscript:
    with chunk.path.open("rb") as audio:
        response = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio,
            response_format="json",
            **({"language": language} if language else {}),
        )

    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return ChunkTranscript(chunk=chunk, text=(response.text or "").strip(), raw_response=raw)


def transcribe_chunks(
    client: OpenAI, chunks: list[Chunk], *, language: str | None = None
) -> list[ChunkTranscript]:
    transcripts: list[ChunkTranscript] = []
    for index, chunk in enumerate(chunks, 1):
        print(f"Transcribing chunk {index}/{len(chunks)}...", file=sys.stderr)
        transcripts.append(transcribe_chunk(client, chunk, language=language))
    return transcripts
