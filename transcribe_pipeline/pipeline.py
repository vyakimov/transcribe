"""End-to-end orchestration of the transcription pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from transcribe_pipeline import alignment, asr, attribution, cleanup, diarization, media
from transcribe_pipeline.config import (
    CLEANUP_MODEL,
    DIARIZATION_MODEL,
    TRANSCRIBE_MODEL,
    require_env,
)
from transcribe_pipeline.models import DiarizationResult, TranscriptDocument, Word


@dataclass
class Options:
    audio: Path
    language: str | None = None
    speakers: int | None = None
    preprocess: bool = True
    denoise: bool = True
    align: bool = True
    clean: bool = True
    device: str | None = None


def _alignment_requests(chunks: list[media.Chunk]) -> list[alignment.AlignmentRequest]:
    requests: list[alignment.AlignmentRequest] = []
    for index, chunk in enumerate(chunks):
        duration = media.probe(chunk.path).duration
        requests.append(
            alignment.AlignmentRequest(
                id=index,
                start=chunk.audio_start,
                end=chunk.audio_start + duration,
                text="",
            )
        )
    return requests


def run(options: Options) -> tuple[TranscriptDocument, list[dict]]:
    media.require_ffmpeg()
    warnings: list[str] = []

    openai_key = require_env("OPENAI_API_KEY", purpose="transcription and cleanup")
    pyannote_key = (
        require_env("PYANNOTE_API_KEY", purpose="speaker diarization") if options.align else ""
    )
    client = OpenAI(api_key=openai_key)

    with tempfile.TemporaryDirectory(prefix="transcribe-") as temporary:
        workspace = Path(temporary)
        source = options.audio
        if options.preprocess:
            source = workspace / "processed.flac"
            media.preprocess(options.audio, source, denoise=options.denoise)

        info = media.probe(source)
        chunks = media.extract_chunks(source, workspace)
        transcripts = asr.transcribe_chunks(client, chunks, language=options.language)
        raw_responses = [
            {"chunk": index, "audio_start": chunk.audio_start, "response": transcript.raw_response}
            for index, (chunk, transcript) in enumerate(zip(chunks, transcripts))
        ]

        diarized = DiarizationResult(turns=())
        words: list[Word] = []

        if options.align:
            diarized = diarization.diarize(
                source, api_key=pyannote_key, num_speakers=options.speakers
            )

            requests = _alignment_requests(chunks)
            for request, transcript in zip(requests, transcripts):
                request.text = transcript.text

            result = alignment.align(
                source, requests, language=options.language, device=options.device
            )
            warnings.extend(result.warnings)

            for index, chunk in enumerate(chunks):
                chunk_words = result.words_by_id.get(index, [])
                if not chunk_words and transcripts[index].text:
                    warnings.append(f"chunk {index + 1} returned no aligned words")
                    chunk_words = [Word(text=token) for token in transcripts[index].text.split()]
                words.extend(attribution.keep_owned(chunk_words, chunk.own_start, chunk.own_end))
        else:
            warnings.append(
                "alignment and diarization were skipped (--no-align), so no speaker "
                "attribution or word timing is available"
            )
            for chunk, transcript in zip(chunks, transcripts):
                words.extend(Word(text=token) for token in transcript.text.split())

    words = attribution.attribute_words(words, diarized.turns, diarized.exclusive_turns)
    segments = attribution.group_segments(words)

    unattributed = sum(1 for word in words if word.aligned and word.speaker is None)
    if unattributed:
        warnings.append(f"{unattributed} aligned word(s) could not be attributed to one speaker")

    if options.clean and segments:
        warnings.extend(cleanup.clean_segments(client, segments))
    elif not options.clean:
        warnings.append("cleanup pass was skipped (--no-clean); clean_text equals raw_text")

    document = TranscriptDocument(
        audio=options.audio,
        duration=info.duration,
        language=options.language,
        speakers=diarized.speakers,
        segments=segments,
        warnings=warnings,
        models={
            "transcription": TRANSCRIBE_MODEL,
            "diarization": DIARIZATION_MODEL if options.align else "none",
            "cleanup": CLEANUP_MODEL if options.clean else "none",
        },
        exclusive_turns=diarized.exclusive_turns,
    )
    return document, raw_responses


def write_raw_responses(path: Path, raw_responses: list[dict]) -> None:
    path.write_text(json.dumps(raw_responses, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw transcription responses: {path}", file=sys.stderr)
