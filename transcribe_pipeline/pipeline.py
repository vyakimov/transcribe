"""End-to-end orchestration of the transcription pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from transcribe_pipeline import (
    alignment,
    asr,
    attribution,
    cleanup,
    diarization,
    local_diarization,
    media,
)
from transcribe_pipeline.config import (
    CLEANUP_MODEL,
    DEFAULT_DIARIZATION_BACKEND,
    DIARIZATION_BACKEND_COMMUNITY,
    DIARIZATION_BACKEND_PYANNOTE,
    TRANSCRIBE_MODEL,
    diarization_model_name,
    require_env,
    validate_diarization_backend,
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
    diarization_backend: str = DEFAULT_DIARIZATION_BACKEND
    diarization_device: str | None = None


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

    backend = validate_diarization_backend(
        options.diarization_backend, source="the diarization backend"
    )

    openai_key = require_env("OPENAI_API_KEY", purpose="transcription and cleanup")
    # Only the hosted backend needs an API key, and only when diarization runs at
    # all. The local backend needs HF_TOKEN just to download the gated model, so
    # it is left to the worker rather than demanded up front.
    hosted_diarization = options.align and backend == DIARIZATION_BACKEND_PYANNOTE
    pyannote_key = (
        require_env("PYANNOTE_API_KEY", purpose="hosted pyannoteAI diarization")
        if hosted_diarization
        else ""
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
            if backend == DIARIZATION_BACKEND_COMMUNITY:
                diarized = local_diarization.diarize(
                    source,
                    num_speakers=options.speakers,
                    device=options.diarization_device,
                )
            else:
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
            "diarization": diarization_model_name(backend) if options.align else "none",
            "cleanup": CLEANUP_MODEL if options.clean else "none",
        },
        exclusive_turns=diarized.exclusive_turns,
    )
    return document, raw_responses


def write_raw_responses(path: Path, raw_responses: list[dict]) -> None:
    path.write_text(json.dumps(raw_responses, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw transcription responses: {path}", file=sys.stderr)
