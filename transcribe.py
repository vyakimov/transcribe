#!/usr/bin/env python3
"""Transcribe, diarize, align, and clean an audio file using hosted models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcribe_pipeline.config import (
    DEFAULT_OUTPUT_DIR,
    DIARIZATION_BACKEND_COMMUNITY,
    DIARIZATION_BACKENDS,
    TRANSCRIPTION_BACKEND_ASSEMBLYAI,
    TRANSCRIPTION_BACKENDS,
    load_environment,
    resolve_diarization_backend,
    resolve_transcription_backend,
)
from transcribe_pipeline.pipeline import Options, run, write_raw_responses
from transcribe_pipeline.render import render_json, render_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce a cleaned, speaker-attributed, timestamped transcript."
    )
    parser.add_argument("audio", type=Path, help="Audio file to transcribe")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the JSON transcript here (default: <output-dir>/<name>.json)",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Write the readable transcript here (default: <output-dir>/<name>.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used for default output paths (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--language", help="ISO-639-1 code of the spoken language, e.g. en")
    parser.add_argument(
        "--speakers",
        type=int,
        help="Known number of speakers, passed to diarization when you know it",
    )
    parser.add_argument("--device", help="Force the alignment device, e.g. cpu or cuda")
    parser.add_argument(
        "--transcription-backend",
        help=(
            f"Which transcriber to use ({', '.join(TRANSCRIPTION_BACKENDS)}): "
            "'assemblyai' (default) gets words, timings, and speakers from one "
            "AssemblyAI job; 'gpt' uses gpt-transcribe with a separate diarizer, "
            "forced alignment, and cleanup. Overrides TRANSCRIPTION_BACKEND."
        ),
    )
    parser.add_argument(
        "--diarization-backend",
        help=(
            "With --transcription-backend gpt only. "
            f"Which diarizer to use ({', '.join(DIARIZATION_BACKENDS)}): 'community' "
            "runs Community-1 locally (default), 'pyannote' is the hosted "
            "precision-2 API, 'assemblyai' uses AssemblyAI speaker labels. "
            "Overrides DIARIZATION_BACKEND."
        ),
    )
    parser.add_argument(
        "--diarization-device",
        help="Force the local diarization device, e.g. cpu, cuda, or mps (community backend)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip high-pass filtering, light denoising, and loudness normalization",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Keep filtering and normalization but disable noise reduction",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip diarization and alignment; produces text without speakers or timings",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the punctuation cleanup pass; clean_text will equal raw_text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the readable transcript to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")

    transcription = resolve_transcription_backend(args.transcription_backend)
    backend = resolve_diarization_backend(
        args.diarization_backend, transcription_backend=transcription
    )
    if transcription == TRANSCRIPTION_BACKEND_ASSEMBLYAI and (
        args.no_preprocess or args.no_denoise or args.no_clean or args.device
    ):
        print(
            "Preprocessing, cleanup, and alignment options have no effect with the "
            "assemblyai transcription backend, which runs none of those stages.",
            file=sys.stderr,
        )
    if args.diarization_device and backend != DIARIZATION_BACKEND_COMMUNITY:
        print(
            f"--diarization-device has no effect with the {backend} backend; "
            "hosted diarizers run on the provider's own hardware.",
            file=sys.stderr,
        )

    document, raw_responses = run(
        Options(
            audio=args.audio,
            language=args.language,
            speakers=args.speakers,
            preprocess=not args.no_preprocess,
            denoise=not args.no_denoise,
            align=not args.no_align,
            clean=not args.no_clean,
            device=args.device,
            transcription_backend=transcription,
            diarization_backend=backend,
            diarization_device=args.diarization_device,
        )
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem

    json_path = args.output or output_dir / f"{stem}.json"
    text_path = args.text_output or output_dir / f"{stem}.txt"
    for path in (json_path, text_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(render_json(document), encoding="utf-8")
    readable = render_text(document)
    text_path.write_text(readable, encoding="utf-8")
    write_raw_responses(output_dir / f"{stem}.raw.json", raw_responses)

    print(f"JSON transcript:     {json_path}", file=sys.stderr)
    print(f"Readable transcript: {text_path}", file=sys.stderr)

    if not args.quiet:
        print(readable, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
