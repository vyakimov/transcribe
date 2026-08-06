"""FFmpeg preprocessing, silence detection, and upload-sized chunking."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from transcribe_pipeline.models import Chunk, MediaInfo

# OpenAI documents a 25 MB upload limit. These lower targets leave room for
# container overhead and small bitrate variations between chunks.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TARGET_UPLOAD_BYTES = 23 * 1024 * 1024
MAX_MODEL_SECONDS = 1400.0
TARGET_CHUNK_SECONDS = 1350.0
FALLBACK_OVERLAP_SECONDS = 5.0


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else None,
    )


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise SystemExit(f"Missing required command(s): {', '.join(missing)}")


def probe(path: Path) -> MediaInfo:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    data = json.loads(result.stdout)["format"]
    return MediaInfo(duration=float(data["duration"]), size=int(data["size"]))


def preprocess(source: Path, destination: Path, *, denoise: bool) -> None:
    # Keep this intentionally gentle: heavy speech enhancement can erase cues
    # useful to both transcription and speaker identification.
    filters = ["highpass=f=60"]
    if denoise:
        filters.append("afftdn=nr=6:nf=-50:tn=1")
    filters.append("loudnorm=I=-18:LRA=11:TP=-1.5")

    print("Preprocessing audio with FFmpeg...", file=sys.stderr)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "flac",
            str(destination),
        ]
    )


def silence_midpoints(path: Path) -> list[float]:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-35dB:d=0.30",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", result.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end: ([0-9.]+)", result.stderr)]
    return [(start + end) / 2 for start, end in zip(starts, ends)]


def plan_boundaries(
    info: MediaInfo, silences: list[float], safety: float
) -> tuple[list[float], set[float]]:
    bytes_per_second = max(1.0, info.size / info.duration)
    max_span = min(
        TARGET_CHUNK_SECONDS,
        TARGET_UPLOAD_BYTES * safety / bytes_per_second,
    )
    boundaries = [0.0]
    hard_boundaries: set[float] = set()

    while info.duration - boundaries[-1] > max_span:
        start = boundaries[-1]
        chunks_left = math.ceil((info.duration - start) / max_span)
        ideal = (info.duration - start) / chunks_left
        target = start + ideal
        lower = start + ideal * 0.60
        upper = min(start + max_span, target + 60.0)
        candidates = [point for point in silences if lower <= point <= upper]

        if candidates:
            boundary = min(candidates, key=lambda point: abs(point - target))
        else:
            boundary = target
            hard_boundaries.add(boundary)
        boundaries.append(boundary)

    boundaries.append(info.duration)
    return boundaries, hard_boundaries


def extract_chunks(source: Path, directory: Path) -> list[Chunk]:
    info = probe(source)
    if info.duration <= MAX_MODEL_SECONDS and info.size <= MAX_UPLOAD_BYTES:
        return [Chunk(source, 0.0, 0.0, info.duration)]

    print(
        f"Chunking {info.duration:.1f}s / {info.size / 1_000_000:.1f}MB audio...",
        file=sys.stderr,
    )
    silences = silence_midpoints(source)

    # Retry with increasingly conservative size estimates if a VBR region is
    # unexpectedly larger than the whole-file average.
    for attempt in range(5):
        attempt_dir = directory / f"chunks-{attempt}"
        attempt_dir.mkdir()
        boundaries, hard_boundaries = plan_boundaries(info, silences, 0.8**attempt)
        chunks: list[Chunk] = []

        for index, (own_start, own_end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            left_overlap = FALLBACK_OVERLAP_SECONDS if own_start in hard_boundaries else 0.0
            right_overlap = FALLBACK_OVERLAP_SECONDS if own_end in hard_boundaries else 0.0
            audio_start = max(0.0, own_start - left_overlap)
            audio_end = min(info.duration, own_end + right_overlap)
            destination = attempt_dir / f"chunk-{index:03d}.flac"
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{audio_start:.6f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{audio_end - audio_start:.6f}",
                    # Re-encode losslessly instead of stream-copying. A sliced
                    # FLAC stream can retain the source's total-sample header,
                    # making ffprobe (and potentially the API) report the full
                    # original duration for every chunk.
                    "-c:a",
                    "flac",
                    str(destination),
                ]
            )
            chunks.append(Chunk(destination, audio_start, own_start, own_end))

        if all(
            chunk.path.stat().st_size <= MAX_UPLOAD_BYTES
            and probe(chunk.path).duration <= MAX_MODEL_SECONDS
            for chunk in chunks
        ):
            return chunks

        shutil.rmtree(attempt_dir)

    raise RuntimeError("Could not produce chunks within the API limits")
