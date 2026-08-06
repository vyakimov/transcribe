"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    size: int


@dataclass(frozen=True)
class Chunk:
    """An upload-sized slice of the preprocessed audio.

    `audio_start` is where the extracted file begins on the original timeline.
    `own_start`/`own_end` bound the region this chunk is authoritative for, which
    is narrower than the audio whenever a fallback overlap was used.
    """

    path: Path
    audio_start: float
    own_start: float
    own_end: float


@dataclass(frozen=True)
class Turn:
    """One diarization interval."""

    speaker: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class DiarizationResult:
    turns: tuple[Turn, ...]
    exclusive_turns: tuple[Turn, ...] = ()

    @property
    def speakers(self) -> list[str]:
        return sorted({turn.speaker for turn in self.turns})


@dataclass
class Word:
    """A single token of GPT transcript text, optionally placed on the timeline."""

    text: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None
    speaker_candidates: list[str] = field(default_factory=list)
    overlap: bool = False

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class TranscriptDocument:
    """Everything the pipeline produces for one input file."""

    audio: Path
    duration: float
    language: str | None
    speakers: list[str]
    segments: list["Segment"]
    warnings: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    exclusive_turns: tuple["Turn", ...] = ()


@dataclass
class Segment:
    """A run of consecutive words sharing one speaker attribution."""

    start: float | None
    end: float | None
    speaker: str | None
    speaker_candidates: list[str]
    overlap: bool
    raw_text: str
    clean_text: str
    aligned: bool = True

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "speaker_candidates": self.speaker_candidates,
            "overlap": self.overlap,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "aligned": self.aligned,
        }
