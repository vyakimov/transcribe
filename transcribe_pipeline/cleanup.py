"""Constrained text-only cleanup pass.

The cleanup model may only touch punctuation, capitalization, whitespace, and
formatting. Every returned segment is verified to contain exactly the same word
sequence as the transcript it came from; anything else falls back to the raw
text. Both versions are kept permanently, so a bad cleanup can never destroy the
model's actual output.
"""

from __future__ import annotations

import re
import sys

from openai import OpenAI
from pydantic import BaseModel

from transcribe_pipeline.config import CLEANUP_MODEL
from transcribe_pipeline.models import Segment

BATCH_SIZE = 25
_NORMALIZE = re.compile(r"[^0-9a-z]+")

INSTRUCTIONS = """\
You are a transcript formatter. For each numbered segment you receive, return \
the same segment with corrected punctuation, capitalization, whitespace, and \
formatting.

Absolute constraints:
- Never add, remove, reorder, or paraphrase words.
- Never expand or contract words (no "gonna" -> "going to", no "25" -> "twenty five").
- Never merge or split segments, and never move text between segments.
- Never invent speaker labels, timestamps, or commentary.
- If a segment is already correct, return it unchanged.
- If a segment is unintelligible or ambiguous, leave it exactly as given rather \
than guessing.

Return one entry per input segment, preserving the given index.
"""


class CleanedSegment(BaseModel):
    index: int
    clean_text: str


class CleanupResponse(BaseModel):
    segments: list[CleanedSegment]


def tokens(text: str) -> list[str]:
    """Reduce text to the word sequence that cleanup is forbidden from changing."""
    return [token for token in (_NORMALIZE.sub("", part.lower()) for part in text.split()) if token]


def preserves_words(raw: str, cleaned: str) -> bool:
    return tokens(raw) == tokens(cleaned)


def _clean_batch(
    client: OpenAI, batch: list[tuple[int, Segment]], warnings: list[str]
) -> dict[int, str]:
    payload = "\n".join(f"[{index}] {segment.raw_text}" for index, segment in batch)

    try:
        response = client.responses.parse(
            model=CLEANUP_MODEL,
            instructions=INSTRUCTIONS,
            input=payload,
            reasoning={"effort": "low"},
            text_format=CleanupResponse,
        )
        parsed = response.output_parsed
    except Exception as error:  # noqa: BLE001 - never let cleanup lose a transcript
        warnings.append(f"cleanup request failed, using raw text: {error}")
        return {}

    if parsed is None:
        warnings.append("cleanup returned no parsable output, using raw text")
        return {}

    accepted: dict[int, str] = {}
    expected = {index for index, _ in batch}
    for item in parsed.segments:
        if item.index not in expected:
            warnings.append(f"cleanup returned unknown segment index {item.index}, ignored")
            continue
        accepted[item.index] = item.clean_text

    return accepted


def clean_segments(client: OpenAI, segments: list[Segment]) -> list[str]:
    """Return warnings; segments are updated in place with validated clean text."""
    warnings: list[str] = []
    indexed = list(enumerate(segments))
    rejected = 0

    for offset in range(0, len(indexed), BATCH_SIZE):
        batch = indexed[offset : offset + BATCH_SIZE]
        print(
            f"Cleaning segments {offset + 1}-{offset + len(batch)} of {len(indexed)}...",
            file=sys.stderr,
        )
        accepted = _clean_batch(client, batch, warnings)

        for index, segment in batch:
            candidate = accepted.get(index)
            if candidate is None:
                continue
            candidate = candidate.strip()
            if candidate and preserves_words(segment.raw_text, candidate):
                segment.clean_text = candidate
            else:
                rejected += 1

    if rejected:
        warnings.append(
            f"{rejected} cleaned segment(s) changed the word sequence and were "
            "rejected in favour of the raw text"
        )
    return warnings
