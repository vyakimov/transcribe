#!/usr/bin/env python3
"""Isolated WhisperX forced-alignment worker.

Runs in its own Python 3.12 environment because whisperx pins torch and
declares `requires-python <3.14`. Only WhisperX's wav2vec2 alignment component
is used; its ASR is never invoked, because the transcript text is authoritative
and comes from `gpt-transcribe`.

Usage: align_worker.py <request.json> <response.json>

Request:
    {"audio": str, "language": "en", "device": "cpu"|"cuda"|null,
     "items": [{"id": int, "start": float, "end": float, "text": str}, ...]}

Response:
    {"items": [{"id": int, "words": [{"word","start","end","score"}],
                "warnings": [str, ...]}, ...]}

`gpt-transcribe` produces no timestamps, so each item's text must be placed
against a potentially very long audio region. Aligning it in one pass is not
viable: WhisperX builds a dense frames x tokens trellis per segment, which for a
20-minute region needs tens of gigabytes. Instead the region is walked in
overlapping windows that carry more audio than text, so the CTC path always has
slack to absorb the tail, and only the confidently-placed prefix of each window
is committed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

WINDOW_SECONDS = 60.0
# Text is sized to fill only part of the window; the remaining audio is slack
# that trailing CTC blanks can absorb, which keeps the tail from dragging
# earlier words out of position.
TEXT_SECONDS = 40.0
COMMIT_SECONDS = 35.0
INITIAL_WORDS_PER_SECOND = 2.5
MIN_WORDS_PER_SECOND = 1.0
MAX_WORDS_PER_SECOND = 6.0
MAX_WORDS_PER_WINDOW = 2000
MIN_SCORE = 0.0
# No spoken word lasts this briefly. Durations below it mean CTC had to squeeze
# tokens against the end of a window, which is what happens when the transcript
# claims more speech than the audio in that region contains.
IMPLAUSIBLE_WORD_SECONDS = 0.02

_NORMALIZE = re.compile(r"[^0-9a-z]+")


def normalize(word: str) -> str:
    return _NORMALIZE.sub("", word.lower())


def pair_aligned(batch: list[str], aligned: list[dict]) -> list[dict | None]:
    """Map WhisperX's word list back onto the words that were submitted.

    WhisperX drops words that reduce to nothing after cleaning, so its output is
    a subsequence of the batch rather than a positional match.
    """
    paired: list[dict | None] = [None] * len(batch)
    normalized = [normalize(word) for word in batch]
    cursor = 0

    for item in aligned:
        target = normalize(str(item.get("word", "")))
        if not target:
            continue
        index = cursor
        while index < len(batch) and normalized[index] != target:
            index += 1
        if index < len(batch):
            paired[index] = item
            cursor = index + 1

    return paired


def usable(entry: dict | None) -> bool:
    if not entry:
        return False
    start, end = entry.get("start"), entry.get("end")
    if start is None or end is None:
        return False
    return float(end) >= float(start)


def align_item(item: dict, model, metadata, audio, device, whisperx) -> dict:
    words = item["text"].split()
    region_start = float(item["start"])
    region_end = float(item["end"])
    warnings: list[str] = []
    placed: list[dict] = []

    cursor = region_start
    index = 0
    rate = INITIAL_WORDS_PER_SECOND

    while index < len(words):
        if cursor >= region_end:
            # Ran out of audio before running out of text.
            for word in words[index:]:
                placed.append({"word": word})
            warnings.append(
                f"{len(words) - index} word(s) had no audio left to align against"
            )
            break

        window_end = min(cursor + WINDOW_SECONDS, region_end)
        is_final = window_end >= region_end - 1e-6

        if is_final:
            take = min(len(words) - index, MAX_WORDS_PER_WINDOW)
        else:
            take = max(1, min(int(TEXT_SECONDS * rate), len(words) - index))

        batch = words[index : index + take]
        segment = {"text": " ".join(batch), "start": cursor, "end": window_end}

        try:
            result = whisperx.align(
                [segment], model, metadata, audio, device, return_char_alignments=False
            )
            paired = pair_aligned(batch, result.get("word_segments") or [])
        except Exception as error:  # noqa: BLE001 - degrade rather than lose the text
            warnings.append(f"alignment failed near {cursor:.1f}s: {error}")
            paired = [None] * len(batch)

        commit_until = window_end if is_final else cursor + COMMIT_SECONDS
        committed = 0
        last_end = cursor
        for offset, entry in enumerate(paired):
            if usable(entry) and float(entry["end"]) > commit_until:
                break
            committed = offset + 1
            if usable(entry):
                last_end = max(last_end, float(entry["end"]))

        if committed == 0:
            # Nothing landed in the committed window. Rather than retry the same
            # span forever, release the words this window was meant to cover as
            # unaligned and step past it.
            committed = max(1, min(int(COMMIT_SECONDS * rate), len(words) - index))
            paired = [None] * committed
            last_end = min(cursor + COMMIT_SECONDS, region_end)
            warnings.append(f"no words could be aligned between {cursor:.1f}s and {last_end:.1f}s")

        for offset in range(committed):
            word = batch[offset] if offset < len(batch) else words[index + offset]
            entry = paired[offset] if offset < len(paired) else None
            if usable(entry):
                placed.append(
                    {
                        "word": word,
                        "start": round(float(entry["start"]), 3),
                        "end": round(float(entry["end"]), 3),
                        "score": round(float(entry.get("score") or MIN_SCORE), 3),
                    }
                )
            else:
                placed.append({"word": word})

        span = max(last_end - cursor, 1e-3)
        observed = committed / span
        if MIN_WORDS_PER_SECOND <= observed <= MAX_WORDS_PER_SECOND:
            rate = 0.5 * rate + 0.5 * observed

        index += committed
        # Never move backwards, and always make progress even if the last
        # committed word ended exactly where the window began.
        cursor = max(last_end, cursor + 1e-3)

    unaligned = sum(1 for word in placed if "start" not in word)
    if unaligned:
        warnings.append(f"{unaligned} of {len(placed)} word(s) could not be aligned")

    crammed = sum(
        1
        for word in placed
        if "start" in word and word["end"] - word["start"] < IMPLAUSIBLE_WORD_SECONDS
    )
    if crammed:
        warnings.append(
            f"{crammed} word(s) were placed with implausibly short durations; the "
            "transcript and the audio likely disagree in that region"
        )

    return {"id": item["id"], "words": placed, "warnings": warnings}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    # WhisperX and its dependencies print to stdout on some paths (model load
    # errors, NLTK downloads). Results travel through a file, so stdout is
    # retargeted at stderr to keep that output out of the caller's stdout.
    sys.stdout = sys.stderr

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    destination = Path(sys.argv[2])

    import torch
    import whisperx

    device = request.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    language = request.get("language") or "en"

    print(f"Loading wav2vec2 alignment model ({language}, {device})...", file=sys.stderr)
    model, metadata = whisperx.load_align_model(language_code=language, device=device)
    audio = torch.from_numpy(whisperx.load_audio(request["audio"]))

    items = []
    for position, item in enumerate(request["items"], 1):
        print(f"Aligning region {position}/{len(request['items'])}...", file=sys.stderr)
        items.append(align_item(item, model, metadata, audio, device, whisperx))

    destination.write_text(json.dumps({"items": items}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
