"""Attribute aligned words to diarization speakers and group them into segments."""

from __future__ import annotations

from transcribe_pipeline.models import Segment, Turn, Word

# A speaker counts as active for a word only if their turn covers a meaningful
# share of it. Without this, a turn boundary landing a few milliseconds inside a
# word would make every such word look like crosstalk.
MIN_ACTIVE_RATIO = 0.15
MIN_ACTIVE_SECONDS = 0.02

# When several speakers are active, one is only credited if it clearly accounts
# for the word; otherwise attribution stays null rather than guessing.
DOMINANCE_SHARE = 0.60
DOMINANCE_RATIO = 2.0

# Diarization leaves small gaps between turns, so a word falling just outside
# every turn is snapped to the nearest one within this tolerance.
GAP_TOLERANCE_SECONDS = 0.75

# Segment grouping.
MAX_SEGMENT_GAP_SECONDS = 1.0
MAX_SEGMENT_SECONDS = 30.0


def overlap_seconds(start: float, end: float, turn: Turn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def active_speakers(start: float, end: float, turns: tuple[Turn, ...]) -> dict[str, float]:
    """Return the seconds each speaker is active within the word's span."""
    duration = max(end - start, 1e-6)
    threshold = max(MIN_ACTIVE_SECONDS, MIN_ACTIVE_RATIO * duration)

    shares: dict[str, float] = {}
    for turn in turns:
        if turn.start >= end:
            break
        covered = overlap_seconds(start, end, turn)
        if covered > 0:
            shares[turn.speaker] = shares.get(turn.speaker, 0.0) + covered

    return {speaker: value for speaker, value in shares.items() if value >= threshold}


def nearest_speaker(start: float, end: float, turns: tuple[Turn, ...]) -> str | None:
    best: str | None = None
    best_distance = GAP_TOLERANCE_SECONDS
    for turn in turns:
        if turn.end < start:
            distance = start - turn.end
        elif turn.start > end:
            distance = turn.start - end
        else:
            distance = 0.0
        if distance <= best_distance:
            best_distance = distance
            best = turn.speaker
    return best


def exclusive_speaker(
    start: float, end: float, exclusive_turns: tuple[Turn, ...], candidates: list[str]
) -> str | None:
    """Break a tie using pyannoteAI's exclusive diarization.

    Exclusive diarization reports the single speaker most likely to be
    transcribed at each instant, which is exactly the question left open when
    several speakers overlap a word. The answer is only accepted if it is one of
    the speakers the canonical diarization already found active, so this can
    narrow an attribution but never invent one.
    """
    if not exclusive_turns or not candidates:
        return None

    shares: dict[str, float] = {}
    for turn in exclusive_turns:
        if turn.start >= end:
            break
        covered = overlap_seconds(start, end, turn)
        if covered > 0:
            shares[turn.speaker] = shares.get(turn.speaker, 0.0) + covered

    allowed = {speaker: value for speaker, value in shares.items() if speaker in candidates}
    if not allowed:
        return None
    return max(allowed.items(), key=lambda item: (item[1], item[0]))[0]


def attribute_word(
    word: Word, turns: tuple[Turn, ...], exclusive_turns: tuple[Turn, ...] = ()
) -> Word:
    if not word.aligned:
        return word

    start, end = float(word.start), float(word.end)
    shares = active_speakers(start, end, turns)

    if not shares:
        word.speaker = nearest_speaker(start, end, turns)
        word.speaker_candidates = []
        word.overlap = False
        return word

    ranked = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
    word.speaker_candidates = sorted(shares)

    if len(ranked) == 1:
        word.speaker = ranked[0][0]
        word.speaker_candidates = []
        word.overlap = False
        return word

    # Overlap is recorded whatever happens next: resolving who to credit never
    # erases the fact that more than one person was speaking.
    word.overlap = True
    total = sum(shares.values())
    leader, leader_share = ranked[0]
    runner_up = ranked[1][1]
    dominates = leader_share >= DOMINANCE_SHARE * total and leader_share >= DOMINANCE_RATIO * runner_up

    if dominates:
        word.speaker = leader
    else:
        word.speaker = exclusive_speaker(start, end, exclusive_turns, word.speaker_candidates)
    return word


def attribute_words(
    words: list[Word], turns: tuple[Turn, ...], exclusive_turns: tuple[Turn, ...] = ()
) -> list[Word]:
    return [attribute_word(word, turns, exclusive_turns) for word in words]


def keep_owned(words: list[Word], own_start: float, own_end: float) -> list[Word]:
    """Drop aligned words the neighbouring chunk owns, so seams do not duplicate text.

    Only aligned words can be placed on either side of a seam; unaligned words are
    kept so their text is never silently lost.
    """
    kept: list[Word] = []
    for word in words:
        if not word.aligned:
            kept.append(word)
            continue
        midpoint = (float(word.start) + float(word.end)) / 2
        if own_start <= midpoint < own_end:
            kept.append(word)
    return kept


def _key(word: Word) -> tuple:
    return (word.aligned, word.speaker, word.overlap, tuple(word.speaker_candidates))


def group_segments(words: list[Word]) -> list[Segment]:
    """Collapse consecutive words that share an attribution into readable segments."""
    segments: list[Segment] = []
    run: list[Word] = []

    def flush() -> None:
        if not run:
            return
        timed = [word for word in run if word.aligned]
        text = " ".join(word.text for word in run).strip()
        segments.append(
            Segment(
                start=min(float(word.start) for word in timed) if timed else None,
                end=max(float(word.end) for word in timed) if timed else None,
                speaker=run[0].speaker,
                speaker_candidates=list(run[0].speaker_candidates),
                overlap=run[0].overlap,
                raw_text=text,
                clean_text=text,
                aligned=bool(timed),
            )
        )
        run.clear()

    for word in words:
        if run:
            previous = run[-1]
            gap_too_wide = (
                word.aligned
                and previous.aligned
                and float(word.start) - float(previous.end) > MAX_SEGMENT_GAP_SECONDS
            )
            too_long = (
                word.aligned
                and run[0].aligned
                and float(word.end) - float(run[0].start) > MAX_SEGMENT_SECONDS
            )
            if _key(word) != _key(previous) or gap_too_wide or too_long:
                flush()
        run.append(word)

    flush()
    return segments
