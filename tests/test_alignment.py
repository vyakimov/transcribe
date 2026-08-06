"""The WhisperX boundary: the worker's windowing logic and the driver's parsing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe_pipeline import alignment
from transcribe_pipeline.alignment import default_command, parse_worker_output

WORKER_PATH = Path(__file__).resolve().parent.parent / "workers" / "align_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("align_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("align_worker", module)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


# --- a stand-in for whisperx.align --------------------------------------------


class FakeWhisperX:
    """Reproduces the parts of WhisperX's contract the worker depends on.

    Words are returned in order with their true times when they fit inside the
    requested window. Words that do not fit are crammed against the window's end,
    which is what CTC forced alignment does when it is handed more text than the
    audio contains.
    """

    def __init__(self, truth: dict[str, tuple[float, float]], *, fail: bool = False):
        self.truth = truth
        self.fail = fail
        self.windows: list[tuple[float, float, int]] = []

    def align(self, segments, model, metadata, audio, device, return_char_alignments=False):
        segment = segments[0]
        words = segment["text"].split()
        self.windows.append((segment["start"], segment["end"], len(words)))

        if self.fail:
            raise RuntimeError("alignment model exploded")

        out = []
        for word in words:
            start, end = self.truth[word]
            if end <= segment["end"]:
                out.append({"word": word, "start": start, "end": end, "score": 0.9})
            else:
                out.append(
                    {
                        "word": word,
                        "start": segment["end"] - 0.01,
                        "end": segment["end"],
                        "score": 0.1,
                    }
                )
        return {"word_segments": out}


def make_speech(count: int, step: float = 0.4, start: float = 0.0):
    """Uniquely named words at a steady rate, plus their ground-truth times."""
    words = [f"w{index:04d}" for index in range(count)]
    truth = {
        word: (start + index * step, start + index * step + step * 0.9)
        for index, word in enumerate(words)
    }
    return words, truth


# --- pair_aligned --------------------------------------------------------------


class TestPairAligned:
    def test_exact_match_pairs_positionally(self):
        paired = worker.pair_aligned(
            ["one", "two"],
            [{"word": "one", "start": 0, "end": 1}, {"word": "two", "start": 1, "end": 2}],
        )

        assert [entry["word"] for entry in paired] == ["one", "two"]

    def test_dropped_words_leave_gaps_rather_than_shifting(self):
        """WhisperX omits words that clean to nothing; the rest must not shift."""
        paired = worker.pair_aligned(
            ["real", "...", "words"],
            [{"word": "real", "start": 0, "end": 1}, {"word": "words", "start": 2, "end": 3}],
        )

        assert paired[0] is not None
        assert paired[1] is None
        assert paired[2] is not None and paired[2]["word"] == "words"

    def test_matching_ignores_punctuation_and_case(self):
        paired = worker.pair_aligned(["Hello,"], [{"word": "hello", "start": 0, "end": 1}])

        assert paired[0] is not None

    def test_repeated_words_are_matched_in_order(self):
        paired = worker.pair_aligned(
            ["no", "no", "no"],
            [
                {"word": "no", "start": 0, "end": 1},
                {"word": "no", "start": 1, "end": 2},
                {"word": "no", "start": 2, "end": 3},
            ],
        )

        assert [entry["start"] for entry in paired] == [0, 1, 2]


# --- windowed alignment --------------------------------------------------------


class TestAlignItem:
    def test_long_region_is_fully_aligned_in_order(self):
        """A 2-minute region needs several windows; no word may be lost or reordered."""
        words, truth = make_speech(300)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 120.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert [entry["word"] for entry in result["words"]] == words
        assert all("start" in entry for entry in result["words"])
        assert result["warnings"] == []

    def test_aligned_times_match_the_ground_truth(self):
        words, truth = make_speech(300)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 120.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        for entry in result["words"]:
            expected_start, expected_end = truth[entry["word"]]
            assert entry["start"] == pytest.approx(expected_start, abs=0.01)
            assert entry["end"] == pytest.approx(expected_end, abs=0.01)

    def test_timestamps_are_monotonic(self):
        words, truth = make_speech(300)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 120.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)
        starts = [entry["start"] for entry in result["words"]]

        assert starts == sorted(starts)

    def test_windows_carry_more_audio_than_text(self):
        """The audio slack is what keeps the crammed tail from dragging words early."""
        words, truth = make_speech(300)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 120.0, "text": " ".join(words)}

        worker.align_item(item, None, None, None, "cpu", fake)

        interior = [window for window in fake.windows if window[1] - window[0] >= worker.WINDOW_SECONDS]
        assert interior, "expected at least one full-width window"
        for start, end, word_count in interior:
            assert word_count * 0.4 < (end - start)

    def test_region_offset_is_respected(self):
        words, truth = make_speech(100, start=600.0)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 600.0, "end": 640.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert result["words"][0]["start"] == pytest.approx(600.0, abs=0.01)

    def test_short_region_aligns_in_a_single_window(self):
        words, truth = make_speech(20)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 8.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert len(fake.windows) == 1
        assert len(result["words"]) == 20

    def test_alignment_failure_keeps_the_text_and_warns(self):
        words, truth = make_speech(50)
        fake = FakeWhisperX(truth, fail=True)
        item = {"id": 0, "start": 0.0, "end": 20.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert [entry["word"] for entry in result["words"]] == words
        assert all("start" not in entry for entry in result["words"])
        assert any("could not be aligned" in warning for warning in result["warnings"])

    def test_more_text_than_audio_still_returns_every_word(self):
        words, truth = make_speech(400)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 30.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert [entry["word"] for entry in result["words"]] == words

    def test_crammed_words_are_reported_rather_than_passed_off_as_aligned(self):
        """Overflowing text gets squeezed against the window end; that must be visible."""
        words, truth = make_speech(400)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 30.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert any("implausibly short" in warning for warning in result["warnings"])

    def test_plausible_alignment_produces_no_cramming_warning(self):
        words, truth = make_speech(300)
        fake = FakeWhisperX(truth)
        item = {"id": 0, "start": 0.0, "end": 120.0, "text": " ".join(words)}

        result = worker.align_item(item, None, None, None, "cpu", fake)

        assert not any("implausibly short" in warning for warning in result["warnings"])

    def test_empty_text_produces_nothing(self):
        item = {"id": 3, "start": 0.0, "end": 30.0, "text": "   "}

        result = worker.align_item(item, None, None, None, "cpu", FakeWhisperX({}))

        assert result == {"id": 3, "words": [], "warnings": []}


# --- driver parsing ------------------------------------------------------------


class TestParseWorkerOutput:
    def test_words_and_warnings_are_collected_per_item(self):
        payload = {
            "items": [
                {
                    "id": 0,
                    "words": [{"word": "hello", "start": 0.1, "end": 0.5, "score": 0.9}],
                    "warnings": ["something odd"],
                },
                {"id": 1, "words": [{"word": "untimed"}], "warnings": []},
            ]
        }

        result = parse_worker_output(payload)

        assert result.words_by_id[0][0].text == "hello"
        assert result.words_by_id[0][0].aligned is True
        assert result.words_by_id[1][0].aligned is False
        assert result.warnings == ["something odd"]

    def test_missing_items_key_is_tolerated(self):
        assert parse_worker_output({}).words_by_id == {}


class TestDefaultCommand:
    def test_an_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_ALIGN_CMD", "/usr/bin/python3 -X dev worker.py")

        assert default_command() == ["/usr/bin/python3", "-X", "dev", "worker.py"]

    def test_a_prepared_environment_is_used_directly(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TRANSCRIBE_ALIGN_CMD", raising=False)
        prepared = tmp_path / "python"
        prepared.write_text("")
        monkeypatch.setattr(alignment, "align_venv_python", lambda: prepared)

        command = default_command()

        assert command[0] == str(prepared)
        assert command[1].endswith("align_worker.py")

    def test_without_a_prepared_environment_uv_builds_one_on_demand(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_ALIGN_CMD", raising=False)
        monkeypatch.setattr(alignment, "align_venv_python", lambda: None)
        monkeypatch.setattr(alignment.shutil, "which", lambda name: "/usr/local/bin/uv")

        command = default_command()

        assert command[:3] == ["uv", "run", "--no-project"]
        assert "whisperx" in command
        assert "3.12" in command

    def test_a_missing_uv_explains_the_alternative(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_ALIGN_CMD", raising=False)
        monkeypatch.setattr(alignment, "align_venv_python", lambda: None)
        monkeypatch.setattr(alignment.shutil, "which", lambda name: None)

        with pytest.raises(alignment.AlignmentError, match="TRANSCRIBE_ALIGN_CMD"):
            default_command()
