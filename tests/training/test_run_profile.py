import csv
from pathlib import Path

import pytest
import torch

from src.training.recording.csv_table import CsvTable
from src.training.recording.run_profile import (
    NO_RUN_PROFILE,
    BuildTiming,
    CsvRunProfile,
    EvaluationTiming,
    ReserveTiming,
    StepTiming,
    WorkSpan,
    card_count,
    stage_field_names,
)
from src.utils.stage_clock import ProfileStage, StageTotals
from tests.dojos.mods.test_card_field_mods import _gwent_card


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stages() -> StageTotals:
    return StageTotals.of(
        {stage: 0.5 if stage is ProfileStage.MODS else 0.0 for stage in ProfileStage},
        7,
    )


class TestCsvTable:
    def test_writes_rows_under_its_header(self, tmp_path: Path) -> None:
        table = CsvTable(tmp_path / "t.csv", ["a", "b"])
        table.write({"a": 1, "b": 0.25})
        table.close()
        assert _rows(tmp_path / "t.csv") == [{"a": "1", "b": "0.250000"}]

    def test_refuses_an_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "t.csv").write_text("x")
        with pytest.raises(FileExistsError):
            CsvTable(tmp_path / "t.csv", ["a"])

    def test_a_closed_table_drops_rows(self, tmp_path: Path) -> None:
        table = CsvTable(tmp_path / "t.csv", ["a"])
        table.close()
        table.write({"a": 1})
        table.close()
        assert _rows(tmp_path / "t.csv") == []


class TestCsvRunProfile:
    def test_writes_one_row_per_record(self, tmp_path: Path) -> None:
        profile = CsvRunProfile(tmp_path / "profile")
        profile.record_step(
            StepTiming("train", 0, 3, "a", 12, True, 0.1, 0.2, 0.5, _stages())
        )
        profile.record_build(BuildTiming("a", 4, 12, 1, 0.3, "worker", _stages()))
        profile.record_reserve(ReserveTiming("a", 12, 0.05, _stages()))
        profile.record_evaluation(
            EvaluationTiming("train", 0, "a", 4, 12, 1.5, False, _stages())
        )
        profile.close()

        (step,) = _rows(tmp_path / "profile" / "steps.csv")
        assert step["dojo"] == "a" and step["cards"] == "12"
        assert step["phase"] == "train"
        assert float(step["remainder_s"]) == pytest.approx(0.2)
        assert float(step["mods_s"]) == 0.5 and step["live_texts"] == "7"
        assert _rows(tmp_path / "profile" / "builds.csv")[0]["thread"] == "worker"
        assert _rows(tmp_path / "profile" / "reserves.csv")[0]["cards"] == "12"
        assert _rows(tmp_path / "profile" / "evaluation.csv")[0]["completed"] == "0"
        marks = [row["mark"] for row in _rows(tmp_path / "profile" / "marks.csv")]
        assert marks == ["profile opened", "profile closed"]

    def test_refuses_a_directory_with_its_csvs(self, tmp_path: Path) -> None:
        CsvRunProfile(tmp_path).close()
        with pytest.raises(FileExistsError):
            CsvRunProfile(tmp_path)

    def test_counts_time_from_opened_at(self, tmp_path: Path) -> None:
        profile = CsvRunProfile(tmp_path, opened_at=0.0)
        assert profile.seconds_since_open() > 1.0  # monotonic is past 0
        profile.close()

    def test_a_write_after_close_is_dropped_quietly(self, tmp_path: Path) -> None:
        profile = CsvRunProfile(tmp_path)
        profile.close()
        profile.mark("late")  # no exception
        assert len(_rows(tmp_path / "marks.csv")) == 2


def test_the_null_profile_drops_everything() -> None:
    NO_RUN_PROFILE.mark("x")
    NO_RUN_PROFILE.record_reserve(ReserveTiming("a", 1, 0.1, StageTotals.zero()))
    NO_RUN_PROFILE.close()


def test_card_count_counts_cards_and_tolerates_other_inputs() -> None:
    card = _gwent_card()
    assert card_count([[card, card], [card]]) == 3
    assert card_count(torch.zeros(4, 3)) == 0


def test_stage_columns_end_with_live_texts() -> None:
    names = stage_field_names()
    assert names[-1] == "live_texts"
    assert len(names) == len(ProfileStage) + 1


def test_card_count_survives_inputs_it_cannot_walk() -> None:
    assert card_count("a string recurses forever") == 0  # type: ignore[arg-type]


def test_a_work_span_measures_time_and_stages() -> None:
    span = WorkSpan()
    assert span.seconds() >= 0
    assert span.stages().live_texts == 0
