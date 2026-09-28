import csv
from pathlib import Path

from src.training.recording.reports import CheckpointRecord, DojoStatus, RoundReport
from src.training.recording.run_listener import CsvRunListener


def _report(
    phase: str = "phase_a",
    round_index: int = 0,
    step: int = 5,
    quarantined: frozenset[str] = frozenset(),
) -> RoundReport:
    return RoundReport(
        phase=phase,
        round_index=round_index,
        step=step,
        per_dojo_test_loss={"color_mask": 0.5, "faction_mask": 1.25},
        statuses={
            "color_mask": DojoStatus.ACTIVE,
            "faction_mask": DojoStatus.SATURATED,
        },
        quarantined=quarantined,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


class TestOnRoundEnd:
    def test_writes_one_row_per_dojo_with_a_header(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        listener.on_round_end(_report())

        rows = _rows(tmp_path / "rounds.csv")
        assert len(rows) == 2
        assert {row["dojo"] for row in rows} == {"color_mask", "faction_mask"}
        color_row = next(row for row in rows if row["dojo"] == "color_mask")
        assert color_row["phase"] == "phase_a"
        assert color_row["round_index"] == "0"
        assert color_row["step"] == "5"
        assert color_row["test_loss"] == "0.5"
        assert color_row["status"] == "active"
        assert color_row["quarantined"] == "False"

    def test_quarantined_dojo_is_flagged(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        listener.on_round_end(_report(quarantined=frozenset({"faction_mask"})))

        rows = _rows(tmp_path / "rounds.csv")
        faction_row = next(row for row in rows if row["dojo"] == "faction_mask")
        color_row = next(row for row in rows if row["dojo"] == "color_mask")
        assert faction_row["quarantined"] == "True"
        assert color_row["quarantined"] == "False"

    def test_a_dojo_missing_from_statuses_gets_an_empty_status(
        self, tmp_path: Path
    ) -> None:
        report = RoundReport(
            phase="phase_a",
            round_index=0,
            step=0,
            per_dojo_test_loss={"held_out": 2.0},
            statuses={},
            quarantined=frozenset(),
        )
        listener = CsvRunListener(tmp_path / "rounds.csv")
        listener.on_round_end(report)

        rows = _rows(tmp_path / "rounds.csv")
        assert rows[0]["status"] == ""

    def test_repeated_calls_append_rather_than_overwrite(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        listener.on_round_end(_report(round_index=0))
        listener.on_round_end(_report(round_index=1))

        rows = _rows(tmp_path / "rounds.csv")
        assert len(rows) == 4
        assert {row["round_index"] for row in rows} == {"0", "1"}

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "runs" / "rounds.csv"
        CsvRunListener(path).on_round_end(_report())
        assert path.exists()


class TestOnCheckpoint:
    def test_writes_one_row_with_a_header(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        record = CheckpointRecord(
            path=tmp_path / "phase_a_round0000" / "state.pt",
            encoder_path=tmp_path / "phase_a_round0000" / "encoder.pt",
            report=_report(),
        )
        listener.on_checkpoint(record)

        rows = _rows(tmp_path / "rounds_checkpoints.csv")
        assert len(rows) == 1
        assert rows[0]["phase"] == "phase_a"
        assert rows[0]["round_index"] == "0"
        assert rows[0]["step"] == "5"
        assert rows[0]["path"] == str(record.path)
        assert rows[0]["encoder_path"] == str(record.encoder_path)

    def test_checkpoints_path_can_be_set_explicitly(self, tmp_path: Path) -> None:
        checkpoints_path = tmp_path / "ckpts.csv"
        listener = CsvRunListener(tmp_path / "rounds.csv", checkpoints_path)
        record = CheckpointRecord(
            path=tmp_path / "state.pt",
            encoder_path=tmp_path / "encoder.pt",
            report=_report(),
        )
        listener.on_checkpoint(record)
        assert checkpoints_path.exists()
        assert not (tmp_path / "rounds_checkpoints.csv").exists()

    def test_repeated_calls_append_rather_than_overwrite(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        first = CheckpointRecord(
            path=tmp_path / "a" / "state.pt",
            encoder_path=tmp_path / "a" / "encoder.pt",
            report=_report(round_index=0),
        )
        second = CheckpointRecord(
            path=tmp_path / "b" / "state.pt",
            encoder_path=tmp_path / "b" / "encoder.pt",
            report=_report(round_index=1),
        )
        listener.on_checkpoint(first)
        listener.on_checkpoint(second)

        rows = _rows(tmp_path / "rounds_checkpoints.csv")
        assert [row["round_index"] for row in rows] == ["0", "1"]

    def test_rounds_and_checkpoints_files_are_independent(self, tmp_path: Path) -> None:
        listener = CsvRunListener(tmp_path / "rounds.csv")
        listener.on_round_end(_report())
        record = CheckpointRecord(
            path=tmp_path / "state.pt",
            encoder_path=tmp_path / "encoder.pt",
            report=_report(),
        )
        listener.on_checkpoint(record)

        assert len(_rows(tmp_path / "rounds.csv")) == 2
        assert len(_rows(tmp_path / "rounds_checkpoints.csv")) == 1
