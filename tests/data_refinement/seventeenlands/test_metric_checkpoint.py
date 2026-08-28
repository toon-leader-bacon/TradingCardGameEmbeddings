import json
from pathlib import Path
from uuid import UUID

from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.metric_result import MetricResult


class _FakeMetric:
    name = "fake_metric"

    def __init__(self) -> None:
        self.rows_seen = 0

    def finalize(self) -> dict[UUID, MetricResult]:
        return {}

    def save_state(self) -> dict:
        return {"rows_seen": self.rows_seen}

    def load_state(self, state: dict) -> None:
        self.rows_seen = state["rows_seen"]


class TestRestore:
    def test_no_checkpoint_returns_zero_and_empty_extra(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "does_not_exist.json")

        rows_processed, extra = checkpoint.restore([_FakeMetric()])

        assert rows_processed == 0
        assert extra == {}

    def test_restores_metric_state_and_rows_processed(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "rows_processed": 42,
                    "metric_states": {"fake_metric": {"rows_seen": 42}},
                    "extra": {},
                }
            )
        )
        metric = _FakeMetric()
        checkpoint = MetricCheckpoint(checkpoint_path)

        rows_processed, extra = checkpoint.restore([metric])

        assert rows_processed == 42
        assert metric.rows_seen == 42
        assert extra == {}

    def test_restores_extra_payload(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "rows_processed": 10,
                    "metric_states": {"fake_metric": {"rows_seen": 10}},
                    "extra": {"unresolved_arena_ids": ["1", "2"]},
                }
            )
        )
        checkpoint = MetricCheckpoint(checkpoint_path)

        _, extra = checkpoint.restore([_FakeMetric()])

        assert extra == {"unresolved_arena_ids": ["1", "2"]}


class TestWriteThenRestore:
    def test_round_trips_rows_processed_and_metric_state(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")
        metric = _FakeMetric()
        metric.rows_seen = 99

        checkpoint.write(99, [metric])

        restored_metric = _FakeMetric()
        rows_processed, extra = checkpoint.restore([restored_metric])
        assert rows_processed == 99
        assert restored_metric.rows_seen == 99
        assert extra == {}

    def test_round_trips_extra_payload(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        checkpoint.write(5, [_FakeMetric()], extra={"unresolved_arena_ids": ["9"]})

        _, extra = checkpoint.restore([_FakeMetric()])
        assert extra == {"unresolved_arena_ids": ["9"]}

    def test_write_overwrites_previous_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        checkpoint.write(1, [_FakeMetric()])
        checkpoint.write(2, [_FakeMetric()])

        rows_processed, _ = checkpoint.restore([_FakeMetric()])
        assert rows_processed == 2

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "nested" / "dir" / "checkpoint.json")

        checkpoint.write(1, [_FakeMetric()])

        assert (tmp_path / "nested" / "dir" / "checkpoint.json").exists()


class TestDelete:
    def test_deletes_existing_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")
        checkpoint.write(1, [_FakeMetric()])

        checkpoint.delete()

        assert not (tmp_path / "checkpoint.json").exists()

    def test_missing_checkpoint_is_a_noop(self, tmp_path: Path) -> None:
        checkpoint = MetricCheckpoint(tmp_path / "does_not_exist.json")

        checkpoint.delete()  # must not raise
