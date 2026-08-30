import warnings
from pathlib import Path

import pandas as pd

from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner


class _FakeMetric:
    """Test double isolating CsvScanner's driver mechanics from any
    particular metric's own accumulate()/finalize() logic.
    """

    def __init__(self, name: str, output_path: Path) -> None:
        self.name = name
        self._output_path = output_path
        self.chunks_seen = 0
        self.finalize_called = False

    def accumulate(self, chunk: pd.DataFrame) -> None:
        self.chunks_seen += 1

    def finalize(self) -> Path:
        self.finalize_called = True
        self._output_path.write_text("fake output")
        return self._output_path


class _MissingOutputMetric:
    """A metric whose finalize() reports a path it never actually wrote."""

    name = "missing_output"

    def accumulate(self, chunk: pd.DataFrame) -> None:
        pass

    def finalize(self) -> Path:
        return Path("/nonexistent/path/that/was/never/written.jsonl")


def _write_csv(path: Path, num_rows: int) -> None:
    lines = ["a,b\n"]
    for i in range(num_rows):
        lines.append(f"{i},{i}\n")
    path.write_text("".join(lines))


class TestScan:
    def test_drives_every_metric_over_every_chunk(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, num_rows=250_000)  # spans 3 chunks at chunk_size=100_000
        metric_a = _FakeMetric("a", tmp_path / "a.out")
        metric_b = _FakeMetric("b", tmp_path / "b.out")

        scanner = CsvScanner(csv_path, chunk_size=100_000, metrics=[metric_a, metric_b])
        scanner.scan()

        assert metric_a.chunks_seen == 3
        assert metric_b.chunks_seen == 3  # same single pass drove both metrics

    def test_finalizes_every_metric_exactly_once(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, num_rows=10)
        metric = _FakeMetric("a", tmp_path / "a.out")

        scanner = CsvScanner(csv_path, chunk_size=100_000, metrics=[metric])
        scanner.scan()

        assert metric.finalize_called is True

    def test_returns_paths_in_metrics_order(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, num_rows=10)
        metric_a = _FakeMetric("a", tmp_path / "a.out")
        metric_b = _FakeMetric("b", tmp_path / "b.out")

        scanner = CsvScanner(csv_path, chunk_size=100_000, metrics=[metric_a, metric_b])
        result = scanner.scan()

        assert result == [tmp_path / "a.out", tmp_path / "b.out"]

    def test_warns_but_does_not_raise_when_a_metric_reports_a_missing_path(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, num_rows=10)
        scanner = CsvScanner(csv_path, chunk_size=100_000, metrics=[_MissingOutputMetric()])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = scanner.scan()

        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)
        assert result == [Path("/nonexistent/path/that/was/never/written.jsonl")]
