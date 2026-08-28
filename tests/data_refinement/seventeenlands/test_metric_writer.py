from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult
from src.data_refinement.seventeenlands.metric_writer import write_metric_results


class _FakeMetric:
    def __init__(self, name: str, results: dict[UUID, MetricResult]) -> None:
        self.name = name
        self._results = results

    def finalize(self) -> dict[UUID, MetricResult]:
        return self._results

    def save_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


class TestWriteMetricResults:
    def test_writes_one_row_per_metric_result(self, tmp_path: Path) -> None:
        uuid = uuid4()
        result = MetricResult(
            nocab_uuid=uuid,
            metric_name="win_rate",
            value=0.55,
            sample_size=100,
            expansion="MSH",
            format="PremierDraft",
        )
        metric = _FakeMetric("win_rate", {uuid: result})
        output_path = tmp_path / "out.parquet"

        write_metric_results([metric], output_path)

        df = pd.read_parquet(output_path)
        assert len(df) == 1
        assert df.iloc[0]["nocab_uuid"] == str(uuid)
        assert df.iloc[0]["metric_name"] == "win_rate"
        assert df.iloc[0]["value"] == 0.55
        assert df.iloc[0]["sample_size"] == 100
        assert df.iloc[0]["expansion"] == "MSH"
        assert df.iloc[0]["format"] == "PremierDraft"

    def test_merges_rows_from_multiple_metrics(self, tmp_path: Path) -> None:
        uuid_a, uuid_b = uuid4(), uuid4()
        metric_a = _FakeMetric(
            "win_rate",
            {
                uuid_a: MetricResult(
                    nocab_uuid=uuid_a,
                    metric_name="win_rate",
                    value=0.5,
                    sample_size=10,
                    expansion="MSH",
                    format="PremierDraft",
                )
            },
        )
        metric_b = _FakeMetric(
            "inclusion_rate",
            {
                uuid_b: MetricResult(
                    nocab_uuid=uuid_b,
                    metric_name="inclusion_rate",
                    value=0.2,
                    sample_size=10,
                    expansion="MSH",
                    format="PremierDraft",
                )
            },
        )
        output_path = tmp_path / "out.parquet"

        write_metric_results([metric_a, metric_b], output_path)

        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert set(df["metric_name"]) == {"win_rate", "inclusion_rate"}

    def test_empty_metrics_writes_correctly_typed_empty_parquet(
        self, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "out.parquet"

        write_metric_results([], output_path)

        df = pd.read_parquet(output_path)
        assert len(df) == 0
        assert str(df["value"].dtype) == "float64"
        assert str(df["sample_size"].dtype) == "int64"
        assert str(df["nocab_uuid"].dtype) == "string"

    def test_creates_output_parent_directory(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "dir" / "out.parquet"

        write_metric_results([], output_path)

        assert output_path.exists()
