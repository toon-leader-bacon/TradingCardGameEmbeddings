import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.draft_data_metrics.draft_metric_scanner import (
    DraftMetricScanner,
    DraftMetricScanResult,
)
from src.data_refinement.seventeenlands.draft_data_metrics.metrics.average_pick_number import (
    AveragePickNumberMetric,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class _FakeDraftMetric:
    """Test double isolating DraftMetricScanner's driver mechanics
    (chunking, checkpointing) from any particular metric's own
    accumulate() logic.
    """

    name = "fake_draft_metric"

    def __init__(self) -> None:
        self.rows_seen = 0
        self.chunks_seen = 0

    def accumulate(self, chunk: pd.DataFrame, resolved_picks: pd.Series) -> None:
        self.rows_seen += len(chunk)
        self.chunks_seen += 1

    def finalize(self) -> dict[UUID, MetricResult]:
        return {
            uuid4(): MetricResult(
                nocab_uuid=uuid4(),
                metric_name=self.name,
                value=1.0,
                sample_size=self.rows_seen,
                expansion="MSH",
                format="PremierDraft",
            )
        }

    def save_state(self) -> dict:
        return {"rows_seen": self.rows_seen, "chunks_seen": self.chunks_seen}

    def load_state(self, state: dict) -> None:
        self.rows_seen = state["rows_seen"]
        self.chunks_seen = state["chunks_seen"]


class _RaisingAfterNChunksMetric(_FakeDraftMetric):
    """A metric whose accumulate() raises after N chunks — used to
    simulate a mid-scan crash after some checkpoints have already been
    written.
    """

    def __init__(self, raise_after_chunks: int) -> None:
        super().__init__()
        self._raise_after_chunks = raise_after_chunks

    def accumulate(self, chunk: pd.DataFrame, resolved_picks: pd.Series) -> None:
        if self.chunks_seen >= self._raise_after_chunks:
            raise RuntimeError("simulated crash")
        super().accumulate(chunk, resolved_picks)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _write_csv(path: Path, num_rows: int, pick: str = "Bolt") -> None:
    header = "expansion,pack_number,pick_number,pick,pick_maindeck_rate,pick_sideboard_in_rate\n"
    lines = [header]
    for i in range(num_rows):
        pick_number = i % 15
        lines.append(f"MSH,0,{pick_number},{pick},0.9,0.1\n")
    path.write_text("".join(lines))


class TestScan:
    def test_first_scan_no_checkpoint(self, tmp_path: Path) -> None:
        registry = CardBinder()
        registry.add(_card("Bolt"))
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=10)
        metric = _FakeDraftMetric()

        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[metric],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=tmp_path / "checkpoint.json",
            source_game=GameId.MTG,
        )
        result = scanner.scan()

        assert isinstance(result, DraftMetricScanResult)
        assert metric.rows_seen == 10

    def test_successful_completion_deletes_checkpoint(self, tmp_path: Path) -> None:
        registry = CardBinder()
        registry.add(_card("Bolt"))
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=5)
        checkpoint_path = tmp_path / "checkpoint.json"

        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[_FakeDraftMetric()],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=checkpoint_path,
            source_game=GameId.MTG,
        )
        scanner.scan()

        assert not checkpoint_path.exists()

    def test_unresolved_pick_names_surface_in_result(self, tmp_path: Path) -> None:
        registry = CardBinder()  # no cards registered — "Bolt" unresolved
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=3)

        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[_FakeDraftMetric()],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=tmp_path / "checkpoint.json",
            source_game=GameId.MTG,
        )
        result = scanner.scan()

        assert result.unresolved_pick_names == ["Bolt"]

    def test_empty_results_still_write_a_correctly_typed_parquet_file(
        self, tmp_path: Path
    ) -> None:
        # Regression test mirroring game_data_metrics/test_metric_scanner.py's
        # own dtype regression test: with no metrics at all,
        # pd.DataFrame([], columns=[...]) has nothing to infer dtypes
        # from and defaults every column to `object`. _write_results()
        # must cast explicitly so the schema is identical whether or
        # not any rows were produced.
        registry = CardBinder()
        registry.add(_card("Bolt"))
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=3)

        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=tmp_path / "checkpoint.json",
            source_game=GameId.MTG,
        )
        result = scanner.scan()

        df = pd.read_parquet(result.output_path)
        assert len(df) == 0
        assert str(df["value"].dtype) == "float64"
        assert str(df["sample_size"].dtype) == "int64"

    def test_output_parquet_round_trips(self, tmp_path: Path) -> None:
        registry = CardBinder()
        bolt = registry.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=4)

        metric = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[metric],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=tmp_path / "checkpoint.json",
            source_game=GameId.MTG,
        )
        result = scanner.scan()

        df = pd.read_parquet(result.output_path)
        assert len(df) == 1
        assert df.iloc[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert df.iloc[0]["metric_name"] == "average_pick_number"
        assert df.iloc[0]["sample_size"] == 4
        # pick_number values for 4 rows: 0, 1, 2, 3 -> mean 1.5
        assert df.iloc[0]["value"] == 1.5

    def test_resumes_from_checkpoint_without_reprocessing_rows(
        self, tmp_path: Path
    ) -> None:
        registry = CardBinder()
        registry.add(_card("Bolt"))
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=25)
        checkpoint_path = tmp_path / "checkpoint.json"
        output_path = tmp_path / "out.parquet"

        # First scan: crashes partway through, but not before at least
        # one checkpoint has been written (chunksize is 100_000 rows,
        # larger than our whole fixture — simulate the crash directly
        # by writing a checkpoint reflecting partial progress, matching
        # what a real crash mid-stream would leave behind, then confirm
        # a fresh scan resumes from exactly that point rather than
        # reprocessing from row 0.
        partial_metric = AveragePickNumberMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        partial_chunk = pd.read_csv(csv_path, nrows=15)
        bolt_uuid = registry.get_by_name(GameId.MTG, "Bolt").nocab_uuid
        resolved_picks = pd.Series([bolt_uuid] * len(partial_chunk))
        partial_metric.accumulate(partial_chunk, resolved_picks)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "rows_processed": 15,
                    "metric_states": {partial_metric.name: partial_metric.save_state()},
                    "extra": {},
                }
            )
        )

        # Second scan: fresh metric instance, resumes from the checkpoint.
        resumed_metric = AveragePickNumberMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[resumed_metric],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            source_game=GameId.MTG,
        )
        scanner.scan()

        assert not checkpoint_path.exists()

        # Compare against a from-scratch single-pass scan over the
        # whole file — the resumed scan must produce identical totals,
        # proving no rows were skipped or double-counted.
        expected_metric = AveragePickNumberMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        full_df = pd.read_csv(csv_path)
        full_resolved_picks = pd.Series([bolt_uuid] * len(full_df))
        expected_metric.accumulate(full_df, full_resolved_picks)
        expected = expected_metric.finalize()[bolt_uuid]

        result_df = pd.read_parquet(output_path)
        assert result_df.iloc[0]["sample_size"] == expected.sample_size
        assert result_df.iloc[0]["value"] == pytest.approx(expected.value)

    def test_checkpoint_written_periodically_reflects_cumulative_rows(
        self, tmp_path: Path
    ) -> None:
        # DraftMetricScanner's internal chunk size is 100_000 rows, so
        # this fixture needs to exceed that to exercise a second chunk.
        registry = CardBinder()
        registry.add(_card("Bolt"))
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, num_rows=100_001)
        checkpoint_path = tmp_path / "checkpoint.json"
        metric = _RaisingAfterNChunksMetric(raise_after_chunks=1)

        scanner = DraftMetricScanner(
            raw_csv_path=csv_path,
            card_binder=registry,
            metrics=[metric],
            output_path=tmp_path / "out.parquet",
            checkpoint_path=checkpoint_path,
            source_game=GameId.MTG,
            checkpoint_every_n_chunks=1,
        )

        with pytest.raises(RuntimeError):
            scanner.scan()

        assert checkpoint_path.exists()
        state = json.loads(checkpoint_path.read_text())
        assert state["rows_processed"] == 100_000  # only chunk 1 succeeded
        assert state["metric_states"][metric.name]["rows_seen"] == 100_000


# Checkpoint restore/write behavior now lives on the shared
# MetricCheckpoint (see metric_checkpoint.py) — its own dedicated tests
# are in tests/data_refinement/seventeenlands/test_metric_checkpoint.py,
# not duplicated here.
