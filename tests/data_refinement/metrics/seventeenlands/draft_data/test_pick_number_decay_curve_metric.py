from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (
    PickNumberDecayCurveMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = ["draft_id", "pack_number", "pick_number", "pick", "pack_card_Owlbear"]


def _make_card(name: str) -> GenericCard:
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


def _binder_with_cards(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_make_card(name))
    return binder


def _row(pick: str, owlbear_count: int, pick_number: int) -> dict:
    return {
        "draft_id": "draft1",
        "pack_number": 0,
        "pick_number": pick_number,
        "pick": pick,
        "pack_card_Owlbear": owlbear_count,
    }


def _row_dict(table_rows: list[dict], card_uuid: str) -> dict:
    return next(row for row in table_rows if row["nocab_uuid"] == card_uuid)


class TestFinalize:
    def test_one_row_per_card_with_pick_number_indexed_vectors(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = PickNumberDecayCurveMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        owlbear_uuid = str(binder.get_by_name(GameId.MTG, "Owlbear")[0].nocab_uuid)

        metric.accumulate(_row("Owlbear", 1, pick_number=0))
        metric.accumulate(_row("Some Other Pick", 1, pick_number=1))
        metric.accumulate(_row("Owlbear", 1, pick_number=2))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        rows = table.to_pylist()
        assert len(rows) == 1
        row = _row_dict(rows, owlbear_uuid)
        assert row["sample_count_by_pick_number"] == [1, 1, 1]
        assert row["take_rate_by_pick_number"] == [1.0, 0.0, 1.0]

    def test_bucket_count_is_derived_from_max_pick_number_seen(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = PickNumberDecayCurveMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, pick_number=0))
        metric.accumulate(_row("Owlbear", 1, pick_number=4))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        row = table.to_pylist()[0]
        assert len(row["sample_count_by_pick_number"]) == 5
        # bucket 4 was seen and picked; buckets 1-3 were never seen at
        # all for this card, so they're zero-sample/null-rate, not
        # simply absent.
        assert row["sample_count_by_pick_number"] == [1, 0, 0, 0, 1]
        assert row["take_rate_by_pick_number"][1] is None
        assert row["take_rate_by_pick_number"][4] == 1.0

    def test_no_rows_seen_writes_empty_file(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = PickNumberDecayCurveMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        output_path = metric.finalize()

        table = pq.read_table(output_path)
        assert table.num_rows == 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = PickNumberDecayCurveMetric(
            binder, _HEADER, GameId.MTG, output_path=nested_path
        )
        metric.accumulate(_row("Owlbear", 1, pick_number=0))

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()


def test_default_output_path() -> None:
    assert PickNumberDecayCurveMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/seventeenlands/draft_data/pick_number_decay_curve.parquet"
    )
