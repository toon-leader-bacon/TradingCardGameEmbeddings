from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (
    PoolConditionedPickMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
    "draft_id",
    "pack_number",
    "pick_number",
    "pick",
    "pack_card_Owlbear",
    "pack_card_Goblin Morningstar",
    "pool_Lightning Bolt",
]


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


def _row(
    pick: str, owlbear_count: int, morningstar_count: int, bolt_pool_count: int
) -> dict:
    return {
        "draft_id": "draft1",
        "pack_number": 1,
        "pick_number": 3,
        "pick": pick,
        "pack_card_Owlbear": owlbear_count,
        "pack_card_Goblin Morningstar": morningstar_count,
        "pool_Lightning Bolt": bolt_pool_count,
    }


class TestAccumulate:
    def test_writes_pool_and_pack_options_and_pick(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar", "Lightning Bolt"])
        owlbear_uuid = str(binder.get_by_name(GameId.MTG, "Owlbear")[0].nocab_uuid)
        bolt_uuid = str(binder.get_by_name(GameId.MTG, "Lightning Bolt")[0].nocab_uuid)
        metric = PoolConditionedPickMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 1, 2))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        row = table.to_pylist()[0]
        assert row["draft_id"] == "draft1"
        assert row["pack_number"] == 1
        assert row["pick_number"] == 3
        assert row["pool_uuids"] == [bolt_uuid]
        assert row["pick_uuid"] == owlbear_uuid

    def test_pool_absent_when_count_is_zero(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar", "Lightning Bolt"])
        metric = PoolConditionedPickMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0, 0))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.to_pylist()[0]["pool_uuids"] == []

    def test_unmatched_pick_is_written_as_a_null_column(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar", "Lightning Bolt"])
        metric = PoolConditionedPickMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Some Unmatchable Name", 1, 0, 0))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.to_pylist()[0]["pick_uuid"] is None


class TestFinalize:
    def test_is_idempotent(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = PoolConditionedPickMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        metric.accumulate(_row("Owlbear", 1, 0, 0))

        first = metric.finalize()
        second = metric.finalize()

        assert first == second == tmp_path / "out.parquet"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = PoolConditionedPickMetric(
            binder, _HEADER, GameId.MTG, output_path=nested_path
        )

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()


def test_default_output_path() -> None:
    assert PoolConditionedPickMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/seventeenlands/draft_data/pool_conditioned_pick.parquet"
    )
