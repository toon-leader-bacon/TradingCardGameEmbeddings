from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (
    PackToPickChoiceSetMetric,
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


def _row(pick: str, owlbear_count: int, morningstar_count: int) -> dict:
    return {
        "draft_id": "draft1",
        "pack_number": 0,
        "pick_number": 2,
        "pick": pick,
        "pack_card_Owlbear": owlbear_count,
        "pack_card_Goblin Morningstar": morningstar_count,
    }


class TestAccumulate:
    def test_writes_one_row_with_pack_options_and_pick(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        owlbear_uuid = str(binder.get_by_name(GameId.MTG, "Owlbear")[0].nocab_uuid)
        morningstar_uuid = str(
            binder.get_by_name(GameId.MTG, "Goblin Morningstar")[0].nocab_uuid
        )
        metric = PackToPickChoiceSetMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 1))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        rows = table.to_pylist()
        assert len(rows) == 1
        row = rows[0]
        assert row["draft_id"] == "draft1"
        assert row["pack_number"] == 0
        assert row["pick_number"] == 2
        assert set(row["pack_option_uuids"]) == {owlbear_uuid, morningstar_uuid}
        assert row["pick_uuid"] == owlbear_uuid

    def test_unmatched_pick_is_written_as_a_null_column(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = PackToPickChoiceSetMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Some Unmatchable Name", 1, 0))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.to_pylist()[0]["pick_uuid"] is None

    def test_writes_one_row_per_accumulate_call(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = PackToPickChoiceSetMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 1))
        metric.accumulate(_row("Goblin Morningstar", 1, 1))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.num_rows == 2


class TestFinalize:
    def test_is_idempotent(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = PackToPickChoiceSetMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        metric.accumulate(_row("Owlbear", 1, 0))

        first = metric.finalize()
        second = metric.finalize()

        assert first == second == tmp_path / "out.parquet"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = PackToPickChoiceSetMetric(
            binder, _HEADER, GameId.MTG, output_path=nested_path
        )

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()


def test_default_output_path() -> None:
    assert PackToPickChoiceSetMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/seventeenlands/draft_data/pack_to_pick_choice_set.parquet"
    )
