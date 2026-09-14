from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


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


_HEADER = [
    "draft_id",
    "pack_number",
    "pick_number",
    "pick",
    "rank",
    "pack_card_Owlbear",
    "pack_card_Goblin Morningstar",
]


def _row(
    pick: str,
    owlbear_count: int,
    morningstar_count: int,
    pack_number: int = 0,
    pick_number: int = 0,
    rank: str = "gold",
) -> dict:
    return {
        "draft_id": "draft1",
        "pack_number": pack_number,
        "pick_number": pick_number,
        "pick": pick,
        "rank": rank,
        "pack_card_Owlbear": owlbear_count,
        "pack_card_Goblin Morningstar": morningstar_count,
    }


def _uuid_for(binder: CardBinder, name: str) -> object:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


class TestCardTakeRateMetric:
    def test_take_rate_tallies_across_pack_number_and_pick_number(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = CardTakeRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 1))
        metric.accumulate(_row("Goblin Morningstar", 1, 1))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        morningstar_uuid = str(_uuid_for(binder, "Goblin Morningstar"))
        assert df.loc[owlbear_uuid, "take_rate"] == 0.5
        assert df.loc[owlbear_uuid, "sample_count"] == 2
        assert df.loc[morningstar_uuid, "take_rate"] == 0.5
        assert df.loc[morningstar_uuid, "sample_count"] == 2
        assert set(df.columns) == {
            "pack_number",
            "pick_number",
            "take_rate",
            "sample_count",
        }

    def test_absent_card_is_never_tallied(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = CardTakeRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1

    def test_different_pack_pick_number_produce_separate_keys(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = CardTakeRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0, pack_number=0, pick_number=0))
        metric.accumulate(_row("Owlbear", 1, 0, pack_number=1, pick_number=0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 2

    def test_unmatched_pick_name_is_not_treated_as_a_match(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = CardTakeRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Some Unmatchable Name", 1, 0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "take_rate"] == 0.0
        assert df.loc[owlbear_uuid, "sample_count"] == 1


class TestFirstPickRateMetric:
    def test_only_pack_zero_pick_zero_rows_are_tallied(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = FirstPickRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0, pack_number=0, pick_number=0))
        metric.accumulate(_row("Owlbear", 1, 0, pack_number=0, pick_number=1))
        metric.accumulate(_row("Owlbear", 1, 0, pack_number=1, pick_number=0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "sample_count"] == 1
        assert df.loc[owlbear_uuid, "take_rate"] == 1.0

    def test_key_has_no_extra_columns(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = FirstPickRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0, pack_number=0, pick_number=0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert set(df.columns) == {"nocab_uuid", "take_rate", "sample_count"}


class TestRankStratifiedTakeRateMetric:
    def test_separate_ranks_are_tallied_independently(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = RankStratifiedTakeRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row("Owlbear", 1, 0, rank="gold"))
        metric.accumulate(_row("Goblin Morningstar", 1, 0, rank="bronze"))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 2
        assert set(df["rank"]) == {"gold", "bronze"}
        assert set(df.columns) == {
            "nocab_uuid",
            "pack_number",
            "pick_number",
            "rank",
            "take_rate",
            "sample_count",
        }


@pytest.mark.parametrize(
    "metric_cls",
    [CardTakeRateMetric, FirstPickRateMetric, RankStratifiedTakeRateMetric],
)
def test_no_rows_accumulated_writes_empty_file(
    metric_cls: type, tmp_path: Path
) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = metric_cls(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    output_path = metric.finalize()

    df = pd.read_parquet(output_path)
    assert len(df) == 0


def test_default_output_paths_are_all_distinct() -> None:
    output_paths = {
        CardTakeRateMetric.DEFAULT_OUTPUT_PATH,
        FirstPickRateMetric.DEFAULT_OUTPUT_PATH,
        RankStratifiedTakeRateMetric.DEFAULT_OUTPUT_PATH,
    }
    assert len(output_paths) == 3


def test_default_output_paths_are_under_seventeenlands_draft_data() -> None:
    for metric_cls in (
        CardTakeRateMetric,
        FirstPickRateMetric,
        RankStratifiedTakeRateMetric,
    ):
        assert metric_cls.DEFAULT_OUTPUT_PATH.parent == Path(
            "data/metrics/seventeenlands/draft_data"
        )
