"""Skeleton-stage test stubs for game_card_average_metric.py's
GameCardAverageMetric, covered through its three
game_card_average_metrics.py concretes (WinRateWhenInDeckMetric,
OpeningHandWinRateMetric, DrawnWinRateMetric) - the same "cover the
shared base through its concretes" convention
tests/data_refinement/metrics/seventeenlands/draft_data/test_pack_card_tally_metrics.py
uses for PackCardTallyMetric. Bodies are filled in by
design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
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
    "won",
    "num_turns",
    "opening_hand_Owlbear",
    "drawn_Owlbear",
    "deck_Owlbear",
    "opening_hand_Goblin Morningstar",
    "drawn_Goblin Morningstar",
    "deck_Goblin Morningstar",
]


def _row(
    won: bool,
    owlbear_deck: int = 0,
    owlbear_opening_hand: int = 0,
    owlbear_drawn: int = 0,
    morningstar_deck: int = 0,
) -> dict:
    return {
        "won": won,
        "num_turns": 8,
        "opening_hand_Owlbear": owlbear_opening_hand,
        "drawn_Owlbear": owlbear_drawn,
        "deck_Owlbear": owlbear_deck,
        "opening_hand_Goblin Morningstar": 0,
        "drawn_Goblin Morningstar": 0,
        "deck_Goblin Morningstar": morningstar_deck,
    }


def _uuid_for(binder: CardBinder, name: str) -> object:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


class TestWinRateWhenInDeckMetric:
    def test_averages_won_indicator_across_games_with_card_in_deck(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = WinRateWhenInDeckMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row(won=True, owlbear_deck=4))
        metric.accumulate(_row(won=False, owlbear_deck=4))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "win_rate_when_in_deck"] == 0.5
        assert df.loc[owlbear_uuid, "sample_count"] == 2

    def test_card_never_in_any_deck_never_appears_in_output(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        metric = WinRateWhenInDeckMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row(won=True, owlbear_deck=4, morningstar_deck=0))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        morningstar_uuid = str(_uuid_for(binder, "Goblin Morningstar"))
        assert morningstar_uuid not in set(df["nocab_uuid"])

    def test_sample_count_matches_number_of_qualifying_games(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = WinRateWhenInDeckMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row(won=True, owlbear_deck=4))
        metric.accumulate(_row(won=True, owlbear_deck=4))
        metric.accumulate(_row(won=True, owlbear_deck=0))  # not present this game
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "sample_count"] == 2


class TestOpeningHandWinRateMetric:
    def test_averages_won_indicator_across_games_with_card_in_opening_hand(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = OpeningHandWinRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row(won=True, owlbear_opening_hand=1))
        metric.accumulate(_row(won=False, owlbear_opening_hand=1))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "opening_hand_win_rate"] == 0.5
        assert df.loc[owlbear_uuid, "sample_count"] == 2


class TestDrawnWinRateMetric:
    def test_averages_won_indicator_across_games_with_card_drawn(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DrawnWinRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_row(won=True, owlbear_drawn=1))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "drawn_win_rate"] == 1.0

    def test_counts_a_card_drawn_after_the_opening_hand_too(
        self, tmp_path: Path
    ) -> None:
        """drawn_<name> covers a card seen at any point in the game,
        not just the opening hand - distinct from
        OpeningHandWinRateMetric."""
        binder = _binder_with_cards(["Owlbear"])
        metric = DrawnWinRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        # Drawn later in the game, not in the opening hand.
        metric.accumulate(_row(won=True, owlbear_opening_hand=0, owlbear_drawn=1))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
        assert df.loc[owlbear_uuid, "sample_count"] == 1


def test_finalize_writes_one_parquet_row_per_card_with_expected_columns(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = WinRateWhenInDeckMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(won=True, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert set(df.columns) == {"nocab_uuid", "win_rate_when_in_deck", "sample_count"}
