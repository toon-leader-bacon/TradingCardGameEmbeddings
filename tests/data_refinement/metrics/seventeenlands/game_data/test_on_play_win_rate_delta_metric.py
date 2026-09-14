"""Skeleton-stage test stubs for on_play_win_rate_delta_metric.py's
OnPlayWinRateDeltaMetric. Bodies are filled in by
design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
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


_HEADER = ["on_play", "won", "deck_Owlbear", "deck_Goblin Morningstar"]


def _row(on_play: bool, won: bool, owlbear_deck: int = 0) -> dict:
    return {
        "on_play": on_play,
        "won": won,
        "deck_Owlbear": owlbear_deck,
        "deck_Goblin Morningstar": 0,
    }


def _uuid_for(binder: CardBinder, name: str) -> object:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


def test_computes_on_play_rate_minus_on_draw_rate_per_card(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateDeltaMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    # On play: 2 wins / 2 games = 1.0. On draw: 0 wins / 1 game = 0.0.
    metric.accumulate(_row(on_play=True, won=True, owlbear_deck=4))
    metric.accumulate(_row(on_play=True, won=True, owlbear_deck=4))
    metric.accumulate(_row(on_play=False, won=False, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    assert df.loc[owlbear_uuid, "on_play_win_rate_delta"] == 1.0


def test_delta_is_none_when_card_never_seen_on_one_side(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateDeltaMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    # Only ever seen on the play, never on the draw.
    metric.accumulate(_row(on_play=True, won=True, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    # A pandas parquet round trip represents a written None as NaN in a
    # float64 column, not Python None - pd.isna() is the correct check.
    assert pd.isna(df.loc[owlbear_uuid, "on_play_win_rate_delta"])


def test_sample_count_sums_both_sides(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateDeltaMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(on_play=True, won=True, owlbear_deck=4))
    metric.accumulate(_row(on_play=True, won=False, owlbear_deck=4))
    metric.accumulate(_row(on_play=False, won=True, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    assert df.loc[owlbear_uuid, "sample_count"] == 3


def test_only_deck_present_cards_are_tallied(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = OnPlayWinRateDeltaMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(on_play=True, won=True, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    morningstar_uuid = str(_uuid_for(binder, "Goblin Morningstar"))
    assert morningstar_uuid not in set(df["nocab_uuid"])
