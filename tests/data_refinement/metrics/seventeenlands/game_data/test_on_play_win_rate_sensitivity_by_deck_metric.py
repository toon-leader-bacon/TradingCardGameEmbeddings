"""Skeleton-stage test stubs for
on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric. Bodies are filled in by
design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
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
    "match_number",
    "game_number",
    "on_play",
    "won",
    "deck_Owlbear",
]


def _row(
    on_play: bool,
    won: bool,
    owlbear_deck: int = 4,
    match_number: int = 0,
    game_number: int = 0,
) -> dict:
    return {
        "draft_id": "draft1",
        "match_number": match_number,
        "game_number": game_number,
        "on_play": on_play,
        "won": won,
        "deck_Owlbear": owlbear_deck,
    }


def test_computes_on_play_rate_minus_on_draw_rate_per_deck(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateSensitivityByDeckMetric(
        binder, _HEADER, GameId.MTG, DeckBox(), output_path=tmp_path / "out.parquet"
    )

    # Same deck (deck_Owlbear=4) across every game - on play: 1/1; on
    # draw: 0/1.
    metric.accumulate(_row(on_play=True, won=True, game_number=0))
    metric.accumulate(_row(on_play=False, won=False, game_number=1))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert df.iloc[0]["on_play_win_rate_sensitivity"] == 1.0


def test_sensitivity_is_none_when_deck_never_seen_on_one_side(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateSensitivityByDeckMetric(
        binder, _HEADER, GameId.MTG, DeckBox(), output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(on_play=True, won=True))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    # A pandas parquet round trip represents a written None as NaN in a
    # float64 column, not Python None - pd.isna() is the correct check.
    assert pd.isna(df.iloc[0]["on_play_win_rate_sensitivity"])


def test_identical_decks_across_games_dedupe_in_the_shared_deck_box(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    deck_box = DeckBox()
    metric = OnPlayWinRateSensitivityByDeckMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(on_play=True, won=True, game_number=0))
    metric.accumulate(_row(on_play=False, won=False, game_number=1))
    metric.finalize()

    assert len(list(deck_box.all_uuids(GameId.MTG))) == 1


def test_sample_count_sums_both_sides(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = OnPlayWinRateSensitivityByDeckMetric(
        binder, _HEADER, GameId.MTG, DeckBox(), output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(on_play=True, won=True, game_number=0))
    metric.accumulate(_row(on_play=True, won=False, game_number=1))
    metric.accumulate(_row(on_play=False, won=True, game_number=2))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["sample_count"] == 3
