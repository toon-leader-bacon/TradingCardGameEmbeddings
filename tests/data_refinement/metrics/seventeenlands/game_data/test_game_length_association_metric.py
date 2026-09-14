"""Skeleton-stage test stubs for
game_length_association_metric.py's GameLengthAssociationMetric.
Bodies are filled in by design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
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


_HEADER = ["num_turns", "deck_Owlbear", "deck_Goblin Morningstar"]


def _row(num_turns: int, owlbear_deck: int = 0, morningstar_deck: int = 0) -> dict:
    return {
        "num_turns": num_turns,
        "deck_Owlbear": owlbear_deck,
        "deck_Goblin Morningstar": morningstar_deck,
    }


def _uuid_for(binder: CardBinder, name: str) -> object:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


def test_subtracts_format_wide_average_num_turns_from_each_cards_own_average(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = GameLengthAssociationMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    # Owlbear games run long (12 turns); Goblin Morningstar games are
    # quick (4 turns) - format-wide average is (12 + 4) / 2 = 8.
    metric.accumulate(_row(num_turns=12, owlbear_deck=4))
    metric.accumulate(_row(num_turns=4, morningstar_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    morningstar_uuid = str(_uuid_for(binder, "Goblin Morningstar"))
    assert df.loc[owlbear_uuid, "game_length_association"] == 4.0  # 12 - 8
    assert df.loc[morningstar_uuid, "game_length_association"] == -4.0  # 4 - 8


def test_global_baseline_is_tracked_even_for_games_with_no_tallied_card(
    tmp_path: Path,
) -> None:
    """_extra_accumulate() updates the format-wide baseline
    unconditionally, every row - not gated on any card's presence."""
    binder = _binder_with_cards(["Owlbear"])
    metric = GameLengthAssociationMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    # A game with no tallied card at all still counts toward the
    # global baseline.
    metric.accumulate(_row(num_turns=20))
    metric.accumulate(_row(num_turns=10, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    # global average = (20 + 10) / 2 = 15; owlbear's own average = 10.
    assert df.loc[owlbear_uuid, "game_length_association"] == -5.0


def test_never_overrides_accumulate_itself() -> None:
    """GameLengthAssociationMetric must go through
    GameCardAverageMetric's _extra_accumulate() hook, not override
    accumulate() directly - see the module docstring's Template Method
    rationale."""
    assert "accumulate" not in GameLengthAssociationMetric.__dict__


def test_finalize_writes_one_parquet_row_per_card_with_expected_columns(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = GameLengthAssociationMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(num_turns=10, owlbear_deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert set(df.columns) == {
        "nocab_uuid",
        "game_length_association",
        "sample_count",
    }
