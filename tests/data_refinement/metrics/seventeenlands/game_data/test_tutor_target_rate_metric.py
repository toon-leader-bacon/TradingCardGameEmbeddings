"""Skeleton-stage test stubs for tutor_target_rate_metric.py's
TutorTargetRateMetric. Bodies are filled in by design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
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
    "deck_Owlbear",
    "tutored_Owlbear",
    "deck_Goblin Morningstar",
    "tutored_Goblin Morningstar",
]


def _row(owlbear_deck: int = 0, owlbear_tutored: int = 0) -> dict:
    return {
        "deck_Owlbear": owlbear_deck,
        "tutored_Owlbear": owlbear_tutored,
        "deck_Goblin Morningstar": 0,
        "tutored_Goblin Morningstar": 0,
    }


def _uuid_for(binder: CardBinder, name: str) -> object:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


def test_rate_is_times_tutored_over_times_in_deck(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=1))
    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=0))
    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=0))
    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=0))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    assert df.loc[owlbear_uuid, "tutor_target_rate"] == 0.25
    assert df.loc[owlbear_uuid, "sample_count"] == 4


def test_card_in_deck_but_never_tutored_gets_zero_rate_not_omitted(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=0))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    assert df.loc[owlbear_uuid, "tutor_target_rate"] == 0.0


def test_card_never_in_any_deck_never_appears_in_output(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=1))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    morningstar_uuid = str(_uuid_for(binder, "Goblin Morningstar"))
    assert morningstar_uuid not in set(df["nocab_uuid"])


def test_sample_count_is_times_in_deck(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=1))
    metric.accumulate(_row(owlbear_deck=4, owlbear_tutored=0))
    metric.accumulate(_row(owlbear_deck=0, owlbear_tutored=0))  # not in deck
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = str(_uuid_for(binder, "Owlbear"))
    assert df.loc[owlbear_uuid, "sample_count"] == 2
