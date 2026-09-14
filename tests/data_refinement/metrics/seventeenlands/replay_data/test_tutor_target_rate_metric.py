"""Tests for tutor_target_rate_metric.py's (replay-level)
TutorTargetRateMetric - a distinct class from
game_data.tutor_target_rate_metric.TutorTargetRateMetric, tested
independently here.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = ["deck_Owlbear", "user_turn_1_cards_tutored"]


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


def _binder_with_owlbear(arena_id: str) -> CardBinder:
    binder = CardBinder()
    card = _make_card("Owlbear")
    binder.create(card)
    binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card.nocab_uuid)
    return binder


def _uuid_for(binder: CardBinder, name: str) -> str:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return str(cards[0].nocab_uuid)


def _row(deck: int, user_tutored: str | float = float("nan")) -> dict:
    return {"deck_Owlbear": deck, "user_turn_1_cards_tutored": user_tutored}


def test_rate_is_times_tutored_over_times_in_deck(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, user_tutored="1"))
    metric.accumulate(_row(deck=4))
    metric.accumulate(_row(deck=4))
    metric.accumulate(_row(deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for(binder, "Owlbear")
    assert df.loc[owlbear_uuid, "tutor_target_rate"] == 0.25
    assert df.loc[owlbear_uuid, "sample_count"] == 4


def test_card_in_deck_but_never_tutored_gets_zero_rate_not_omitted(
    tmp_path: Path,
) -> None:
    binder = _binder_with_owlbear("1")
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for(binder, "Owlbear")
    assert df.loc[owlbear_uuid, "tutor_target_rate"] == 0.0


def test_only_scans_user_turn_half_turns(tmp_path: Path) -> None:
    """There is no oppo_turn_N_cards_tutored column at all (opponent
    tutoring is hidden) - unlike CastRateMetric/DiscardRateMetric this
    isn't a scope choice, it's the only column that exists. A header
    with no oppo_turn_* columns at all must still work without error."""
    binder = _binder_with_owlbear("1")
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, user_tutored="1"))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for(binder, "Owlbear")
    assert df.loc[owlbear_uuid, "tutor_target_rate"] == 1.0


def test_card_never_in_any_deck_never_appears_in_output(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=0, user_tutored="1"))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.shape[0] == 0


def test_sample_count_is_times_in_deck(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = TutorTargetRateMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, user_tutored="1"))
    metric.accumulate(_row(deck=4))
    metric.accumulate(_row(deck=0))  # not in deck
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for(binder, "Owlbear")
    assert df.loc[owlbear_uuid, "sample_count"] == 2
