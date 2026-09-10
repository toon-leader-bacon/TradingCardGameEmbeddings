import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.play_gwent.leader_labels import LEADER_NAMES
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _gwent_one_card(card_id: int, name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=str(card_id),
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _card_binder(cards: dict[int, str]) -> CardBinder:
    binder = CardBinder()
    for card_id, name in cards.items():
        card = _gwent_one_card(card_id, name)
        binder.create(card)
        binder.register_alias(
            GameId.GWENT, DataSource.GWENT_ONE, str(card_id), card.nocab_uuid
        )
    binder.ensure_unknown_card(GameId.GWENT)
    return binder


def _guide_row(
    guide_id: int,
    leader_id: int | None,
    other_card_ids: list[int],
    name: str | None = None,
) -> dict:
    src_card_templates = (
        other_card_ids if leader_id is None else [leader_id, *other_card_ids]
    )
    row: dict = {
        "id": guide_id,
        "deck": {"id": guide_id * 10, "srcCardTemplates": src_card_templates},
    }
    if leader_id is not None:
        row["leaderId"] = leader_id
    if name is not None:
        row["name"] = name
    return row


class TestAccumulateWithValidTarget:
    def test_writes_row_and_registers_deck(self, tmp_path: Path) -> None:
        leader_name = LEADER_NAMES[0]
        binder = _card_binder({1: leader_name, 2: "Some Unit"})
        box = DeckBox()
        output_path = tmp_path / "leader_mask.parquet"
        metric = LeaderMaskedFromDeckMetric(binder, box, output_path=output_path)

        metric.accumulate(_guide_row(407697, leader_id=1, other_card_ids=[2, 2]))
        metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 1
        leader_uuid = binder.get_by_alias(
            GameId.GWENT, DataSource.GWENT_ONE, "1"
        ).nocab_uuid
        unit_uuid = binder.get_by_alias(
            GameId.GWENT, DataSource.GWENT_ONE, "2"
        ).nocab_uuid
        assert df.iloc[0]["target_card_uuid"] == str(leader_uuid)
        assert df.iloc[0]["label"] == leader_name

        deck = box.get_by_uuid(UUID(df.iloc[0]["deck_uuid"]))
        assert deck is not None
        assert deck.card_nocab_uuids == [leader_uuid, unit_uuid, unit_uuid]


class TestLabelFallback:
    def test_name_outside_leader_names_falls_back_to_other(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _card_binder({1: "Not A Real Leader", 2: "Some Unit"})
        box = DeckBox()
        output_path = tmp_path / "leader_mask.parquet"
        metric = LeaderMaskedFromDeckMetric(binder, box, output_path=output_path)

        with caplog.at_level(logging.ERROR):
            metric.accumulate(_guide_row(1, leader_id=1, other_card_ids=[2]))
        metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 1
        assert df.iloc[0]["label"] == OTHER_LABEL
        assert "not in LEADER_NAMES" in caplog.text


class TestMissingOrUnresolvableLeader:
    def test_missing_leader_id_skips_row_but_still_registers_deck(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _card_binder({2: "Some Unit"})
        box = DeckBox()
        output_path = tmp_path / "leader_mask.parquet"
        metric = LeaderMaskedFromDeckMetric(binder, box, output_path=output_path)

        with caplog.at_level(logging.ERROR):
            metric.accumulate(_guide_row(1, leader_id=None, other_card_ids=[2]))
        metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 0
        assert len(list(box.all_decks(GameId.GWENT))) == 1
        assert "missing leaderId" in caplog.text

    def test_unresolvable_leader_id_skips_row(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _card_binder({2: "Some Unit"})
        box = DeckBox()
        output_path = tmp_path / "leader_mask.parquet"
        metric = LeaderMaskedFromDeckMetric(binder, box, output_path=output_path)

        with caplog.at_level(logging.ERROR):
            metric.accumulate(_guide_row(1, leader_id=999999, other_card_ids=[2]))
        metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 0
        assert "did not resolve to a known card" in caplog.text


class TestDeckBoxIdempotency:
    def test_repeated_accumulate_does_not_duplicate_deck(self, tmp_path: Path) -> None:
        leader_name = LEADER_NAMES[0]
        binder = _card_binder({1: leader_name, 2: "Some Unit"})
        box = DeckBox()
        output_path = tmp_path / "leader_mask.parquet"
        metric = LeaderMaskedFromDeckMetric(binder, box, output_path=output_path)
        row = _guide_row(407697, leader_id=1, other_card_ids=[2])

        metric.accumulate(row)
        metric.accumulate(row)
        metric.finalize()

        assert len(list(box.all_decks(GameId.GWENT))) == 1
        df = pd.read_parquet(output_path)
        assert len(df) == 2  # one output row per accumulate() call, same deck_uuid
        assert df.iloc[0]["deck_uuid"] == df.iloc[1]["deck_uuid"]


def test_default_output_path() -> None:
    assert LeaderMaskedFromDeckMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/play_gwent/leader_masked_from_deck.parquet"
    )
