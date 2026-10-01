from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.metrics.generic.masked_field_multi_label_metric import (
    MaskedFieldMultiLabelMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(raw_content: dict) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Test Card",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class _FakeCardLookup:
    def __init__(self, cards: list[GenericCard]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return [card for card in self._cards if card.source_game == source_game]

    def version_for(self, source_game: GameId) -> str:
        return "test-version"


class _ColorsMetric(MaskedFieldMultiLabelMetric):
    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["colors"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("W", "U", "B")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _is_eligible(self, card: GenericCard) -> bool:
        return "skip" not in card.raw_content

    def _labels_for_card(self, card: GenericCard) -> frozenset[str]:
        return frozenset(self._raw_field_values(card))


def _scan(cards: list[GenericCard], tmp_path: Path) -> pd.DataFrame:
    metric = _ColorsMetric(_FakeCardLookup(cards), output_path=tmp_path / "o.parquet")
    return pd.read_parquet(metric.scan())


class TestScan:
    def test_labels_follow_label_values_order(self, tmp_path: Path) -> None:
        df = _scan([_card({"colors": ["B", "W"]})], tmp_path)

        assert list(df.iloc[0]["label"]) == ["W", "B"]
        assert list(df.iloc[0]["masked_field"]) == ["colors"]

    def test_absent_field_is_the_empty_set(self, tmp_path: Path) -> None:
        df = _scan([_card({"name": "Colorless"})], tmp_path)

        assert list(df.iloc[0]["label"]) == []

    def test_values_outside_the_vocabulary_are_dropped(self, tmp_path: Path) -> None:
        df = _scan([_card({"colors": ["U", "G"]})], tmp_path)

        assert list(df.iloc[0]["label"]) == ["U"]

    def test_a_scalar_field_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            _scan([_card({"colors": "WU"})], tmp_path)

    def test_ineligible_and_unknown_cards_are_skipped(self, tmp_path: Path) -> None:
        cards = [_card({"colors": ["W"]}), _card({"skip": 1}), _card({})]

        df = _scan(cards, tmp_path)

        assert len(df) == 1
        assert df.iloc[0]["nocab_uuid"] == str(cards[0].nocab_uuid)
