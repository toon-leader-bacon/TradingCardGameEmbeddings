from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.metrics.masked_field_metric import MaskedFieldMetric
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(raw_content: dict) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name="Test Card",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id="1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class _FakeCardLookup:
    """A minimal CardLookup - only all_cards() is exercised by
    MaskedFieldMetric, so that's the only method this fake implements."""

    def __init__(self, cards: list[GenericCard]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return [card for card in self._cards if card.source_game == source_game]


class _TopLevelFieldMetric(MaskedFieldMetric):
    """A concrete MaskedFieldMetric over a flat top-level field, with
    no eligibility filter - exercises scan()'s default sequence."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["faction"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("skellige", "monster")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class _NestedFieldMetric(MaskedFieldMetric):
    """A concrete MaskedFieldMetric over a nested field path -
    exercises _raw_field_value()'s multi-segment walk."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["stat_block", "attack"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("5",)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class _IneligibleTypeMetric(MaskedFieldMetric):
    """A concrete MaskedFieldMetric that overrides _is_eligible() -
    exercises scan() actually filtering a card out."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["faction"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("skellige",)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _is_eligible(self, card: GenericCard) -> bool:
        return card.raw_content["type"] == "unit"

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class TestScan:
    def test_writes_one_row_per_card(self, tmp_path: Path) -> None:
        cards = [
            _card({"faction": "skellige"}),
            _card({"faction": "monster"}),
        ]
        metric = _TopLevelFieldMetric(
            _FakeCardLookup(cards), output_path=tmp_path / "out.parquet"
        )

        output_path = metric.scan()

        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert set(df["label"]) == {"skellige", "monster"}
        assert set(df["nocab_uuid"]) == {str(card.nocab_uuid) for card in cards}
        assert all(list(field) == ["faction"] for field in df["masked_field"])

    def test_ineligible_card_excluded(self, tmp_path: Path) -> None:
        cards = [
            _card({"faction": "skellige", "type": "unit"}),
            _card({"faction": "skellige", "type": "artifact"}),
        ]
        metric = _IneligibleTypeMetric(
            _FakeCardLookup(cards), output_path=tmp_path / "out.parquet"
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1

    def test_other_game_cards_excluded(self, tmp_path: Path) -> None:
        gwent_card = _card({"faction": "skellige"})
        other_card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name="Not Gwent",
            raw_content={"faction": "monster"},
            provenance=Provenance(
                data_source=DataSource.GWENT_ONE,
                source_id="2",
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        metric = _TopLevelFieldMetric(
            _FakeCardLookup([gwent_card, other_card]),
            output_path=tmp_path / "out.parquet",
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["nocab_uuid"] == str(gwent_card.nocab_uuid)

    def test_no_eligible_cards_writes_empty_file(self, tmp_path: Path) -> None:
        metric = _TopLevelFieldMetric(
            _FakeCardLookup([]), output_path=tmp_path / "out.parquet"
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = _TopLevelFieldMetric(
            _FakeCardLookup([_card({"faction": "skellige"})]),
            output_path=nested_path,
        )

        output_path = metric.scan()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_default_output_path_used_when_not_overridden(self) -> None:
        metric = _TopLevelFieldMetric(_FakeCardLookup([]))
        assert metric._output_path == _TopLevelFieldMetric.DEFAULT_OUTPUT_PATH


class TestRawFieldValue:
    def test_top_level_field(self) -> None:
        metric = _TopLevelFieldMetric(_FakeCardLookup([]))
        card = _card({"faction": "skellige"})
        assert metric._raw_field_value(card) == "skellige"

    def test_nested_field_path(self) -> None:
        metric = _NestedFieldMetric(_FakeCardLookup([]))
        card = _card({"stat_block": {"attack": "5"}})
        assert metric._raw_field_value(card) == "5"

    def test_missing_segment_raises_key_error(self) -> None:
        metric = _TopLevelFieldMetric(_FakeCardLookup([]))
        card = _card({})
        with pytest.raises(KeyError):
            metric._raw_field_value(card)
