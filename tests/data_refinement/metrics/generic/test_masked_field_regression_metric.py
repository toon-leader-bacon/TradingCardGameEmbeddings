from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import pandas as pd

from src.data_refinement.metrics.generic.masked_field_regression_metric import (
    MaskedFieldRegressionMetric,
)
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
    """A minimal CardLookup - only all_cards()/version_for() are
    exercised by MaskedFieldRegressionMetric, so those are the only
    methods this fake implements."""

    def __init__(self, cards: list[GenericCard]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return [card for card in self._cards if card.source_game == source_game]

    def version_for(self, source_game: GameId) -> str:
        return "test-version"


class _TopLevelFieldMetric(MaskedFieldRegressionMetric):
    """A concrete MaskedFieldRegressionMetric over a flat top-level
    field, with no eligibility filter - exercises scan()'s default
    sequence."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["provision"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _value_for_card(self, card: GenericCard) -> float:
        return float(self._raw_field_value(card))


class _IneligibleTypeMetric(MaskedFieldRegressionMetric):
    """A concrete MaskedFieldRegressionMetric that overrides
    _is_eligible() - exercises scan() actually filtering a card out."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["provision"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("unused.parquet")

    def _is_eligible(self, card: GenericCard) -> bool:
        return card.raw_content["type"] == "unit"

    def _value_for_card(self, card: GenericCard) -> float:
        return float(self._raw_field_value(card))


class TestScan:
    def test_writes_one_row_per_card(self, tmp_path: Path) -> None:
        cards = [_card({"provision": "5"}), _card({"provision": "10"})]
        metric = _TopLevelFieldMetric(
            _FakeCardLookup(cards), output_path=tmp_path / "out.parquet"
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 2
        assert set(df["label"]) == {5.0, 10.0}
        assert set(df["nocab_uuid"]) == {str(card.nocab_uuid) for card in cards}
        assert all(list(field) == ["provision"] for field in df["masked_field"])

    def test_ineligible_card_excluded(self, tmp_path: Path) -> None:
        cards = [
            _card({"provision": "5", "type": "unit"}),
            _card({"provision": "5", "type": "stratagem"}),
        ]
        metric = _IneligibleTypeMetric(
            _FakeCardLookup(cards), output_path=tmp_path / "out.parquet"
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1

    def test_unknown_sentinel_excluded_without_raising(self, tmp_path: Path) -> None:
        # Same structural guard as MaskedFieldMetric.scan() - see that
        # class's own test of the same name.
        cards = [_card({"provision": "5"}), _card({})]
        metric = _TopLevelFieldMetric(
            _FakeCardLookup(cards), output_path=tmp_path / "out.parquet"
        )

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["label"] == 5.0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = _TopLevelFieldMetric(
            _FakeCardLookup([_card({"provision": "5"})]),
            output_path=nested_path,
        )

        output_path = metric.scan()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_default_output_path_used_when_not_overridden(self) -> None:
        metric = _TopLevelFieldMetric(_FakeCardLookup([]))
        assert metric._output_path == _TopLevelFieldMetric.DEFAULT_OUTPUT_PATH
