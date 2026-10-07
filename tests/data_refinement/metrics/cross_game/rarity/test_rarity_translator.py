"""Tests for rarity_translator.py's FieldRarityTranslator."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    FieldRarityTranslator,
    UnmappedRarityError,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.rarity_tier import RarityTier

_TRANSLATOR = FieldRarityTranslator(
    game=GameId.GWENT,
    field="rarity",
    tiers={"common": RarityTier.TIER_1, "legendary": RarityTier.TIER_4},
)


def _card(raw_content: dict[str, Any]) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name="Card",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id="x",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestFieldRarityTranslator:
    def test_maps_a_listed_value_to_its_tier(self) -> None:
        card = _card({"rarity": "legendary"})

        assert _TRANSLATOR.rarity_tier_of(card) is RarityTier.TIER_4
        assert _TRANSLATOR.raw_rarity_of(card) == "legendary"

    @pytest.mark.parametrize("raw_content", [{}, {"rarity": None}, {"rarity": "  "}])
    def test_a_missing_null_or_blank_field_has_no_tier(
        self, raw_content: dict[str, Any]
    ) -> None:
        card = _card(raw_content)

        assert _TRANSLATOR.rarity_tier_of(card) is None
        assert _TRANSLATOR.raw_rarity_of(card) is None

    def test_an_unlisted_value_raises_unmapped_rarity_error(self) -> None:
        with pytest.raises(UnmappedRarityError, match="epic"):
            _TRANSLATOR.rarity_tier_of(_card({"rarity": "epic"}))

    def test_a_non_string_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="rarity"):
            _TRANSLATOR.raw_rarity_of(_card({"rarity": 3}))

    def test_masked_paths_is_the_rarity_field(self) -> None:
        assert _TRANSLATOR.masked_paths() == (("rarity",),)


def test_unmapped_rarity_error_names_the_game_and_value() -> None:
    error = UnmappedRarityError(GameId.POKEMON, "Weird Rare")

    assert error.game is GameId.POKEMON
    assert error.raw_value == "Weird Rare"
    assert "pokemon" in str(error) and "Weird Rare" in str(error)
