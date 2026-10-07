"""Tests for translator_tables.py: the per-game tables and
RARITY_TRANSLATORS."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    FieldRarityTranslator,
)
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
)
from src.data_refinement.metrics.gwent_one.rarity_mask_metric import (
    RarityMaskMetric as GwentRarityMaskMetric,
)
from src.data_refinement.metrics.hearthstonejson.card_mask_metrics import (
    RarityMaskMetric as HearthstoneRarityMaskMetric,
)
from src.data_refinement.metrics.scryfall.card_mask_metrics import (
    RarityMaskMetric as ScryfallRarityMaskMetric,
)
from src.data_refinement.metrics.spire_codex.card_mask_metrics import (
    RarityMaskMetric as SpireCodexRarityMaskMetric,
)
from src.schema.game_id import GameId
from src.schema.rarity_tier import LADDER

_RAW_POKEMON_CARDS = Path("data/raw/pokemon_tcg/cards")


def _table_of(game: GameId) -> dict:
    translator = RARITY_TRANSLATORS[game]
    assert isinstance(translator, FieldRarityTranslator)
    return dict(translator.tiers)


def test_every_translated_game_has_a_table_and_dominion_has_none() -> None:
    assert set(RARITY_TRANSLATORS) == {
        GameId.MTG,
        GameId.POKEMON,
        GameId.HEARTHSTONE,
        GameId.GWENT,
        GameId.SLAY_THE_SPIRE_2,
        GameId.FLESH_AND_BLOOD,
    }
    for game, translator in RARITY_TRANSLATORS.items():
        assert translator.game is game


@pytest.mark.parametrize("game", sorted(RARITY_TRANSLATORS, key=lambda g: g.value))
def test_every_game_maps_something_to_every_rung_of_the_ladder(
    game: GameId,
) -> None:
    assert set(LADDER) <= set(_table_of(game).values())


@pytest.mark.parametrize(
    "game, mask_metric_field",
    [
        (GameId.MTG, ScryfallRarityMaskMetric.MASKED_FIELD),
        (GameId.HEARTHSTONE, HearthstoneRarityMaskMetric.MASKED_FIELD),
        (GameId.GWENT, GwentRarityMaskMetric.MASKED_FIELD),
        (GameId.SLAY_THE_SPIRE_2, SpireCodexRarityMaskMetric.MASKED_FIELD),
    ],
)
def test_a_translator_field_is_its_games_rarity_mask_field(
    game: GameId, mask_metric_field: list[str]
) -> None:
    assert [RARITY_TRANSLATORS[game].masked_paths()[0][0]] == mask_metric_field


@pytest.mark.parametrize("game", sorted(RARITY_TRANSLATORS, key=lambda g: g.value))
def test_every_raw_rarity_in_the_live_binder_is_mapped(game: GameId) -> None:
    path = CardBinder.default_output_path(game)
    if not path.exists():
        pytest.skip(f"no {game.value} binder on disk")
    translator = RARITY_TRANSLATORS[game]

    for card in CardBinder.load([path]).all_cards(game):
        translator.rarity_tier_of(card)  # raises on an unmapped value


def test_every_raw_pokemon_rarity_is_mapped() -> None:
    """The binder drops Pokemon rarity until its ingestion change lands,
    so pin the table against the raw set files instead."""
    if not _RAW_POKEMON_CARDS.exists():
        pytest.skip("no raw pokemon_tcg cards on disk")
    table = _table_of(GameId.POKEMON)
    raw_values: Counter[str] = Counter()
    for path in _RAW_POKEMON_CARDS.glob("**/*.json"):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("rarity"):
                raw_values[row["rarity"]] += 1

    assert not set(raw_values) - set(table)
