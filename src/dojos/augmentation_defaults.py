"""Default train-only augmentations per game: which card fields to jitter
and how often.

This is data knowledge: each game's raw_content keys are chosen by its
card-binder ingestion stage (named per section below), so a key renamed
there must be renamed here. The preflight tally report
(scripts/run_training.py) flags a mask that stops matching.

A game's default applies to every contrastive dojo over that game's cards;
a run config can replace it per dojo (`mods:`). Masking the deck-defining
field (the one field that nearly identifies which deck a card came from)
keeps same-deck contrastive positives from being solved by that field
alone. A game with no entry gets no augmentation.
"""

from typing import Mapping

from src.dojos.mods.card_field_mods import FieldMask
from src.dojos.mods.mod_specs import (
    ModSpec,
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    WeightedFieldMaskSpec,
)
from src.schema.game_id import GameId
from src.utils.drop_table import DropTable

# Weights below were checked against each game's final deck box on
# 2026-10-01: for a field that nearly names the deck, two cards of one deck
# share its value far more often than two random deck cards (Gwent faction
# 70% vs 15%, StS2 color 76% vs 16%, MTG set 62% vs 4%, Pokemon set 62% vs
# 2%, FaB class 46% vs 13%), so that field is masked about half the time.

# gwent_one (card_binder/gwent_one/ingestion_stage.py). A deck is one
# faction plus neutrals; faction-duo (15 cards) names the faction too.
_GWENT_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (25, FieldMask()),
        (50, FieldMask.of_keys("faction", "faction-duo")),
        (15, FieldMask.of_keys("name")),
        (10, FieldMask.of_keys("category")),
    ]
)

# spire_codex (card_binder/spire_codex/ingestion_stage.py). A run is one
# character's cards (color) plus colorless ones.
_STS2_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (25, FieldMask()),
        (50, FieldMask.of_keys("color")),
        (15, FieldMask.of_keys("name")),
        (10, FieldMask.of_keys("upgrade_description")),
    ]
)

# cardvault_fabtcg (card_binder/cardvault_fabtcg/ingestion_stage.py). The
# deck-defining class is a word inside typebox (the leaned card has no
# classes/talents keys), so it is reachable only by masking the whole
# typebox; back_face exists on ~3% of deck cards.
_FAB_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (35, FieldMask()),
        (25, FieldMask.of_keys("name")),
        (
            10,
            DropTable.of(
                [
                    (1, FieldMask((("back_face", "name"),))),
                    (1, FieldMask.of_keys("back_face")),
                ]
            ),
        ),
        (30, FieldMask.of_keys("typebox")),
    ]
)

# scryfall (card_binder/scryfall/ingestion_stage.py). The MTG deck box holds
# 17lands limited decks: one set, usually two colors. set (and legalities,
# which track the set's age) nearly names the deck, so it is masked most;
# mana_cost spells out the colors too, so it is sometimes masked with them.
_MTG_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (25, FieldMask()),
        (50, FieldMask.of_keys("set", "legalities")),
        (
            15,
            DropTable.of(
                [
                    (1, FieldMask.of_keys("colors", "color_identity")),
                    (1, FieldMask.of_keys("colors", "color_identity", "mana_cost")),
                ]
            ),
        ),
        (10, FieldMask.of_keys("name")),
    ]
)

# pokemon_tcg (card_binder/pokemon_tcg/ingestion_stage.py). Theme decks are
# built from one set (regulationMark tracks the set too) around one or two
# energy types, with whole evolution lines, so each of those nearly pairs
# a deck's cards by itself.
_POKEMON_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (25, FieldMask()),
        (45, FieldMask.of_keys("set", "regulationMark")),
        (10, FieldMask.of_keys("types")),
        (10, FieldMask.of_keys("evolvesFrom", "evolvesTo")),
        (10, FieldMask.of_keys("name")),
    ]
)

# dominiontabs (card_binder/dominiontabs/ingestion_stage.py). A final deck
# is drawn from a random ten-card kingdom plus the base cards, so no field
# names the deck; set is not ingested. description carries the card's
# identity and is never masked.
_DOMINION_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (45, FieldMask()),
        (20, FieldMask.of_keys("name")),
        (15, FieldMask.of_keys("cost", "potcost", "debtcost")),
        (20, FieldMask.of_keys("types")),
    ]
)


# hearthstonejson (card_binder/hearthstonejson/ingestion_stage.py). A deck
# is one class plus neutrals, so cardClass is the deck-defining field;
# classes (multiclass cards) and runeCost (Death Knight only) name it too.
# No Hearthstone deck box exists yet, so these weights are by analogy with
# Gwent's faction, not measured.
_HEARTHSTONE_MASKS: DropTable[FieldMask] = DropTable.of(
    [
        (25, FieldMask()),
        (50, FieldMask.of_keys("cardClass", "classes", "runeCost")),
        (15, FieldMask.of_keys("name")),
        (10, FieldMask.of_keys("set")),
    ]
)


def _standard_augmentations(masks: DropTable[FieldMask]) -> tuple[ModSpec, ...]:
    """Key-order shuffle, then the game's weighted masks, then a small
    chance of one more random key masked.

    Inputs: masks. Output: tuple of ModSpec. Side effects: none.
    Exceptions: none.
    """
    return (
        ShuffleKeysSpec(),
        WeightedFieldMaskSpec(masks),
        RandomKeyMaskSpec(probability=0.1),
    )


DEFAULT_AUGMENTATIONS: Mapping[GameId, tuple[ModSpec, ...]] = {
    GameId.GWENT: _standard_augmentations(_GWENT_MASKS),
    GameId.SLAY_THE_SPIRE_2: _standard_augmentations(_STS2_MASKS),
    GameId.FLESH_AND_BLOOD: _standard_augmentations(_FAB_MASKS),
    GameId.MTG: _standard_augmentations(_MTG_MASKS),
    GameId.POKEMON: _standard_augmentations(_POKEMON_MASKS),
    GameId.DOMINION: _standard_augmentations(_DOMINION_MASKS),
    GameId.HEARTHSTONE: _standard_augmentations(_HEARTHSTONE_MASKS),
}


def default_augmentations_for(game: GameId) -> tuple[ModSpec, ...]:
    """game's default augmentation specs; () for a game with none.

    Inputs: game. Output: tuple of ModSpec.
    Side effects: none. Exceptions: none.

    Example:
        >>> [type(spec).__name__ for spec in default_augmentations_for(GameId.GWENT)]
        ['ShuffleKeysSpec', 'WeightedFieldMaskSpec', 'RandomKeyMaskSpec']
    """
    return DEFAULT_AUGMENTATIONS.get(game, ())
