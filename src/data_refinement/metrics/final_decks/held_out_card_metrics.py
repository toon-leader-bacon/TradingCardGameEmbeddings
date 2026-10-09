"""One HeldOutDeckCardMetric per published deck box
(data/final/decks/<game>.db). ClassVars only; the family lives in
../generic/held_out_deck_card/.

Sampling caps keep every output in the hundreds of thousands of rows:
the small boxes use every deck and several targets per deck, the big
ones (StS2 2.8M decks, MTG 4.8M) a 500k-deck sample with one target
each, so no deck sits in two splits there.
"""

from pathlib import Path

from src.data_refinement.metrics.generic.held_out_deck_card.metric import (
    HeldOutDeckCardMetric,
)
from src.data_refinement.metrics.generic.held_out_deck_card.sampling import (
    HeldOutCardSampling,
)
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/final_decks")
_DECOY_COUNT = 7
_BIG_BOX_MAX_DECKS = 500_000


class PokemonHeldOutCardMetric(HeldOutDeckCardMetric):
    """Pokemon deck minus one card -> which of 8 candidates (188 decks).

    Runtime: about 5 s for the 188-deck box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.POKEMON
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_pokemon.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=None, targets_per_deck=8, decoy_count=_DECOY_COUNT
    )


class FleshAndBloodHeldOutCardMetric(HeldOutDeckCardMetric):
    """FaB deck minus one card -> which of 8 candidates (4.2k decks).

    Runtime: about 10 s for the 4,172-deck box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.FLESH_AND_BLOOD
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_flesh_and_blood.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=None, targets_per_deck=8, decoy_count=_DECOY_COUNT
    )


class GwentHeldOutCardMetric(HeldOutDeckCardMetric):
    """Gwent deck minus one card -> which of 8 candidates (60k decks).

    Runtime: about 40 s for the 60,292-deck box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.GWENT
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_gwent.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=None, targets_per_deck=4, decoy_count=_DECOY_COUNT
    )


class DominionHeldOutCardMetric(HeldOutDeckCardMetric):
    """Dominion deck minus one card -> which of 8 candidates (532k decks).

    Runtime: about 3.5 min for the 531,675-deck box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.DOMINION
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_dominion.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=None, targets_per_deck=1, decoy_count=_DECOY_COUNT
    )


class SlayTheSpire2HeldOutCardMetric(HeldOutDeckCardMetric):
    """StS2 deck minus one card -> which of 8 candidates (500k of 2.8M decks).

    Runtime: about 6.5 min for the 12 GB box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.SLAY_THE_SPIRE_2
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_slay_the_spire_2.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=_BIG_BOX_MAX_DECKS, targets_per_deck=1, decoy_count=_DECOY_COUNT
    )


class MtgHeldOutCardMetric(HeldOutDeckCardMetric):
    """MTG limited deck minus one card -> which of 8 candidates (500k of 4.8M decks).

    Runtime: about 7 min for the 6.9M-deck, 40 GB box (measured 2026-10-08).
    """

    SOURCE_GAME = GameId.MTG
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "held_out_card_mtg.parquet"
    SAMPLING = HeldOutCardSampling(
        max_decks=_BIG_BOX_MAX_DECKS, targets_per_deck=1, decoy_count=_DECOY_COUNT
    )
