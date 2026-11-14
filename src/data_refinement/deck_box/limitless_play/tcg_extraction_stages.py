"""Limitless deck stages for the card games other than Pokemon.

One subclass per game sets SPEC (game_spec.py); __init_subclass__
derives SOURCE_GAME, DEFAULT_RAW_PATH and DECK_SIZES from it, because
DeckExtractionStage reads SOURCE_GAME as a ClassVar. The bare
LimitlessTcgDeckExtractionStage has no SPEC and is never registered.
Cards are matched by tcg_card_index.py. Everything else (the loop, full
decks only, identity, return value) is extraction_stage.py's.
"""

from typing import Any, ClassVar

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.card_matching import CachedCardMatcher
from src.data_refinement.deck_box.limitless_play.extraction_stage import (
    LimitlessDeckExtractionStage,
)
from src.data_refinement.deck_box.limitless_play.game_spec import (
    BATTLE_SPIRITS_SAGA_SPEC,
    DIGIMON_SPEC,
    DRAGON_BALL_FUSION_WORLD_SPEC,
    DRAGON_BALL_MASTERS_SPEC,
    GUNDAM_SPEC,
    LORCANA_SPEC,
    ONE_PIECE_SPEC,
    STAR_WARS_UNLIMITED_SPEC,
    LimitlessDeckSpec,
)
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardRef,
)
from src.data_refinement.deck_box.limitless_play.tcg_card_index import (
    LimitlessTcgCardIndex,
)
from src.data_retrieval.limitless_play.downloader import LimitlessPlayDownloader


class LimitlessTcgDeckExtractionStage(LimitlessDeckExtractionStage):
    """Limitless decklists -> GenericDecks for SPEC's game (a subclass
    per game sets SPEC)."""

    SPEC: ClassVar[LimitlessDeckSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set SOURCE_GAME, DEFAULT_RAW_PATH and DECK_SIZES from the
        subclass's SPEC.

        Inputs: kwargs (passed on). Output: none.
        Side effects: sets the three class constants on cls.
        Exceptions: TypeError if the subclass sets no SPEC.
        """
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("SPEC")
        if not isinstance(spec, LimitlessDeckSpec):
            raise TypeError(f"{cls.__name__} must set SPEC to a LimitlessDeckSpec")
        cls.SOURCE_GAME = spec.game
        cls.DEFAULT_RAW_PATH = (
            LimitlessPlayDownloader.raw_data_dir_for(spec.limitless_game)
            / LimitlessPlayDownloader.STANDINGS_FILENAME
        )
        cls.DECK_SIZES = spec.deck_sizes

    def _build_card_matcher(
        self, card_lookup: CardLookup
    ) -> CachedCardMatcher[LimitlessCardRef]:
        """LimitlessTcgCardIndex over SOURCE_GAME's binder cards.

        Inputs: card_lookup. Output: LimitlessTcgCardIndex.
        Side effects: reads every binder card of SOURCE_GAME once.
        Exceptions: none.
        """
        return LimitlessTcgCardIndex.from_card_lookup(
            self.SOURCE_GAME, card_lookup, self.SPEC.match_by_number
        )


class LimitlessOnePieceDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = ONE_PIECE_SPEC


class LimitlessDigimonDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = DIGIMON_SPEC


class LimitlessGundamDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = GUNDAM_SPEC


class LimitlessDragonBallFusionWorldDeckExtractionStage(
    LimitlessTcgDeckExtractionStage
):
    SPEC = DRAGON_BALL_FUSION_WORLD_SPEC


class LimitlessDragonBallMastersDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = DRAGON_BALL_MASTERS_SPEC


class LimitlessBattleSpiritsSagaDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = BATTLE_SPIRITS_SAGA_SPEC


class LimitlessStarWarsUnlimitedDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = STAR_WARS_UNLIMITED_SPEC


class LimitlessLorcanaDeckExtractionStage(LimitlessTcgDeckExtractionStage):
    SPEC = LORCANA_SPEC


# Every stage above, for the run script
ALL_TCG_STAGES: tuple[type[LimitlessTcgDeckExtractionStage], ...] = (
    LimitlessOnePieceDeckExtractionStage,
    LimitlessDigimonDeckExtractionStage,
    LimitlessGundamDeckExtractionStage,
    LimitlessDragonBallFusionWorldDeckExtractionStage,
    LimitlessDragonBallMastersDeckExtractionStage,
    LimitlessBattleSpiritsSagaDeckExtractionStage,
    LimitlessStarWarsUnlimitedDeckExtractionStage,
    LimitlessLorcanaDeckExtractionStage,
)
