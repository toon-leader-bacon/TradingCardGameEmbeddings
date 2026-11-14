"""Per-game specs for the non-Pokemon Limitless deck stages.

LimitlessDeckSpec is one game's settings, read by
tcg_extraction_stages.py; its DeckSizes is deck_box/deck_sizes.py's.

Bounds count every section but the side deck (limitless_decklist.py
drops it). They are each game's construction rules; on the raw files
of 2026-11-08 every decklist already fit them, because Limitless
validates lists.
"""

from dataclasses import dataclass

from src.data_refinement.deck_box.deck_sizes import DeckSizes
from src.data_retrieval.limitless_play.downloader import LimitlessGame
from src.schema.game_id import GameId


@dataclass(frozen=True)
class LimitlessDeckSpec:
    """One non-Pokemon Limitless game.

    match_by_number: the game's binder carries PRINTED_NUMBER aliases
    ("OP13-001"), so a card is first matched by "<set>-<number>".
    """

    limitless_game: LimitlessGame
    game: GameId
    deck_sizes: DeckSizes
    match_by_number: bool


# Leader + 50
ONE_PIECE_SPEC = LimitlessDeckSpec(
    LimitlessGame.ONE_PIECE, GameId.ONE_PIECE, DeckSizes(51, 51), match_by_number=True
)
# 50 + 0-5 Digi-Eggs
DIGIMON_SPEC = LimitlessDeckSpec(
    LimitlessGame.DIGIMON, GameId.DIGIMON, DeckSizes(50, 55), match_by_number=True
)
GUNDAM_SPEC = LimitlessDeckSpec(
    LimitlessGame.GUNDAM, GameId.GUNDAM, DeckSizes(50, 50), match_by_number=True
)
# Leader + 50-60
DRAGON_BALL_FUSION_WORLD_SPEC = LimitlessDeckSpec(
    LimitlessGame.DRAGON_BALL_FUSION_WORLD,
    GameId.DRAGON_BALL_FUSION_WORLD,
    DeckSizes(51, 61),
    match_by_number=True,
)
# Leader + 50-60 main + the Z-deck
DRAGON_BALL_MASTERS_SPEC = LimitlessDeckSpec(
    LimitlessGame.DRAGON_BALL_MASTERS,
    GameId.DRAGON_BALL_MASTERS,
    DeckSizes(51, None),
    match_by_number=True,
)
BATTLE_SPIRITS_SAGA_SPEC = LimitlessDeckSpec(
    LimitlessGame.BATTLE_SPIRITS_SAGA,
    GameId.BATTLE_SPIRITS_SAGA,
    DeckSizes(50, 60),
    match_by_number=True,
)
# Leader + base + 50 or more
STAR_WARS_UNLIMITED_SPEC = LimitlessDeckSpec(
    LimitlessGame.STAR_WARS_UNLIMITED,
    GameId.STAR_WARS_UNLIMITED,
    DeckSizes(52, None),
    match_by_number=False,
)
LORCANA_SPEC = LimitlessDeckSpec(
    LimitlessGame.LORCANA, GameId.LORCANA, DeckSizes(60, None), match_by_number=False
)
