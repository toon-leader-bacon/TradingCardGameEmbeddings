"""The fixed set of trading card games this system onboards.

GameId / GenericCard / GenericDeck form a thin identity envelope
(nocab_uuid / provenance / source_game / name) plus a free-form
raw_content blob, rather than a rigid shared schema that would force
semantically different per-game concepts (MTG mana cost vs. Pokemon
HP) into shared fields.
"""

from enum import Enum


class GameId(str, Enum):
    """Which trading card game a GenericCard/GenericDeck belongs to.

    Extensible per new game onboarded — adding a game to this system
    means adding a case here plus a CardIngestionStage for at least
    one raw source of that game's cards.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    MTG = "mtg"
    POKEMON = "pokemon"
    YUGIOH = "yugioh"
    HEARTHSTONE = "hearthstone"
    GWENT = "gwent"
    SLAY_THE_SPIRE_2 = "slay_the_spire_2"  # Slay the Spire 2 (see
    # src/data_retrieval/spire_codex/downloader.py and
    # src/data_refinement/card_binder/spire_codex/ingestion_stage.py).
    FLESH_AND_BLOOD = "flesh_and_blood"  # Flesh and Blood (see
    # src/data_retrieval/cardvault_fabtcg/card_downloader.py and
    # src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py).
    DOMINION = "dominion"  # Dominion, the board/deckbuilding game (see
    # src/data_retrieval/dominiontabs/downloader.py and
    # src/data_refinement/card_binder/dominiontabs/ingestion_stage.py).
    LORCANA = "lorcana"  # Disney Lorcana (see
    # src/data_retrieval/tcgjson/downloader.py and
    # src/data_refinement/card_binder/tcgjson/ingestion_stage.py).
    # The tcgjson games, like LORCANA (cards from
    # src/data_refinement/card_binder/tcgjson/; YUGIOH above is one too).
    DIGIMON = "digimon"
    ONE_PIECE = "one_piece"
    STAR_WARS_UNLIMITED = "star_wars_unlimited"
    UNION_ARENA = "union_arena"
    RIFTBOUND = "riftbound"
