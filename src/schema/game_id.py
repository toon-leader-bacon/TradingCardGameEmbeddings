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
