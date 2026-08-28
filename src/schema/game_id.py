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
