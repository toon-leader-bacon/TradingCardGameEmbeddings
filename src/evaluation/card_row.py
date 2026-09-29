"""CardRow: one stored card's identity, the key that the embedding table,
card labels, samples and analyses all share."""

from dataclasses import dataclass
from uuid import UUID

from src.schema.game_id import GameId


@dataclass(frozen=True)
class CardRow:
    """One stored card's identity: what labels and analyses key on.
    Construction validates nothing and raises nothing."""

    nocab_uuid: UUID
    source_game: GameId
