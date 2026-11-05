"""DeckSource: what a ContrastiveDojo deals its decks from.

DeckBoxDealer (real decks from a deck snapshot) and NeighbourDealer
(pseudo-decks of teacher neighbours) both satisfy it, so every contrastive
style trains on either without knowing which.
"""

from typing import Iterator, Protocol

from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.splits import Split


class DeckSource(Protocol):
    """Deals one game's decks, split into TRAIN, TEST and VALIDATION."""

    @property
    def deck_split_key(self) -> str:
        """The holdout's deck_split_key(source_game) this source was built
        under (ContrastiveDojo checks it against the run's holdout)."""
        ...

    @property
    def source_game(self) -> GameId:
        """The game every dealt deck belongs to."""
        ...

    @property
    def card_binder_version(self) -> str | None:
        """The CardBinder version the dealt cards were minted against."""
        ...

    def decks_for(
        self,
        split: Split,
        decks_per_sample: int,
        shuffle: bool = True,
        keep_remainder: bool = False,
    ) -> Iterator[list[GenericDeck]]:
        """Yield split's decks in groups of decks_per_sample (see
        DeckBoxDealer.decks_for for the shared meaning of shuffle and
        keep_remainder). Side effects: a shuffled pass advances the
        source's shuffle state. Exceptions: ValueError if
        decks_per_sample <= 0."""
        ...

    def deck_count(self, split: Split) -> int:
        """How many decks one pass over split deals. Side effects: none."""
        ...
