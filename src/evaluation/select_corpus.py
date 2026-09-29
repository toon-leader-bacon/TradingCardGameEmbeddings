"""Corpus selection: which cards an intrinsic evaluation embeds.

A corpus is every card of some games from a CardLookup, optionally
narrowed to an exact set of holdout tiers (e.g. only VALIDATION cards: the
ones a checkpoint never saw). This is how domain shift is expressed - by
which cards go in, not by a separate evaluation mode. Selection is a pure
generator over existing types; nothing is stored.
"""

from dataclasses import dataclass
from typing import Iterator

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec


@dataclass(frozen=True)
class TierFilter:
    """Keep only cards whose tier under `holdout` is one of `tiers`.

    An exact set, not VisibleCardLookup's nested per-split visibility:
    tiers={VALIDATION} keeps VALIDATION cards only.

    Exceptions: ValueError on construction if tiers is empty.
    """

    holdout: HoldoutSpec
    tiers: frozenset[CardTier]

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("a TierFilter needs at least one tier")

    def admits(self, card: GenericCard) -> bool:
        """Whether card's tier under self.holdout is in self.tiers.

        Inputs: card (GenericCard). Output: bool. Side effects: none.
        Exceptions: none.

        Example:
            >>> TierFilter(spec, frozenset({CardTier.VALIDATION})).admits(card)
            False
        """
        return self.holdout.tier_of(card.nocab_uuid, card.source_game) in self.tiers


@dataclass(frozen=True)
class CorpusSpec:
    """Which cards to embed.

    games: every card of these games (non-empty).
    tier_filter: None keeps every card; otherwise only cards it admits.

    Exceptions: ValueError on construction if games is empty.
    """

    games: frozenset[GameId]
    tier_filter: TierFilter | None

    def __post_init__(self) -> None:
        if not self.games:
            raise ValueError("a CorpusSpec needs at least one game")


def select_corpus(cards: CardLookup, spec: CorpusSpec) -> Iterator[GenericCard]:
    """Yield every card spec selects, lazily.

    Inputs: cards (CardLookup), spec (CorpusSpec).
    Output: Iterator[GenericCard]: games in GameId value order (so the same
        spec over the same binder always yields the same sequence), each
        game's cards in all_cards order, filtered by spec.tier_filter.
    Side effects: reads from cards as iterated.
    Exceptions: whatever cards.all_cards raises (e.g. a game not loaded).

    Example:
        >>> spec = CorpusSpec(frozenset({GameId.GWENT}), tier_filter=None)
        >>> sum(1 for _ in select_corpus(binder, spec))
        1273
    """
    # Walk games in a fixed order, independent of frozenset iteration order
    for game in sorted(spec.games, key=lambda game: game.value):
        for card in cards.all_cards(game):
            # Keep the card unless a tier filter excludes it
            if spec.tier_filter is None or spec.tier_filter.admits(card):
                yield card
