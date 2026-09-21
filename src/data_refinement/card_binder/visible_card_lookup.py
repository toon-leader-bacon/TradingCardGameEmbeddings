"""A CardLookup that hides cards outside a split's visible holdout tiers.

Wraps any CardLookup (Decorator) so a dojo building TRAIN-split examples
literally cannot see TEST or VALIDATION cards: every read method treats
such a card as a miss, the same way it treats an unknown card. This is a
read-side filter, not the runtime write-protection wrapper CardLookup's
own docstring rejects.
"""

from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split


class VisibleCardLookup:
    """CardLookup restricted to the tiers visible from one Split.

    Inputs (constructor): inner (the full CardLookup), spec (HoldoutSpec
        assigning tiers), view (the Split whose visibility applies).
    Output: n/a.
    Side effects: none; reads delegate to inner.
    Exceptions: none of its own; inner's exceptions propagate (e.g.
        get_by_name_single(strict=True) on an ambiguous name).

    Example:
        >>> train_view = VisibleCardLookup(binder, spec, Split.TRAIN)
        >>> train_view.get_by_uuid(a_test_tier_uuid) is None
        True
    """

    def __init__(self, inner: CardLookup, spec: HoldoutSpec, view: Split) -> None:
        self._inner = inner
        self._spec = spec
        self._visible = spec.visible_tiers(view)

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericCard | None:
        """See CardLookup.get_by_uuid(); None if the card is hidden."""
        return self._visible_or_none(self._inner.get_by_uuid(nocab_uuid))

    def get_by_name(self, source_game: GameId, name: str) -> list[GenericCard]:
        """See CardLookup.get_by_name(); hidden cards are omitted."""
        return self._only_visible(self._inner.get_by_name(source_game, name))

    def get_by_name_single(
        self, source_game: GameId, name: str, strict: bool = True
    ) -> GenericCard | None:
        """See CardLookup.get_by_name_single(); None if the card is hidden."""
        return self._visible_or_none(
            self._inner.get_by_name_single(source_game, name, strict)
        )

    def get_by_alias(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> GenericCard | None:
        """See CardLookup.get_by_alias(); None if the card is hidden."""
        return self._visible_or_none(
            self._inner.get_by_alias(source_game, data_source, source_id)
        )

    def get_by_name_regex(self, source_game: GameId, pattern: str) -> list[GenericCard]:
        """See CardLookup.get_by_name_regex(); hidden cards are omitted."""
        return self._only_visible(self._inner.get_by_name_regex(source_game, pattern))

    def all_uuids(self, source_game: GameId | None = None) -> Iterable[UUID]:
        """See CardLookup.all_uuids(); hidden cards are omitted."""
        for nocab_uuid in self._inner.all_uuids(source_game):
            card = self.get_by_uuid(nocab_uuid)
            if card is not None:
                yield nocab_uuid

    def all_cards(self, source_game: GameId) -> Iterable[GenericCard]:
        """See CardLookup.all_cards(); hidden cards are omitted."""
        return self._only_visible(self._inner.all_cards(source_game))

    def _is_visible(self, card: GenericCard) -> bool:
        tier = self._spec.tier_of(card.nocab_uuid, card.source_game)
        return tier in self._visible

    def _visible_or_none(self, card: GenericCard | None) -> GenericCard | None:
        if card is not None and self._is_visible(card):
            return card
        return None

    def _only_visible(self, cards: Iterable[GenericCard]) -> list[GenericCard]:
        return [card for card in cards if self._is_visible(card)]
