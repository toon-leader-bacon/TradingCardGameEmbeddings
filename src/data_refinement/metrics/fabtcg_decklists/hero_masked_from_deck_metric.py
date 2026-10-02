"""Masking metric: a Flesh and Blood decklist's hero, masked out and
predicted by IDENTITY (which hero), the FaB twin of
../play_gwent/leader_masked_from_deck_metric.py.

Built on ../generic/deck_card_mask_metric.py's DeckCardMaskMetric. Two
differences from the Gwent leader metric, both because the published
deck box is read-only:

  - _deck_uuid_for_row() writes nothing. The row's deck already sits in
    the published FaB box (data/final/decks/flesh_and_blood.db) under
    the uuid its extraction stage minted; PublishedDecklists finds it by
    slug. A slug the box lacks has no deck (None, row skipped).
  - The target is read off the published deck's cards, not off the raw
    row: a FaB hero is card-intrinsic (its typebox names Hero), so the
    box has not lost "which card is the hero" the way it loses Gwent's
    leader slot. See published_decklists.py.

The label is the hero card's name, one of hero_labels.HERO_NAMES (heroes
with 20+ decks), else OTHER_LABEL.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.hero_labels import HERO_NAMES
from src.data_refinement.metrics.fabtcg_decklists.published_decklists import (
    PublishedDecklists,
)
from src.data_refinement.metrics.generic.deck_card_mask_metric import DeckCardMaskMetric
from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.schema.card import GenericCard
from src.schema.game_id import GameId


class HeroMaskedFromDeckMetric(DeckCardMaskMetric):
    """Published FaB deck (hero masked) -> the hero's name.

    Rows are raw decklist rows {"slug": str} (see scanner.py); one
    output row per slug whose published deck has exactly one hero.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (*HERO_NAMES, OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/fabtcg_decklists/hero_masked_from_deck.parquet"
    )

    def __init__(
        self,
        card_lookup: CardLookup,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """See DeckCardMaskMetric.__init__, except deck_box is the
        published FaB box and is only read (indexed once here by
        PublishedDecklists).

        Side effects: also reads every FaB deck in deck_box once.

        Example:
            >>> box = DeckBox.load([DeckBox.default_output_path(GameId.FLESH_AND_BLOOD)])
            >>> HeroMaskedFromDeckMetric(binder, box)
        """
        super().__init__(card_lookup, deck_box, output_path)
        self._decklists = PublishedDecklists(deck_box, card_lookup)

    def _deck_uuid_for_row(self, row: dict, deck_box: DeckBox) -> UUID | None:
        """The published deck uuid for row's slug; writes nothing.

        Inputs: row ({"slug": str}), deck_box (unused: indexed in
            __init__).
        Output: the deck's uuid, or None if the box has no single-hero
            deck for this slug.
        Side effects: none.
        Exceptions: KeyError if row has no "slug".
        """
        decklist = self._decklists.decklist_for_slug(row["slug"])
        return None if decklist is None else decklist.deck_uuid

    def _target_card_uuid_for_row(
        self, row: dict, card_lookup: CardLookup
    ) -> UUID | None:
        """The hero of row's published deck.

        Inputs: row ({"slug": str}), card_lookup (unused).
        Output: the hero's nocab_uuid, or None as _deck_uuid_for_row.
        Side effects: none.
        Exceptions: KeyError if row has no "slug".
        """
        decklist = self._decklists.decklist_for_slug(row["slug"])
        return None if decklist is None else decklist.hero.nocab_uuid

    def _label_for_card(self, card: GenericCard) -> str:
        """The hero's name if it is in HERO_NAMES, else OTHER_LABEL (the
        expected outcome for the 77 rarely played heroes, not an error).

        Inputs: card (the hero). Output: str.
        Side effects: none. Exceptions: none.
        """
        return card.name if card.name in HERO_NAMES else OTHER_LABEL
