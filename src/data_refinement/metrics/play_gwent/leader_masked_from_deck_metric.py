"""Masking metric: a gwent deck's leader card, masked out and
predicted by IDENTITY (which leader), not by a field of the leader.

See src/data_refinement/metrics/generic/deck_card_mask_metric.py for the
shared accumulate()/finalize() sequence this fixes SOURCE_GAME/
LABEL_VALUES/deck lookup/target-selection/labeling for. Design notes:

  - why the label is the leader's own name rather than e.g. its
    faction (faction is near-trivially recoverable from the rest of a
    Gwent deck's cards, making it an uninteresting prediction target;
    predicting which of 42 leaders it is from deck-composition context
    alone is the actual interesting task - mirrors sts_gg's
    CharacterPredictionMetric precedent).
  - why the target card comes straight from a raw guide row's own
    "leaderId" field (via PlayGwentDeckExtractionStage's own card
    resolution) rather than by scanning a deck's card list for a
    color match - GenericDeck/DeckBox carry no game-specific role
    information once a deck is stored, so this metric never treats an
    already-stored deck as its source of truth for "which card is the
    leader."
  - the published box is read only: each row's deck is looked up by
    the uuid its extraction stage minted (see _deck_uuid_for_row()).
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.play_gwent.extraction_stage import (
    PlayGwentDeckExtractionStage,
)
from src.data_refinement.metrics.generic.deck_card_mask_metric import DeckCardMaskMetric
from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.play_gwent.leader_labels import LEADER_NAMES
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class LeaderMaskedFromDeckMetric(DeckCardMaskMetric):
    """Deck -> its leader card, masked. Every real gwent guide's deck
    has exactly one leader, addressed directly via the guide's own
    "leaderId" field - a row with none/unresolvable is a data-quality
    edge case and is skipped (see _target_card_uuid_for_row())."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    LABEL_VALUES: ClassVar[tuple[str, ...]] = LEADER_NAMES
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/play_gwent/leader_masked_from_deck.parquet"
    )

    def _deck_uuid_for_row(self, row: dict, deck_box: DeckBox) -> UUID | None:
        """The published deck uuid for row's guide; writes nothing.

        The deck already sits in the published Gwent box
        (data/final/decks/gwent.db), under the uuid
        PlayGwentDeckExtractionStage minted from the guide id, so this
        metric never writes decks and the box is never re-saved by a
        metric run (a re-save would make every metric keyed to the box
        stale).

        Inputs:
            row: one raw play_gwent guide object, carrying at least
                "id" (int).
            deck_box: the published Gwent box passed to __init__.
        Output: deck_uuid_for_guide(row["id"]) if deck_box holds that
            deck, else None (row skipped: a guide newer than the box).
        Side effects: none.
        Exceptions: KeyError if row has no "id".
        """
        deck_uuid = PlayGwentDeckExtractionStage.deck_uuid_for_guide(row["id"])
        if deck_box.get_by_uuid(deck_uuid) is None:
            return None
        return deck_uuid

    def _target_card_uuid_for_row(
        self, row: dict, card_lookup: CardLookup
    ) -> UUID | None:
        """Look up row's leader card directly via its own "leaderId".

        Inputs:
            row: one raw play_gwent guide object, carrying "leaderId"
                (int, gwent.one's own card template id for this
                guide's leader).
            card_lookup: used to look up "leaderId" against
                gwent.one's registered aliases (same alias namespace
                PlayGwentDeckExtractionStage's own card resolution
                uses - DataSource.GWENT_ONE).
        Output: the leader card's nocab_uuid, or None if row has no
            "leaderId" or it doesn't resolve to a known card - a real,
            expected data-quality edge case (see this class's
            docstring), not a bug.
        Side effects: emits one logging.error() call when "leaderId"
            is missing or unresolved, before returning None.
        Exceptions: none.
        """
        leader_id = row.get("leaderId")
        if leader_id is None:
            _logger.error(
                "LeaderMaskedFromDeckMetric: guide %r is missing leaderId - "
                "skipping",
                row.get("id"),
            )
            return None

        card = card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.GWENT_ONE, str(leader_id)
        )
        if card is None:
            _logger.error(
                "LeaderMaskedFromDeckMetric: guide %r's leaderId %r did not "
                "resolve to a known card - skipping",
                row.get("id"),
                leader_id,
            )
            return None
        return card.nocab_uuid

    def _label_for_card(self, card: GenericCard) -> str:
        """The leader's own identity (its name), not a field of it.

        Inputs:
            card: the leader card _target_card_uuid_for_row() picked.
        Output: card.raw_content["name"] if it's a member of
            LEADER_NAMES, else OTHER_LABEL
            (masked_field_metric.OTHER_LABEL) - a name outside
            LEADER_NAMES is a real, expected failure mode as new
            leaders get added upstream (see leader_labels.py's
            FRESHNESS caveat), not a bug to raise on. Mirrors
            MaskedFieldMetric's numeric-field subclasses'
            _label_or_other() convention.
        Side effects: emits one logging.error() call on an
            out-of-vocabulary name, before falling back to
            OTHER_LABEL.
        Exceptions: none.
        """
        name = card.raw_content["name"]
        if name not in self.LABEL_VALUES:
            _logger.error(
                "LeaderMaskedFromDeckMetric: leader card %r has name %r, "
                "which is not in LEADER_NAMES - falling back to %s "
                "(leader_labels.py's frozen list may need regenerating)",
                card.nocab_uuid,
                name,
                OTHER_LABEL,
            )
            return OTHER_LABEL
        return name
