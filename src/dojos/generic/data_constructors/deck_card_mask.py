"""DataConstructor for the DeckCardMaskMetric family - see
src/data_refinement/metrics/generic/deck_card_mask_metric.py."""

from typing import List
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.type_hints import MultiCardInput, TrainingDatum


class DeckCardMaskDataConstructor:
    """DataConstructor for the DeckCardMaskMetric family (see
    src/data_refinement/metrics/generic/deck_card_mask_metric.py) - whole deck
    in, minus one masked-out target card, single raw string label out.
    Every concrete DeckCardMaskMetric subclass (e.g.
    LeaderMaskedFromDeckMetric) shares the SAME fixed output schema
    (deck_uuid, target_card_uuid, label) - unlike
    DeckLabelDataConstructor/CardAverageDataConstructor, the column
    name is not expected to actually vary per subclass today
    (DeckCardMaskMetric.__init__ fixes the schema itself, not a
    per-subclass ClassVar). label_column is still accepted as a
    constructor argument, purely to keep every DataConstructor in this
    package to the same interface shape - a wrapper today always passes
    label_column="label" literally; this isn't reconsidered as
    per-subclass-configurable until a metric in this family actually
    needs a different column name.

    Returns the RAW label string as read off the row, unencoded against
    any vocabulary - matches MaskedFieldDataConstructor's convention (a
    metric's LABEL_VALUES-based encoding is FixedClassificationLoss's
    job, not this class's - see plans/dojo_v2.md's "Fixed-classification
    label encoding lives in the generic dojo, not the DataConstructor").
    """

    def __init__(self, deck_box: DeckBox, label_column: str) -> None:
        """
        Inputs:
            deck_box: registry to look up each row's deck_uuid against.
                Never written to.
            label_column: the column name holding this metric's label -
                "label" for every DeckCardMaskMetric subclass today (see
                this class's own docstring for why it's still a
                constructor argument rather than hardcoded).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._deck_box = deck_box
        self._label_column = label_column

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (deck_uuid, target_card_uuid,
        <label_column>) rows into (MultiCardInput, str) TrainingDatum
        pairs, with each row's target card masked out of its deck's
        card list.

        Inputs:
            chunk: one chunk of rows from a DeckCardMaskMetric
                subclass's output parquet file.
        Output: one (List[GenericCard], str) TrainingDatum per row
            whose deck_uuid resolves to a deck with at least one
            resolvable non-target card remaining. The label is passed
            through unchanged (DeckCardMaskMetric.accumulate() always
            writes a genuine str, already a member of the paired
            metric's LABEL_VALUES or its OTHER fallback). A single
            non-target card within a resolved deck that fails to
            resolve is dropped from that deck's card list, not treated
            as a reason to skip the row (mirrors
            DeckLabelDataConstructor.build()'s convention). A row is
            skipped outright if its deck_uuid doesn't resolve, or if
            every one of that deck's non-target cards fails to resolve.
        Side effects: none.
        Exceptions: none expected (per-row/per-card failures are
            skipped, not raised - mirrors DeckLabelDataConstructor.build()).

        Example:
            >>> constructor = DeckCardMaskDataConstructor(deck_box, "label")
            >>> constructor.build(chunk, lookup)
            [([<GenericCard>, <GenericCard>], "Geralt"), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's deck, minus its target card, independently;
        # skip rows that fail deck resolution or end up with no cards
        # left at all, same tolerance-of-individual-unresolved-cards
        # convention as DeckLabelDataConstructor.build().
        for _, row in chunk.iterrows():
            deck_cards = self._deck_cards_excluding_target(
                lookup, row["deck_uuid"], row["target_card_uuid"]
            )
            if not deck_cards:
                continue
            results.append((deck_cards, row[self._label_column]))

        return results

    def _deck_cards_excluding_target(
        self, lookup: CardLookup, raw_deck_uuid: object, raw_target_uuid: object
    ) -> MultiCardInput:
        """Look up one row's deck, with its target card masked out.

        Private helper - single consumer is build().

        Inputs:
            lookup: the split's holdout-filtered card lookup.
            raw_deck_uuid: a row's "deck_uuid" cell, expected to be a
                str parseable as a UUID.
            raw_target_uuid: a row's "target_card_uuid" cell, expected
                to be a str parseable as a UUID - excluded from the
                deck's card_nocab_uuids BEFORE card resolution, per
                plans/dojo_v2.md's masking note (filter first, resolve
                second - no separate masking Mod needed for the
                multi-card case, unlike MaskTargetKeyMod's single-card
                approach).
        Output: the resolved deck's cards minus the target card, as a
            MultiCardInput - omitting any remaining card_nocab_uuid
            that doesn't resolve against lookup. Empty (not
            None) if raw_deck_uuid doesn't parse, self._deck_box has no
            deck for it, or every remaining card fails to resolve - an
            empty result is build()'s own signal to skip the row.
        Side effects: none.
        Exceptions: none - all failures collapse to an empty list.
        """
        try:
            deck_uuid = UUID(str(raw_deck_uuid))
        except (TypeError, ValueError):
            return []
        deck = self._deck_box.get_by_uuid(deck_uuid)
        if deck is None:
            return []

        try:
            target_uuid: UUID | None = UUID(str(raw_target_uuid))
        except (TypeError, ValueError):
            # An unparseable target doesn't invalidate an otherwise-
            # resolvable deck - it just means nothing gets filtered out
            # below (defensive; DeckCardMaskMetric's own schema always
            # writes a valid uuid string here).
            target_uuid = None

        deck_cards: MultiCardInput = []
        for card_uuid in deck.card_nocab_uuids:
            if card_uuid == target_uuid:
                continue
            card = lookup.get_by_uuid(card_uuid)
            if card is None:
                continue
            deck_cards.append(card)
        return deck_cards
