"""DataConstructor for the DeckCardMaskMetric family - see
src/data_refinement/metrics/generic/deck_card_mask_metric.py."""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors.row_values import (
    deck_cards_excluding,
    parsed_uuid,
)
from src.schema.type_hints import TrainingDatum


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
        Side effects: reads deck_box (one get_by_uuid per row).
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
            # An unparseable target masks nothing (defensive; the
            # metric always writes a valid uuid string here)
            deck_cards = deck_cards_excluding(
                self._deck_box,
                lookup,
                row["deck_uuid"],
                parsed_uuid(row["target_card_uuid"]),
            )
            if not deck_cards:
                continue
            results.append((deck_cards, row[self._label_column]))

        return results
