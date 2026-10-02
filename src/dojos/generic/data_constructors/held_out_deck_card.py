"""DataConstructor for the HeldOutDeckCardMetric family - see
src/data_refinement/metrics/generic/held_out_deck_card/metric.py."""

from typing import List, cast
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors.row_values import (
    card_for_uuid,
    deck_cards_excluding,
    parsed_uuid,
)
from src.schema.card import GenericCard
from src.schema.type_hints import MultiGroupInput, TrainingDatum


class HeldOutDeckCardDataConstructor:
    """(deck_uuid, target_card_uuid, candidate_uuids) rows ->
    ([candidate_cards, deck_minus_target_cards], target's index).

    Feeds MultiGroupOptionSelectionDojo: candidates MUST stay group 0
    (never empty; input_shape_of() peeks group 0, and the cell's
    baseline counts options as len(input[0])), the deck context group 1.

    Holdout (lookup is the split's VisibleCardLookup, a hidden card
    reads as None):
    - target hidden or unparseable -> the row is skipped, so a TEST-tier
      target never reaches a TRAIN row;
    - a hidden or unparseable decoy is dropped and the label index is
      taken after dropping; no decoy left -> skipped;
    - every copy of the target is removed from the deck context, hidden
      context cards are dropped; empty context -> skipped.
    """

    def __init__(self, deck_box: DeckBox) -> None:
        """
        Inputs: deck_box, the published box the rows' deck_uuids point
            into; only read.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self._deck_box = deck_box

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into training data.

        Inputs:
            chunk: rows with deck_uuid (str), target_card_uuid (str),
                candidate_uuids (list[str]).
            lookup: the split's holdout-filtered card lookup.
        Output: one ([candidates, deck_context], int) TrainingDatum per
            row that survives the rules in the class docstring.
        Side effects: reads deck_box (one get_by_uuid per row).
        Exceptions: none (per-row failures are skipped, as in every
            DataConstructor in this package).

        Example:
            >>> HeldOutDeckCardDataConstructor(box).build(chunk, lookup)
            [([[<GenericCard>, ...], [<GenericCard>, ...]], 3), ...]
        """
        result: List[TrainingDatum] = []

        # Each row: candidates first (they carry the label), then context
        for _, row in chunk.iterrows():
            target_uuid = parsed_uuid(row["target_card_uuid"])
            if target_uuid is None:
                continue
            options = _visible_candidates(lookup, row["candidate_uuids"], target_uuid)
            if options is None:
                continue
            candidate_cards, target_index = options
            context = deck_cards_excluding(
                self._deck_box, lookup, row["deck_uuid"], target_uuid
            )
            if not context:
                continue
            group: MultiGroupInput = [candidate_cards, context]
            result.append((group, target_index))
        return result


def _visible_candidates(
    lookup: CardLookup, raw_candidate_uuids: object, target_uuid: UUID
) -> tuple[list[GenericCard], int] | None:
    """The row's visible candidate cards and the target's index among
    them.

    Inputs: lookup, raw_candidate_uuids (a row's list[str] cell),
        target_uuid.
    Output: (cards in row order with hidden/unparseable decoys dropped,
        target's index), or None if the target is hidden or absent from
        the candidates, or no decoy survives.
    Side effects: none. Exceptions: none.

    Deliberately NOT option_cards_and_pick_index() (row_values.py),
    which skips the whole row when any option is missing: there the
    label is a position fixed by the raw pack, here the index is
    computed after dropping, so a hidden decoy only shrinks the
    candidate set. Skipping instead would keep only ~0.9^(K+1) of rows
    under a 10% holdout.
    """
    cards: list[GenericCard] = []
    target_index: int | None = None
    for raw_uuid in cast(List[str], raw_candidate_uuids):
        card = card_for_uuid(lookup, raw_uuid)
        if card is None:
            continue
        if parsed_uuid(raw_uuid) == target_uuid:
            target_index = len(cards)
        cards.append(card)

    if target_index is None or len(cards) < 2:
        return None
    return cards, target_index
