"""GroupPickDataConstructor - an option CardGroup, an optional context
CardGroup and a list of picked card uuids -> one option-selection datum
per pick.

Serves the isotropic "which card(s)" metrics, whose label is a list of
1-4 card uuids rather than one index: kingdom opening buy, next buy,
next trashed card, kingdom veto. Each picked card becomes its own datum
with the same input, so a two-card pick trains the softmax toward both
cards. All of a row's data come from that row, so they share its split.

Feeds MultiCardOptionSelectionDojo (no context: the input is the option
list) or MultiGroupOptionSelectionDojo (input [options, context];
options must stay group 0).
"""

from typing import List, cast
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors.row_values import parsed_uuid
from src.dojos.isotropic.card_groups import CardGroup
from src.schema.card import GenericCard
from src.schema.type_hints import MultiGroupInput, TrainingDatum, TrainingInput


class GroupPickDataConstructor:
    """DataConstructor: one (input, picked option's index) per pick.

    Per pick: a uuid that does not parse, or is not among the visible
    options (a hidden card, a Black Market buy outside the supply, a
    trashed card the rebuilt partial deck lacks), is skipped. A row with
    fewer than 2 visible options is skipped (nothing to choose).
    Duplicate options keep their first copy only, so each card has one
    softmax slot. A card picked twice in one row (e.g. a Silver/Silver
    opening) yields two identical data: that row counts twice toward it.
    """

    def __init__(
        self,
        options: CardGroup,
        picks_column: str,
        context: CardGroup | None = None,
    ) -> None:
        """
        Inputs:
            options: the row's choosable cards (deduplicated here).
            picks_column: the row's list[str] column of picked uuids.
            context: the conditioning group (group 1), or None for an
                options-only input.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self._options = options
        self._picks_column = picks_column
        self._context = context

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into option-selection data.

        Inputs: chunk (metric rows), lookup (the split's holdout-filtered
            lookup).
        Output: one (input, index) per surviving pick; input is the
            option list, or [options, context] when context is set.
        Side effects: reads the groups' DeckBox. Exceptions: none.

        Example:
            >>> GroupPickDataConstructor(
            ...     DeckColumnGroup(box, "candidate_pool_uuid"),
            ...     "vetoed_card_uuids",
            ... ).build(chunk, lookup)
            [([<GenericCard>, ...], 3), ...]
        """
        result: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            options = _first_copies(self._options.cards(row, lookup))
            if len(options) < 2:
                continue
            row_input = self._row_input(row, lookup, options)

            # One datum per pick that is among the options
            option_index = {card.nocab_uuid: i for i, card in enumerate(options)}
            for pick_uuid in _parsed_picks(row[self._picks_column]):
                if pick_uuid in option_index:
                    result.append((row_input, option_index[pick_uuid]))
        return result

    def _row_input(
        self, row: pd.Series, lookup: CardLookup, options: List[GenericCard]
    ) -> TrainingInput:
        """options alone, or [options, context] when a context is set.

        Inputs: row, lookup, options (the row's visible options).
        Output: TrainingInput. Side effects: may read a DeckBox.
        Exceptions: none.
        """
        if self._context is None:
            return options
        group: MultiGroupInput = [options, self._context.cards(row, lookup)]
        return group


def _first_copies(cards: List[GenericCard]) -> List[GenericCard]:
    """cards with each nocab_uuid kept at its first position only.

    Inputs: cards. Output: a new List[GenericCard].
    Side effects: none. Exceptions: none.
    """
    seen: set[UUID] = set()
    result: List[GenericCard] = []
    for card in cards:
        if card.nocab_uuid not in seen:
            seen.add(card.nocab_uuid)
            result.append(card)
    return result


def _parsed_picks(raw_picks: object) -> List[UUID]:
    """A row's picked-uuid list cell as UUIDs, unparseable entries dropped.

    Inputs: raw_picks (a list[str]-like cell). Output: List[UUID].
    Side effects: none. Exceptions: none.
    """
    parsed = (parsed_uuid(raw) for raw in cast(List[str], raw_picks))
    return [pick for pick in parsed if pick is not None]
