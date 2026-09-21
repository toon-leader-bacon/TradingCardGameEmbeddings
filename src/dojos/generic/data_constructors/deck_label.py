"""DataConstructor for the DeckLabelMetric family - see
src/data_refinement/metrics/sts_gg/deck_label_metric.py."""

import math
from typing import Callable, List
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors._uuid_resolution import _cast_to_float
from src.schema.type_hints import Label, MultiCardInput, TrainingDatum


class DeckLabelDataConstructor:
    """DataConstructor for the DeckLabelMetric family (see
    src/data_refinement/metrics/sts_gg/deck_label_metric.py) - whole
    deck in, single int/bool/str label out. Every concrete
    DeckLabelMetric subclass (deck_label_metrics.py) shares this same
    row shape (run_id, deck_uuid, LABEL_COLUMN), differing only in
    which column holds the label and that column's type - so, like
    CardAverageDataConstructor, label_column is a constructor argument
    here, letting one class serve every subclass. Consumed by
    MultiCardRegressionDojo (int64/bool-labeled subclasses, via the
    default float caster) and MultiCardFixedClassificationDojo
    (str-labeled subclasses, via label_caster=str) - see label_caster
    below for why casting is configurable rather than a hardcoded
    float() call.
    """

    def __init__(
        self,
        deck_box: DeckBox,
        label_column: str,
        label_caster: Callable[[object], Label] = _cast_to_float,
    ) -> None:
        """
        Inputs:
            deck_box: registry to look up each row's deck_uuid against.
                Never written to.
            label_column: the column name holding this metric's label
                (e.g. "relic_count", "total_damage_taken") - read off
                the paired DeckLabelMetric subclass's own LABEL_COLUMN
                ClassVar by the thin wrapper that constructs this,
                never duplicated as a literal (see plans/dojo_v2.md).
            label_caster: converts each row's raw label_column cell to
                the Label type the consuming dojo's loss expects.
                Defaults to float, matching every DeckLabelMetric
                subclass consumed by MultiCardRegressionDojo (int64 and
                WinMetric's bool column both cast fine). A thin wrapper
                for a str-labeled subclass (e.g. CharacterDojo, backed
                by MultiCardFixedClassificationDojo/FixedClassificationLoss)
                passes label_caster=str instead - the wrapper already
                knows which generic dojo cell it targets, so the choice
                belongs there, not as a branch inside this class.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._deck_box = deck_box
        self._label_column = label_column
        self._label_caster = label_caster

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (run_id, deck_uuid, <label_column>) rows
        into (MultiCardInput, float) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from a DeckLabelMetric subclass's
                output parquet file.
        Output: one (List[GenericCard], Label) TrainingDatum per row
            whose deck_uuid resolves to a deck with at least one
            resolvable card. The label is passed through
            self._label_caster - see __init__'s label_caster docstring
            for why this isn't a hardcoded float() call. A single card
            within a resolved deck that fails to resolve is dropped
            from that deck's card list, not treated as a reason to skip
            the row (human decision - training data is inherently
            messy; see plans/dojo_v2.md). A row is skipped outright if
            its deck_uuid doesn't resolve, or if every one of that
            deck's cards fails to resolve (an empty MultiCardInput is
            never a valid training datum).
        Side effects: none.
        Exceptions: none expected (per-row/per-card failures are
            skipped, not raised - mirrors CardAverageDataConstructor.build()).

        Example:
            >>> constructor = DeckLabelDataConstructor(deck_box, "relic_count")
            >>> constructor.build(chunk, lookup)
            [([<GenericCard>, <GenericCard>], 3.0), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's deck, then that deck's cards, independently;
        # skip rows that fail deck resolution or end up with no cards at
        # all, but tolerate individual unresolved cards within an
        # otherwise-resolvable deck.
        for _, row in chunk.iterrows():
            raw_label = row[self._label_column]
            if isinstance(raw_label, float) and math.isnan(raw_label):
                # A metric's own nullable-output convention (e.g.
                # OnPlayWinRateSensitivityByDeckMetric's None for a deck
                # never seen on one side) round-trips through a float64
                # parquet column as NaN, not None - skip it the same way
                # an unresolvable deck is skipped, rather than casting a
                # NaN into a training label (see DataConstructor
                # Protocol's "resolution failure is expected" contract).
                continue
            deck_cards = self._deck_cards_for_uuid(lookup, row["deck_uuid"])
            if not deck_cards:
                continue
            results.append((deck_cards, self._label_caster(raw_label)))

        return results

    def _deck_cards_for_uuid(
        self, lookup: CardLookup, raw_deck_uuid: object
    ) -> MultiCardInput:
        """Look up and build the card list for one row's raw deck_uuid value.

        Private helper - single consumer is build().

        Inputs:
            lookup: the split's holdout-filtered card lookup.
            raw_deck_uuid: a row's "deck_uuid" cell, expected to be a
                str parseable as a UUID.
        Output: the resolved deck's cards, as a MultiCardInput -
            omitting any of the deck's card_nocab_uuids that don't
            resolve against lookup. Empty (not None) if
            raw_deck_uuid doesn't parse, self._deck_box has no deck for
            it, or every one of the deck's cards fails to resolve - an
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

        deck_cards: MultiCardInput = []
        for card_uuid in deck.card_nocab_uuids:
            card = lookup.get_by_uuid(card_uuid)
            if card is None:
                continue
            deck_cards.append(card)
        return deck_cards
