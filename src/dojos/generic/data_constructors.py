"""One DataConstructor (see data_constructor.py) per metric FAMILY - the
Template Method bases under src/data_refinement/metrics/. Grouped in one
file per plans/dojo_v2.md's directory layout, since each is a thin,
single-purpose class; more are added here as later dojo_v2 slices consume
more metric families (MaskedFieldMetric, DeckCardMaskMetric, DeckLabelMetric).
"""

from typing import Callable, Dict, List
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericCard
from src.schema.type_hints import Label, MultiCardInput, TrainingDatum


def _cast_to_float(raw_label: object) -> float:
    """DeckLabelDataConstructor's default label_caster - float() typed
    against object rather than its builtin overloads, so it satisfies
    Callable[[object], Label]."""
    return float(raw_label)  # type: ignore[arg-type]


class CardAverageDataConstructor:
    """DataConstructor for the CardAverageMetric family (see
    src/data_refinement/metrics/sts_gg/card_average_metric.py) - single
    card in, single float label out. Every concrete CardAverageMetric
    subclass (card_average_metrics.py) shares this same row shape
    (nocab_uuid, LABEL_COLUMN, sample_count), differing only in which
    column holds the label - so, unlike AveragePickNumberDataConstructor
    (v1, hardcoded to "average_pick_number"), label_column is a
    constructor argument here, letting one class serve all 9 subclasses.
    """

    def __init__(self, card_binder: CardBinder, label_column: str) -> None:
        """
        Inputs:
            card_binder: registry to resolve each row's nocab_uuid
                against. Never written to.
            label_column: the column name holding this metric's label
                (e.g. "average_relic_count", "win_rate") - read off the
                paired CardAverageMetric subclass's own LABEL_COLUMN
                ClassVar by the thin wrapper that constructs this, never
                duplicated as a literal (see plans/dojo_v2.md).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._label_column = label_column

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, <label_column>) rows into
        (SingleCardInput, Label) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from a CardAverageMetric subclass's
                output parquet file.
        Output: one (GenericCard, float) TrainingDatum per row whose
            nocab_uuid resolves to a card and whose label parses as a
            float. Rows that fail either check are skipped.
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors AveragePickNumberDataConstructor.build()).
        """
        results: List[TrainingDatum] = []

        # Resolve each row's card and label independently; skip rows that
        # fail either resolution rather than raising, since a metric's
        # output may contain the odd unresolvable card id.
        for _, row in chunk.iterrows():
            card = self._card_for_uuid(row["nocab_uuid"])
            if card is None:
                continue
            label = self._label_as_float(row[self._label_column])
            if label is None:
                continue
            results.append((card, label))

        return results

    def _card_for_uuid(self, raw_nocab_uuid: object) -> GenericCard | None:
        """Look up the card for one row's raw nocab_uuid value.

        Private helper - single consumer is build().

        Inputs:
            raw_nocab_uuid: a row's "nocab_uuid" cell, expected to be a
                str parseable as a UUID.
        Output: the matching GenericCard, or None if raw_nocab_uuid
            doesn't parse as a UUID or self._card_binder has no card for
            it.
        Side effects: none.
        Exceptions: none - all failures collapse to None.
        """
        try:
            card_uuid = UUID(str(raw_nocab_uuid))
        except (TypeError, ValueError):
            return None
        return self._card_binder.get_by_uuid(card_uuid)

    def _label_as_float(self, raw_label: object) -> float | None:
        """Parse one row's raw label cell as a float.

        Private helper - single consumer is build().

        Inputs:
            raw_label: a row's label-column cell (float, int, or bool -
                see CardAverageMetric's WIN RATE IS AN AVERAGE note for
                why a bool is a legitimate input here).
        Output: raw_label as a float, or None if it doesn't convert.
        Side effects: none.
        Exceptions: none - all failures collapse to None.
        """
        if isinstance(raw_label, (float, int, bool)):
            return float(raw_label)
        return None


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
        card_binder: CardBinder,
        deck_box: DeckBox,
        label_column: str,
        label_caster: Callable[[object], Label] = _cast_to_float,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve each deck's card
                nocab_uuids against. Never written to.
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
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._label_column = label_column
        self._label_caster = label_caster

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
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
            >>> constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")
            >>> constructor.build(chunk)
            [([<GenericCard>, <GenericCard>], 3.0), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's deck, then that deck's cards, independently;
        # skip rows that fail deck resolution or end up with no cards at
        # all, but tolerate individual unresolved cards within an
        # otherwise-resolvable deck.
        for _, row in chunk.iterrows():
            deck_cards = self._deck_cards_for_uuid(row["deck_uuid"])
            if not deck_cards:
                continue
            results.append((deck_cards, self._label_caster(row[self._label_column])))

        return results

    def _deck_cards_for_uuid(self, raw_deck_uuid: object) -> MultiCardInput:
        """Look up and build the card list for one row's raw deck_uuid value.

        Private helper - single consumer is build().

        Inputs:
            raw_deck_uuid: a row's "deck_uuid" cell, expected to be a
                str parseable as a UUID.
        Output: the resolved deck's cards, as a MultiCardInput -
            omitting any of the deck's card_nocab_uuids that don't
            resolve against self._card_binder. Empty (not None) if
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
            card = self._card_binder.get_by_uuid(card_uuid)
            if card is None:
                continue
            deck_cards.append(card)
        return deck_cards


class DeckCardMaskDataConstructor:
    """DataConstructor for the DeckCardMaskMetric family (see
    src/data_refinement/metrics/deck_card_mask_metric.py) - whole deck
    in, minus one masked-out target card, single raw string label out.
    Every concrete DeckCardMaskMetric subclass (e.g.
    LeaderMaskedFromDeckMetric) shares the SAME fixed output schema
    (deck_uuid, target_card_uuid, label) - unlike
    DeckLabelDataConstructor/CardAverageDataConstructor, the column
    name is not expected to actually vary per subclass today
    (DeckCardMaskMetric.__init__ fixes the schema itself, not a
    per-subclass ClassVar). label_column is still accepted as a
    constructor argument, purely to keep every DataConstructor in this
    module to the same interface shape - a wrapper today always passes
    label_column="label" literally; this isn't reconsidered as
    per-subclass-configurable until a metric in this family actually
    needs a different column name.

    Returns the RAW label string as read off the row, unencoded against
    any vocabulary - matches MaskedFieldDataConstructor's convention (a
    metric's LABEL_VALUES-based encoding is FixedClassificationLoss's
    job, not this class's - see plans/dojo_v2.md's "Fixed-classification
    label encoding lives in the generic dojo, not the DataConstructor").
    """

    def __init__(
        self, card_binder: CardBinder, deck_box: DeckBox, label_column: str
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve each deck's remaining
                (non-target) card nocab_uuids against. Never written to.
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
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._label_column = label_column

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
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
            >>> constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")
            >>> constructor.build(chunk)
            [([<GenericCard>, <GenericCard>], "Geralt"), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's deck, minus its target card, independently;
        # skip rows that fail deck resolution or end up with no cards
        # left at all, same tolerance-of-individual-unresolved-cards
        # convention as DeckLabelDataConstructor.build().
        for _, row in chunk.iterrows():
            deck_cards = self._deck_cards_excluding_target(
                row["deck_uuid"], row["target_card_uuid"]
            )
            if not deck_cards:
                continue
            results.append((deck_cards, row[self._label_column]))

        return results

    def _deck_cards_excluding_target(
        self, raw_deck_uuid: object, raw_target_uuid: object
    ) -> MultiCardInput:
        """Look up one row's deck, with its target card masked out.

        Private helper - single consumer is build().

        Inputs:
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
            that doesn't resolve against self._card_binder. Empty (not
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
            card = self._card_binder.get_by_uuid(card_uuid)
            if card is None:
                continue
            deck_cards.append(card)
        return deck_cards


class MaskedFieldDataConstructor:
    """DataConstructor for the MaskedFieldMetric family (see
    src/data_refinement/metrics/masked_field_metric.py) - single card
    in, single raw string label out. Every concrete MaskedFieldMetric
    subclass (e.g. gwent_one's 8) shares the same fixed row shape
    (nocab_uuid, masked_field, label) - unlike CardAverageDataConstructor,
    the label column name isn't expected to actually vary across this
    family today. label_column is still accepted as a constructor
    argument, purely to keep every DataConstructor in this module to
    the same interface shape - a wrapper today always passes
    label_column="label" literally; this isn't reconsidered as
    per-subclass-configurable until a metric in this family actually
    needs a different column name.

    Returns the RAW label string as read off the row, unencoded against
    any vocabulary - a metric's LABEL_VALUES-based encoding is
    FixedClassificationLoss's job, not this class's (see
    plans/dojo_v2.md's "Fixed-classification label encoding lives in
    the generic dojo, not the DataConstructor").
    """

    def __init__(self, card_binder: CardBinder, label_column: str) -> None:
        """
        Inputs:
            card_binder: registry to look up each row's nocab_uuid
                against. Never written to.
            label_column: the column name holding this metric's label -
                "label" for every MaskedFieldMetric subclass today (see
                this class's own docstring for why it's still a
                constructor argument rather than hardcoded).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._label_column = label_column

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, masked_field, <label_column>)
        rows into (SingleCardInput, str) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from a MaskedFieldMetric subclass's
                output parquet file.
        Output: one (GenericCard, str) TrainingDatum per row whose
            nocab_uuid looks up a card - the label is passed through
            unchanged (masked_field_metric.py's scan() always writes a
            genuine str). Rows whose nocab_uuid doesn't look up a card
            are skipped.
        Side effects: none.
        Exceptions: none expected (per-row lookup failures are skipped,
            not raised - mirrors CardAverageDataConstructor.build()).
        """
        results: List[TrainingDatum] = []

        # Look up each row's card independently; skip rows whose
        # nocab_uuid doesn't look up a card, same as
        # CardAverageDataConstructor.build() - a metric's output may
        # contain the odd unresolvable card id.
        for _, row in chunk.iterrows():
            card = self._card_for_uuid(row["nocab_uuid"])
            if card is None:
                continue
            results.append((card, row[self._label_column]))

        return results

    def _card_for_uuid(self, raw_nocab_uuid: object) -> GenericCard | None:
        """Look up the card for one row's raw nocab_uuid value.

        Private helper - single consumer is build(). Same UUID-parsing-
        and-lookup logic as CardAverageDataConstructor's own
        _card_for_uuid() - duplicated rather than shared, since these
        two classes have no other coupling and PRINCIPLES.md's
        "near-identical logic" guidance is about not letting the same
        few lines drift, not about forcing every DataConstructor to
        share a base class for one helper (see design-recipe-implement
        pass for whether a shared home turns out to be worth it once
        both bodies are written).

        Inputs:
            raw_nocab_uuid: a row's "nocab_uuid" cell, expected to be a
                str parseable as a UUID.
        Output: the matching GenericCard, or None if raw_nocab_uuid
            doesn't parse as a UUID or self._card_binder has no card for
            it.
        Side effects: none.
        Exceptions: none - all failures collapse to None.
        """
        try:
            card_uuid = UUID(str(raw_nocab_uuid))
        except (TypeError, ValueError):
            return None
        return self._card_binder.get_by_uuid(card_uuid)


class CardCharacterPredictionDataConstructor:
    """DataConstructor for CardCharacterPredictionMetric (see
    src/data_refinement/metrics/sts_gg/card_character_prediction_metric.py)
    - single card in, a whole character-probability distribution out,
    as a raw Dict[str, float] (character -> probability). This is the
    one metric in its family (no siblings, unlike CardAverageMetric's
    nine), so there's no label_column to parameterize - this metric's
    finalize() always writes the same fixed characters/probabilities
    column pair.

    Returns the RAW distribution as read off the row, unencoded against
    any vocabulary - matches every other DataConstructor's convention
    here (a metric's label_values-based encoding belongs to the
    consuming loss, not this class - see plans/dojo_v2.md's "Fixed-
    classification label encoding lives in the generic dojo, not the
    DataConstructor" note, which this class follows even though its
    consuming loss is SoftClassificationLoss rather than
    FixedClassificationLoss). In particular, this class does NOT fold
    an out-of-vocabulary character to OTHER_LABEL - see
    SoftClassificationLoss's own docstring for why that check lives
    there instead.
    """

    def __init__(self, card_binder: CardBinder) -> None:
        """
        Inputs:
            card_binder: registry to resolve each row's nocab_uuid
                against. Never written to.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, characters, probabilities,
        sample_count) rows into (SingleCardInput, Dict[str, float])
        TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from CardCharacterPredictionMetric's
                output parquet file (see its finalize() docstring for
                the exact column schema - characters/probabilities are
                parallel lists, summing to 1.0 per row).
        Output: one (GenericCard, Dict[str, float]) TrainingDatum per
            row whose nocab_uuid resolves to a card - the dict zips
            that row's characters/probabilities lists together
            unchanged (sample_count is not part of the label; it's a
            metric-side diagnostic column, not consumed here). Rows
            whose nocab_uuid doesn't resolve are skipped.
        Side effects: none.
        Exceptions: none expected (per-row lookup failures are skipped,
            not raised - mirrors CardAverageDataConstructor.build()).

        Example:
            >>> constructor = CardCharacterPredictionDataConstructor(card_binder)
            >>> constructor.build(chunk)
            [(<GenericCard>, {"CHARACTER.SILENT": 0.6, "CHARACTER.REGENT": 0.4}), ...]
        """
        results: List[TrainingDatum] = []

        # Look up each row's card independently; skip rows whose
        # nocab_uuid doesn't look up a card, same as
        # CardAverageDataConstructor.build() - a metric's output may
        # contain the odd unresolvable card id.
        for _, row in chunk.iterrows():
            card = self._card_for_uuid(row["nocab_uuid"])
            if card is None:
                continue
            distribution: Dict[str, float] = dict(
                zip(row["characters"], row["probabilities"])
            )
            results.append((card, distribution))

        return results

    def _card_for_uuid(self, raw_nocab_uuid: object) -> GenericCard | None:
        """Look up the card for one row's raw nocab_uuid value.

        Private helper - single consumer is build(). Same UUID-parsing-
        and-lookup logic as CardAverageDataConstructor's/
        MaskedFieldDataConstructor's own _card_for_uuid() - duplicated
        rather than shared, per this module's established convention
        (see MaskedFieldDataConstructor._card_for_uuid()'s own
        docstring for why).

        Inputs:
            raw_nocab_uuid: a row's "nocab_uuid" cell, expected to be a
                str parseable as a UUID.
        Output: the matching GenericCard, or None if raw_nocab_uuid
            doesn't parse as a UUID or self._card_binder has no card for
            it.
        Side effects: none.
        Exceptions: none - all failures collapse to None.
        """
        try:
            card_uuid = UUID(str(raw_nocab_uuid))
        except (TypeError, ValueError):
            return None
        return self._card_binder.get_by_uuid(card_uuid)
