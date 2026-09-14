"""DataConstructor for the MaskedFieldMetric family - see
src/data_refinement/metrics/generic/masked_field_metric.py."""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors._uuid_resolution import _card_for_uuid
from src.schema.type_hints import TrainingDatum


class MaskedFieldDataConstructor:
    """DataConstructor for the MaskedFieldMetric family (see
    src/data_refinement/metrics/generic/masked_field_metric.py) - single card
    in, single raw string label out. Every concrete MaskedFieldMetric
    subclass (e.g. gwent_one's 8) shares the same fixed row shape
    (nocab_uuid, masked_field, label) - unlike CardAverageDataConstructor,
    the label column name isn't expected to actually vary across this
    family today. label_column is still accepted as a constructor
    argument, purely to keep every DataConstructor in this package to
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
            unchanged (generic/masked_field_metric.py's scan() always writes a
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
            card = _card_for_uuid(self._card_binder, row["nocab_uuid"])
            if card is None:
                continue
            results.append((card, row[self._label_column]))

        return results
