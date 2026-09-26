"""DataConstructor for the MaskedFieldRegressionMetric family - see
src/data_refinement/metrics/generic/masked_field_regression_metric.py."""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors._row_values import (
    _card_for_uuid,
    _label_as_float,
)
from src.schema.type_hints import TrainingDatum


class MaskedFieldRegressionDataConstructor:
    """DataConstructor for the MaskedFieldRegressionMetric family (see
    src/data_refinement/metrics/generic/masked_field_regression_metric.py)
    - single card in, single float label out. Sibling to
    MaskedFieldDataConstructor (masked_field.py): same
    (nocab_uuid, masked_field, label) row shape, but label is a genuine
    float here rather than a raw classification string, so it needs
    CardAverageDataConstructor's float/NaN parsing
    (_label_as_float) instead of MaskedFieldDataConstructor's pass-
    through. Kept as its own class rather than unifying with either
    sibling - mirrors the deliberate split between MaskedFieldMetric
    and MaskedFieldRegressionMetric themselves (see that module's
    docstring for why a shared base isn't worth it).

    label_column, like MaskedFieldDataConstructor, is accepted purely
    to keep every DataConstructor in this package to the same
    constructor shape - today's only consumer (CostRegressionMetric)
    always passes label_column="label".
    """

    def __init__(self, label_column: str) -> None:
        """
        Inputs:
            label_column: the column name holding this metric's label -
                "label" for every MaskedFieldRegressionMetric subclass
                today.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._label_column = label_column

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, masked_field, <label_column>)
        rows into (SingleCardInput, float) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from a MaskedFieldRegressionMetric
                subclass's output parquet file.
        Output: one (GenericCard, float) TrainingDatum per row whose
            nocab_uuid resolves to a card and whose label parses as a
            non-NaN float. Rows that fail either check are skipped.
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors CardAverageDataConstructor.build()).
        """
        results: List[TrainingDatum] = []

        # Resolve each row's card and label independently; skip rows
        # that fail either resolution rather than raising, matching
        # every other DataConstructor in this package.
        for _, row in chunk.iterrows():
            card = _card_for_uuid(lookup, row["nocab_uuid"])
            if card is None:
                continue
            label = _label_as_float(row[self._label_column])
            if label is None:
                continue
            results.append((card, label))

        return results
