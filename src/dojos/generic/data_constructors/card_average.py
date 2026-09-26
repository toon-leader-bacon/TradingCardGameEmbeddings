"""DataConstructor for the CardAverageMetric family - see
src/data_refinement/metrics/sts_gg/card_average_metric.py."""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors._row_values import (
    _card_for_uuid,
    _label_as_float,
)
from src.schema.type_hints import TrainingDatum


class CardAverageDataConstructor:
    """DataConstructor for the CardAverageMetric family (see
    src/data_refinement/metrics/sts_gg/card_average_metric.py) - single
    card in, single float label out. Every concrete CardAverageMetric
    subclass (card_average_metrics.py) shares this same row shape
    (nocab_uuid, LABEL_COLUMN, sample_count), differing only in which
    column holds the label - so, unlike AveragePickNumberDataConstructor
    (v1, hardcoded to "average_pick_number"), label_column is a
    constructor argument here, letting one class serve all 9 subclasses.

    uuid_column defaults to "nocab_uuid" for that same family, but is
    itself a constructor argument for the rare metric whose id column
    is named something else because it isn't "the" card in the usual
    single-card-per-row sense - e.g. TutorTargetPoolMetric's
    "pool_card_uuid", one of several pool cards per output row.
    """

    def __init__(
        self,
        label_column: str,
        uuid_column: str = "nocab_uuid",
    ) -> None:
        """
        Inputs:
            label_column: the column name holding this metric's label
                (e.g. "average_relic_count", "win_rate") - read off the
                paired CardAverageMetric subclass's own LABEL_COLUMN
                ClassVar by the thin wrapper that constructs this, never
                duplicated as a literal (see plans/dojo_v2.md).
            uuid_column: the column name holding this row's card id.
                Defaults to "nocab_uuid" (every CardAverageMetric
                subclass); override for a metric family whose id column
                is named differently (see class docstring).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._label_column = label_column
        self._uuid_column = uuid_column

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (uuid_column, <label_column>) rows into
        (SingleCardInput, Label) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from a CardAverageMetric subclass's
                (or equivalently-shaped metric's) output parquet file.
        Output: one (GenericCard, float) TrainingDatum per row whose
            uuid_column resolves to a card and whose label parses as a
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
            card = _card_for_uuid(lookup, row[self._uuid_column])
            if card is None:
                continue
            label = _label_as_float(row[self._label_column])
            if label is None:
                continue
            results.append((card, label))

        return results
