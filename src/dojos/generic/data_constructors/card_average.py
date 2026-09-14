"""DataConstructor for the CardAverageMetric family - see
src/data_refinement/metrics/sts_gg/card_average_metric.py."""

import math
from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors._uuid_resolution import _card_for_uuid
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
        card_binder: CardBinder,
        label_column: str,
        uuid_column: str = "nocab_uuid",
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve each row's uuid_column
                against. Never written to.
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
        self._card_binder = card_binder
        self._label_column = label_column
        self._uuid_column = uuid_column

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
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
            card = _card_for_uuid(self._card_binder, row[self._uuid_column])
            if card is None:
                continue
            label = self._label_as_float(row[self._label_column])
            if label is None:
                continue
            results.append((card, label))

        return results

    def _label_as_float(self, raw_label: object) -> float | None:
        """Parse one row's raw label cell as a float.

        Private helper - single consumer is build().

        Inputs:
            raw_label: a row's label-column cell (float, int, or bool -
                see CardAverageMetric's WIN RATE IS AN AVERAGE note for
                why a bool is a legitimate input here).
        Output: raw_label as a float, or None if it doesn't convert, or
            if it's NaN - a metric's own nullable-output convention
            (e.g. OnPlayWinRateDeltaMetric's None for a card never seen
            on one side) round-trips through a float64 parquet column
            as NaN, not None, so it's treated the same as an
            unparseable label rather than becoming a NaN training
            label.
        Side effects: none.
        Exceptions: none - all failures collapse to None.
        """
        if isinstance(raw_label, (float, int, bool)):
            value = float(raw_label)
            return None if math.isnan(value) else value
        return None
