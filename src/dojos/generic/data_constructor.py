"""Structural contract every per-metric-family DataConstructor satisfies.

A DataConstructor maps 1:1 to a metric FAMILY (the Template Method bases
under src/data_refinement/metrics/, e.g. CardAverageMetric), never to an
individual concrete metric subclass and never to a (input shape, task
shape) generic dojo cell - see plans/dojo_v2.md's component overview.
A generic dojo (src/dojos/generic/*/dojo.py) takes one of these as an
injected collaborator rather than owning/building it itself, so the same
DataConstructor implementation can be reused across every concrete metric
in its family (each with its own output path and label vocabulary/column).
"""

from typing import List, Protocol, runtime_checkable

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.type_hints import TrainingDatum


@runtime_checkable
class DataConstructor(Protocol):
    """Converts one chunk of a metric's raw output rows into TrainingDatum
    (input, label) pairs, ready for Batch.from_training_data()."""

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """
        Inputs:
            chunk: one chunk of rows read from a metric's own output
                parquet file (schema is family-specific - e.g.
                CardAverageMetric's nocab_uuid/LABEL_COLUMN/sample_count).
            lookup: the card store to resolve uuids against, supplied by
                the dojo per split (a holdout-filtered view, so a hidden
                card is a miss exactly like an unknown one). Never
                written to; constructors hold no card store of their own.
        Output: one TrainingDatum per row this family can resolve into a
            valid training example. A row that fails to resolve (e.g. an
            unresolvable or held-out card uuid) is skipped, not raised on
            - resolution failure is expected, not exceptional. What a
            miss on an input card means (skip the row vs drop that card)
            is each constructor's own documented convention; a miss on
            the card a label refers to always skips the row.
        Side effects: implementation-defined (expected: none - reads only,
            against `lookup` and any deck box injected at construction).
        Exceptions: implementation-defined.
        """
        ...
