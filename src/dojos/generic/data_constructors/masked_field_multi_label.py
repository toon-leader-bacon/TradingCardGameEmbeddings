"""DataConstructor for the MaskedFieldMultiLabelMetric family - see
src/data_refinement/metrics/generic/masked_field_multi_label_metric.py."""

from typing import Dict, List, Sequence

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors.row_values import card_for_uuid
from src.schema.type_hints import TrainingDatum


class MaskedFieldMultiLabelDataConstructor:
    """Single card in, a dense {position: 0.0 or 1.0} dict out: one entry
    per label_values position, 1.0 where the row's label list holds that
    value. MaskedVectorRegressionLoss (MASKED_VECTOR_REGRESSION_LOSS_SPEC)
    scores each position independently, so every position is observed
    (a dense dict), unlike PickNumberDecayCurveDataConstructor's sparse
    one.
    """

    def __init__(self, label_values: Sequence[str]) -> None:
        """
        Inputs:
            label_values: the metric's ordered vocabulary
                (MaskedFieldMultiLabelMetric.LABEL_VALUES); position i of
                the label dict is label_values[i].
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if label_values is empty or repeats a value.
        """
        if not label_values or len(set(label_values)) != len(label_values):
            raise ValueError(
                f"label_values must be non-empty and distinct: {label_values!r}"
            )
        self._positions = {value: index for index, value in enumerate(label_values)}

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert (nocab_uuid, masked_field, label: list[str]) rows into
        (GenericCard, Dict[int, float]) TrainingDatum pairs.

        Inputs:
            chunk: rows of a MaskedFieldMultiLabelMetric parquet file.
            lookup: CardLookup for the rows' nocab_uuids.
        Output: one datum per row whose nocab_uuid looks up a card; the
            dict has every position, 1.0 for the row's labels and 0.0
            for the rest. A label outside label_values is ignored. Rows
            whose card is not found are skipped.
        Side effects: none.
        Exceptions: TypeError if a row's label is a str rather than a
            list (see _indicator_dict).

        Example:
            >>> MaskedFieldMultiLabelDataConstructor(["W", "U"]).build(chunk, lookup)
            [(<GenericCard>, {0: 1.0, 1: 0.0}), ...]
        """
        results: List[TrainingDatum] = []

        # One dense indicator dict per row whose card is found
        for _, row in chunk.iterrows():
            card = card_for_uuid(lookup, row["nocab_uuid"])
            if card is None:
                continue
            results.append((card, self._indicator_dict(row["label"])))

        return results

    def _indicator_dict(self, labels: Sequence[str]) -> Dict[int, float]:
        """{position: 1.0 if that value is in labels else 0.0} over every
        vocabulary position.

        Private helper - single consumer is build().

        Inputs: labels (one row's label list; a numpy array off parquet).
        Output: Dict[int, float] with len(label_values) entries.
        Side effects: none.
        Exceptions: TypeError if labels is a str: a single-label metric's
            file, whose label would otherwise be read character by
            character into an all-zeros vector.
        """
        if isinstance(labels, str):
            raise TypeError(f"expected a list of labels, got the str {labels!r}")
        present = {
            self._positions[value] for value in labels if value in self._positions
        }
        return {
            index: 1.0 if index in present else 0.0
            for index in self._positions.values()
        }
