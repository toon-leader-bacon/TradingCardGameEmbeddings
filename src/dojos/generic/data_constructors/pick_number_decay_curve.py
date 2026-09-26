"""DataConstructor for PickNumberDecayCurveMetric - see
src/data_refinement/metrics/seventeenlands/draft_data/pick_number_decay_curve_metric.py.
"""

from typing import Dict, List, cast

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors._row_values import _card_for_uuid
from src.schema.type_hints import TrainingDatum


class PickNumberDecayCurveDataConstructor:
    """DataConstructor for PickNumberDecayCurveMetric (see
    src/data_refinement/metrics/seventeenlands/draft_data/pick_number_decay_curve_metric.py)
    - single card in, a sparse bucket_index -> take_rate dict out
    (Dict[int, float]), scored by MaskedVectorRegressionLoss
    (src/dojos/loss/masked_vector_regression_loss.py) rather than
    FixedClassificationLoss/SoftClassificationLoss - see
    plans/seventeen_lands_dojos.md for why this metric's per-bucket
    take-rate values are independent probabilities, not a distribution
    that sums to 1.

    Like CardCharacterPredictionDataConstructor, this is the one metric
    in its family (no siblings), so there's no label_column to
    parameterize. min_sample_count IS a constructor argument though - a
    bucket's take_rate is only trustworthy once enough packs
    contributed to it; below that threshold, a bucket is treated the
    same as one that was never observed at all (see build()'s
    docstring), not as a noisy-but-real signal.

    Returns the RAW sparse dict as read off the row (bucket indices
    that clear min_sample_count only, values unchanged) - matches every
    other DataConstructor's convention of not doing vocabulary/target
    encoding itself (see CardCharacterPredictionDataConstructor's own
    docstring). Here the "vocabulary" is simply which dict keys are
    present, and MaskedVectorRegressionLoss reads that presence as the
    per-example mask.
    """

    def __init__(self, min_sample_count: int) -> None:
        """
        Inputs:
            min_sample_count: minimum sample_count_by_pick_number value
                a bucket needs before its take_rate is trusted enough to
                train on - a bucket below this threshold is dropped from
                that row's label dict the same way an unobserved
                (sample_count == 0) bucket already is. Human-chosen per
                dojo instance (PickNumberDecayCurveDojo uses 10) rather
                than hardcoded here, since it's a statistical-confidence
                choice, not a property of the metric's schema. Must be
                >= 1 - _label_for_row()'s take_rate-is-never-None
                guarantee for an included bucket depends on excluding
                sample_count == 0 (see that method's docstring).
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if min_sample_count < 1.
        """
        if min_sample_count < 1:
            raise ValueError(f"min_sample_count must be >= 1, got {min_sample_count}")
        self._min_sample_count = min_sample_count

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, take_rate_by_pick_number,
        sample_count_by_pick_number) rows into (SingleCardInput,
        Dict[int, float]) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from PickNumberDecayCurveMetric's
                output parquet file (see its finalize() docstring for
                the exact column schema - both list columns share the
                same per-row length and bucket-index order).
        Output: one (GenericCard, Dict[int, float]) TrainingDatum per
            row whose nocab_uuid resolves to a card AND has at least one
            bucket with sample_count_by_pick_number[i] >=
            self._min_sample_count. The dict has one bucket_index ->
            take_rate entry per bucket clearing that threshold; buckets
            below it (including genuinely unobserved ones, where
            take_rate is already None) are simply absent from the dict,
            not included with a filler value. Rows whose nocab_uuid
            doesn't resolve, or whose every bucket falls below the
            threshold, are skipped (an empty label dict carries no
            training signal - same "skip, don't raise" philosophy as
            every other DataConstructor here).
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors CardAverageDataConstructor.build()).

        Example:
            >>> constructor = PickNumberDecayCurveDataConstructor(
            ...     min_sample_count=10
            ... )
            >>> constructor.build(chunk, lookup)
            [(<GenericCard>, {0: 0.05, 1: 0.12, 14: 0.83}), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's card and its per-bucket label dict
        # independently; skip rows that fail either.
        for _, row in chunk.iterrows():
            card = _card_for_uuid(lookup, row["nocab_uuid"])
            if card is None:
                continue
            label = self._label_for_row(
                row["take_rate_by_pick_number"], row["sample_count_by_pick_number"]
            )
            if not label:
                continue
            results.append((card, label))

        return results

    def _label_for_row(
        self, take_rate_by_pick_number: object, sample_count_by_pick_number: object
    ) -> Dict[int, float]:
        """Build one row's sparse bucket_index -> take_rate dict.

        Private helper - single consumer is build().

        Inputs:
            take_rate_by_pick_number: a row's "take_rate_by_pick_number"
                cell (list[float | None]).
            sample_count_by_pick_number: that row's parallel
                "sample_count_by_pick_number" cell (list[int]), same
                length.
        Output: a bucket_index -> take_rate dict, one entry per index i
            where sample_count_by_pick_number[i] >=
            self._min_sample_count (implies take_rate_by_pick_number[i]
            is not None, since min_sample_count is always >= 1). Empty
            if no bucket clears the threshold.
        Side effects: none.
        Exceptions: none.
        """
        sample_counts = cast(List[int], sample_count_by_pick_number)
        take_rates = cast(List[float], take_rate_by_pick_number)

        label: Dict[int, float] = {}
        for bucket_index, sample_count in enumerate(sample_counts):
            if sample_count < self._min_sample_count:
                continue
            label[bucket_index] = float(take_rates[bucket_index])
        return label
