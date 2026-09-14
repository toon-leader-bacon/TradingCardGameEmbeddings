"""DataConstructor for CardCharacterPredictionMetric - see
src/data_refinement/metrics/sts_gg/card_character_prediction_metric.py."""

from typing import Dict, List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors._uuid_resolution import _card_for_uuid
from src.schema.type_hints import TrainingDatum


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
            card = _card_for_uuid(self._card_binder, row["nocab_uuid"])
            if card is None:
                continue
            distribution: Dict[str, float] = dict(
                zip(row["characters"], row["probabilities"])
            )
            results.append((card, distribution))

        return results
