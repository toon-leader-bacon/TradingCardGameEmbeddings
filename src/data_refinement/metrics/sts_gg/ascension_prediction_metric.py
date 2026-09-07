"""Deck -> ascension (BRAINSTORM.md multi-card #6).

Subclasses DeckLabelMetric (deck_label_metric.py) - a Template Method
base every metrics/sts_gg/deck_label_metrics.py class also subclasses.
This class predates that base (it was hand-written first, while the
DeckBox-reference architecture was still being worked out - see
plans/sts_gg_metrics.md's history) and has been folded onto it since,
rather than kept as an independent copy of the same sequence.
"""

from pathlib import Path

import pyarrow as pa

from src.data_refinement.metrics.sts_gg.deck_label_metric import DeckLabelMetric


class AscensionPredictionMetric(DeckLabelMetric):
    """Deck -> ascension level attempted, one row per run.

    BRAINSTORM.md frames this against a deck snapshot at a fixed floor
    - this class uses the final deck instead (this data source's only
    available snapshot - see ../README.md's "Deck references"), the
    same simplification every other DeckLabelMetric subclass makes.
    """

    LABEL_COLUMN = "ascension"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/ascension_prediction.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["ascension"]
