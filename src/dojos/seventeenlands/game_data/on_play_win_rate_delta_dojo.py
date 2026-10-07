"""Wrapper over OnPlayWinRateDeltaMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_delta_metric.py).

NULLABLE LABEL: a card never seen on one side of on_play has a null
delta (NaN in the float64 column); row_values.label_as_float() skips
those rows (see src/dojos/generic/data_constructors/).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class OnPlayWinRateDeltaDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(won | in deck, on_play) - P(won | in deck,
    on_draw) (OnPlayWinRateDeltaMetric)."""

    METRIC = OnPlayWinRateDeltaMetric
