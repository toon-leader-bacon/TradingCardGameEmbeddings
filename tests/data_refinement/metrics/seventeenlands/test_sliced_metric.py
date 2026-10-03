"""Tests for sliced_metric.py's is_count_table."""

import pytest

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.sliced_metric import is_count_table


@pytest.mark.parametrize(
    "metric", [DrawnWinRateMetric, OnPlayWinRateSensitivityByDeckMetric]
)
def test_count_tables(metric: type) -> None:
    assert is_count_table(metric)


def test_row_stream() -> None:
    assert not is_count_table(DeckWinPredictionMetric)


def test_a_class_of_both_kinds_or_neither_is_rejected() -> None:
    class Both:
        COUNT_COLUMNS = ("a",)
        IS_ROW_STREAM = True

    class Neither:
        pass

    for metric in (Both, Neither):
        with pytest.raises(TypeError, match="exactly one"):
            is_count_table(metric)  # type: ignore[arg-type]
