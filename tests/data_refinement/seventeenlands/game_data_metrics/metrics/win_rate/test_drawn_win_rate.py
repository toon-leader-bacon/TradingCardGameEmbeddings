from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.drawn_win_rate import (
    DrawnWinRateMetric,
)


def _card_columns(nocab_uuid, name: str = "Bolt") -> CardColumnSet:
    return CardColumnSet(
        nocab_uuid=nocab_uuid,
        opening_hand=f"opening_hand_{name}",
        drawn=f"drawn_{name}",
        tutored=f"tutored_{name}",
        deck=f"deck_{name}",
        sideboard=f"sideboard_{name}",
    )


class TestDrawnWinRateMetric:
    def test_uses_drawn_column_not_deck(self) -> None:
        job = DrawnWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        # deck_Bolt is irrelevant here — only drawn_Bolt should matter.
        chunk = pd.DataFrame(
            {
                "drawn_Bolt": [1, 1, 0],
                "deck_Bolt": [0, 0, 0],
                "won": [True, False, True],
            }
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 0.5

    def test_metric_name_is_drawn_win_rate(self) -> None:
        job = DrawnWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"drawn_Bolt": [1], "won": [True]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize()[uuid].metric_name == "drawn_win_rate"
        assert job.name == "drawn_win_rate"

    def test_card_never_drawn_is_absent_from_results(self) -> None:
        job = DrawnWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"drawn_Bolt": [0, 0], "won": [True, False]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}
