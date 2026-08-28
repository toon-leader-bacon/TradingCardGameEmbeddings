from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.opening_hand_win_rate import (
    OpeningHandWinRateMetric,
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


class TestOpeningHandWinRateMetric:
    def test_uses_opening_hand_column_not_deck(self) -> None:
        job = OpeningHandWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame(
            {
                "opening_hand_Bolt": [1, 1, 0],
                "deck_Bolt": [0, 0, 0],
                "won": [True, False, True],
            }
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 0.5

    def test_metric_name_is_opening_hand_win_rate(self) -> None:
        job = OpeningHandWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"opening_hand_Bolt": [1], "won": [True]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize()[uuid].metric_name == "opening_hand_win_rate"
        assert job.name == "opening_hand_win_rate"

    def test_card_never_in_opening_hand_is_absent_from_results(self) -> None:
        job = OpeningHandWinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"opening_hand_Bolt": [0, 0], "won": [True, False]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}
