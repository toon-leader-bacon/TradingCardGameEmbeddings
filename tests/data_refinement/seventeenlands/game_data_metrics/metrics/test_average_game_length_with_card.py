from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.average_game_length_with_card import (
    AverageGameLengthWithCardMetric,
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


class TestAccumulate:
    def test_averages_num_turns_conditioned_on_inclusion(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        # num_turns=99 on the excluded game must not affect the average.
        chunk = pd.DataFrame({"deck_Bolt": [1, 1, 0], "num_turns": [8, 12, 99]})

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 10.0

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(
            pd.DataFrame({"deck_Bolt": [1], "num_turns": [10]}), card_columns
        )
        job.accumulate(
            pd.DataFrame({"deck_Bolt": [1], "num_turns": [20]}), card_columns
        )

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 15.0

    def test_card_never_included_is_absent_from_results(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        chunk = pd.DataFrame({"deck_Bolt": [0, 0], "num_turns": [8, 12]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}


class TestSaveAndLoadState:
    def test_round_trips_turn_sum_and_game_count(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"deck_Bolt": [1, 1], "num_turns": [8, 12]}),
            [_card_columns(uuid)],
        )

        state = job.save_state()
        restored = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        job.accumulate(
            pd.DataFrame({"deck_Bolt": [1], "num_turns": [10]}), card_columns
        )

        restored = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(job.save_state())
        restored.accumulate(
            pd.DataFrame({"deck_Bolt": [1], "num_turns": [20]}), card_columns
        )

        result = restored.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 15.0

    def test_state_uses_string_keys_for_json_compatibility(self) -> None:
        job = AverageGameLengthWithCardMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"deck_Bolt": [1], "num_turns": [10]}),
            [_card_columns(uuid)],
        )

        state = job.save_state()

        assert all(isinstance(k, str) for k in state["turn_sum"])
        assert all(isinstance(k, str) for k in state["game_count"])
