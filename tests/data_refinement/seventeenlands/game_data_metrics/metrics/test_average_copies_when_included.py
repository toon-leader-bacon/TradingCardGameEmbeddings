from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.average_copies_when_included import (
    AverageCopiesWhenIncludedMetric,
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
    def test_sums_actual_copy_count_not_boolean(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        # 2 copies and 3 copies, plus a game where the card isn't run.
        chunk = pd.DataFrame({"deck_Bolt": [2, 3, 0]})

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 2.5

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(pd.DataFrame({"deck_Bolt": [2]}), card_columns)
        job.accumulate(pd.DataFrame({"deck_Bolt": [4]}), card_columns)

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 3.0

    def test_card_never_included_is_absent_from_results(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        chunk = pd.DataFrame({"deck_Bolt": [0, 0]})

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}


class TestSaveAndLoadState:
    def test_round_trips_copy_sum_and_inclusion_count(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        job.accumulate(pd.DataFrame({"deck_Bolt": [2, 3]}), [_card_columns(uuid)])

        state = job.save_state()
        restored = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        job.accumulate(pd.DataFrame({"deck_Bolt": [2]}), card_columns)

        restored = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(job.save_state())
        restored.accumulate(pd.DataFrame({"deck_Bolt": [4]}), card_columns)

        result = restored.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 3.0

    def test_state_uses_string_keys_for_json_compatibility(self) -> None:
        job = AverageCopiesWhenIncludedMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        uuid = uuid4()
        job.accumulate(pd.DataFrame({"deck_Bolt": [1]}), [_card_columns(uuid)])

        state = job.save_state()

        assert all(isinstance(k, str) for k in state["copy_sum"])
        assert all(isinstance(k, str) for k in state["inclusion_count"])
