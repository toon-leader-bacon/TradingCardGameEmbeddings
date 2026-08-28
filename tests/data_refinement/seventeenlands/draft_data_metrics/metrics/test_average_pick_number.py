from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.draft_data_metrics.metrics.average_pick_number import (
    AveragePickNumberMetric,
)


def _resolved_picks(*uuids) -> pd.Series:
    return pd.Series(list(uuids))


class TestAccumulate:
    def test_averages_pick_number_for_a_single_card(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        other = uuid4()
        chunk = pd.DataFrame({"pick_number": [0, 4, 9]})
        resolved_picks = _resolved_picks(uuid, uuid, other)

        job.accumulate(chunk, resolved_picks)

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 2.0

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()

        job.accumulate(pd.DataFrame({"pick_number": [0]}), _resolved_picks(uuid))
        job.accumulate(pd.DataFrame({"pick_number": [8]}), _resolved_picks(uuid))

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 4.0

    def test_card_absent_from_chunk_leaves_state_untouched(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        other = uuid4()

        job.accumulate(
            pd.DataFrame({"pick_number": [0, 0]}), _resolved_picks(other, other)
        )

        assert uuid not in job.finalize()

    def test_unresolved_rows_are_excluded(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"pick_number": [0, 5]})
        resolved_picks = _resolved_picks(uuid, None)

        job.accumulate(chunk, resolved_picks)

        result = job.finalize()[uuid]
        assert result.sample_size == 1
        assert result.value == 0.0


class TestSaveAndLoadState:
    def test_round_trips(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"pick_number": [1, 3]}), _resolved_picks(uuid, uuid)
        )

        state = job.save_state()
        restored = AveragePickNumberMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_state_uses_string_keys(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(pd.DataFrame({"pick_number": [1]}), _resolved_picks(uuid))

        state = job.save_state()

        assert all(isinstance(k, str) for k in state["pick_number_sum"])
        assert all(isinstance(k, str) for k in state["pick_count"])

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(pd.DataFrame({"pick_number": [0]}), _resolved_picks(uuid))

        restored = AveragePickNumberMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(job.save_state())
        restored.accumulate(pd.DataFrame({"pick_number": [8]}), _resolved_picks(uuid))

        result = restored.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 4.0
