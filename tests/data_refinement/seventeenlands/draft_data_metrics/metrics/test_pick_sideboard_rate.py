from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.draft_data_metrics.metrics.pick_sideboard_rate import (
    PickSideboardRateMetric,
)


def _resolved_picks(*uuids) -> pd.Series:
    return pd.Series(list(uuids))


class TestAccumulate:
    def test_averages_sideboard_rate_for_a_single_card(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        other = uuid4()
        chunk = pd.DataFrame({"pick_sideboard_in_rate": [0.2, 0.6, 0.9]})
        resolved_picks = _resolved_picks(uuid, uuid, other)

        job.accumulate(chunk, resolved_picks)

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 0.4

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()

        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.2]}), _resolved_picks(uuid)
        )
        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.6]}), _resolved_picks(uuid)
        )

        result = job.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 0.4

    def test_card_absent_from_chunk_leaves_state_untouched(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        other = uuid4()

        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.1, 0.1]}),
            _resolved_picks(other, other),
        )

        assert uuid not in job.finalize()

    def test_unresolved_rows_are_excluded(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame({"pick_sideboard_in_rate": [0.2, 0.9]})
        resolved_picks = _resolved_picks(uuid, None)

        job.accumulate(chunk, resolved_picks)

        result = job.finalize()[uuid]
        assert result.sample_size == 1
        assert result.value == 0.2


class TestSaveAndLoadState:
    def test_round_trips(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.2, 0.4]}),
            _resolved_picks(uuid, uuid),
        )

        state = job.save_state()
        restored = PickSideboardRateMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_state_uses_string_keys(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.2]}), _resolved_picks(uuid)
        )

        state = job.save_state()

        assert all(isinstance(k, str) for k in state["sideboard_rate_sum"])
        assert all(isinstance(k, str) for k in state["pick_count"])

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = PickSideboardRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.2]}), _resolved_picks(uuid)
        )

        restored = PickSideboardRateMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(job.save_state())
        restored.accumulate(
            pd.DataFrame({"pick_sideboard_in_rate": [0.6]}), _resolved_picks(uuid)
        )

        result = restored.finalize()[uuid]
        assert result.sample_size == 2
        assert result.value == 0.4
