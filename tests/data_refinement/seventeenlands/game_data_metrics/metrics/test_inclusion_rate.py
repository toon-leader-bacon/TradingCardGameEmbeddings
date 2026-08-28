from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.inclusion_rate import (
    InclusionRateMetric,
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


def _chunk(deck_values: list[int]) -> pd.DataFrame:
    # Deliberately no "won" column — inclusion rate never reads it.
    return pd.DataFrame({"deck_Bolt": deck_values})


class TestAccumulate:
    def test_denominator_is_all_games_not_just_inclusions(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()

        job.accumulate(_chunk([1, 1, 0, 0]), [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == 4
        assert result.value == 0.5

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(_chunk([1, 0]), card_columns)
        job.accumulate(_chunk([1, 1]), card_columns)

        result = job.finalize()[uuid]
        assert result.sample_size == 4
        assert result.value == 3 / 4

    def test_card_never_included_is_absent_from_results(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()

        job.accumulate(_chunk([0, 0, 0]), [_card_columns(uuid)])

        assert job.finalize() == {}

    def test_denominator_grows_even_for_untouched_cards(self) -> None:
        # A card with zero inclusions still contributes to
        # games_processed via every OTHER card's accumulate() call in
        # the same chunk (games_processed is chunk-wide, not per-card).
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        never_included = uuid4()
        always_included = uuid4()
        chunk = pd.DataFrame(
            {"deck_Never": [0, 0], "deck_Always": [1, 1]},
        )
        card_columns = [
            _card_columns(never_included, name="Never"),
            _card_columns(always_included, name="Always"),
        ]

        job.accumulate(chunk, card_columns)

        assert never_included not in job.finalize()
        result = job.finalize()[always_included]
        assert result.sample_size == 2
        assert result.value == 1.0


class TestSaveAndLoadState:
    def test_round_trips_games_processed_and_inclusions(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(_chunk([1, 0]), [_card_columns(uuid)])

        state = job.save_state()
        restored = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        job.accumulate(_chunk([1, 0]), card_columns)

        restored = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        restored.load_state(job.save_state())
        restored.accumulate(_chunk([1, 1]), card_columns)

        result = restored.finalize()[uuid]
        assert result.sample_size == 4
        assert result.value == 3 / 4

    def test_state_uses_string_keys_for_json_compatibility(self) -> None:
        job = InclusionRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(_chunk([1]), [_card_columns(uuid)])

        state = job.save_state()

        assert isinstance(state["games_processed"], int)
        assert all(isinstance(k, str) for k in state["inclusions"])
