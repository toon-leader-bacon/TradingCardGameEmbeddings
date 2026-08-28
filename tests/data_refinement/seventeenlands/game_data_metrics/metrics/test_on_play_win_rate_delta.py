from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.on_play_win_rate_delta import (
    OnPlayWinRateDeltaMetric,
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
    def test_computes_delta_between_on_play_and_on_draw_win_rates(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        # On play: 2 games, 2 wins (1.0). On draw: 2 games, 0 wins (0.0).
        chunk = pd.DataFrame(
            {
                "deck_Bolt": [1, 1, 1, 1],
                "on_play": [True, True, False, False],
                "won": [True, True, False, False],
            }
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.value == 1.0
        assert result.sample_size == 2

    def test_not_in_deck_rows_contribute_to_neither_bucket(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame(
            {
                "deck_Bolt": [0, 0],
                "on_play": [True, False],
                "won": [True, True],
            }
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(
            pd.DataFrame(
                {"deck_Bolt": [1, 1], "on_play": [True, False], "won": [True, False]}
            ),
            card_columns,
        )
        job.accumulate(
            pd.DataFrame(
                {"deck_Bolt": [1, 1], "on_play": [True, False], "won": [True, True]}
            ),
            card_columns,
        )

        result = job.finalize()[uuid]
        # on_play: 2 games, 2 wins (1.0); on_draw: 2 games, 1 win (0.5)
        assert result.value == 0.5
        assert result.sample_size == 2

    def test_card_observed_only_on_play_is_absent_from_results(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame(
            {"deck_Bolt": [1, 1], "on_play": [True, True], "won": [True, False]}
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}

    def test_card_observed_only_on_draw_is_absent_from_results(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame(
            {"deck_Bolt": [1, 1], "on_play": [False, False], "won": [True, False]}
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        assert job.finalize() == {}

    def test_sample_size_is_the_smaller_bucket(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        on_play_rows = 10
        on_draw_rows = 3
        chunk = pd.DataFrame(
            {
                "deck_Bolt": [1] * (on_play_rows + on_draw_rows),
                "on_play": [True] * on_play_rows + [False] * on_draw_rows,
                "won": [True] * on_play_rows + [True] * on_draw_rows,
            }
        )

        job.accumulate(chunk, [_card_columns(uuid)])

        result = job.finalize()[uuid]
        assert result.sample_size == on_draw_rows


class TestSaveAndLoadState:
    def test_round_trips_all_four_accumulators(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        chunk = pd.DataFrame(
            {
                "deck_Bolt": [1, 1, 1, 1],
                "on_play": [True, True, False, False],
                "won": [True, False, True, False],
            }
        )
        job.accumulate(chunk, [_card_columns(uuid)])

        state = job.save_state()
        restored = OnPlayWinRateDeltaMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        job.accumulate(
            pd.DataFrame(
                {"deck_Bolt": [1, 1], "on_play": [True, False], "won": [True, False]}
            ),
            card_columns,
        )

        restored = OnPlayWinRateDeltaMetric(
            expansion="MSH", format_code="PremierDraft"
        )
        restored.load_state(job.save_state())
        restored.accumulate(
            pd.DataFrame(
                {"deck_Bolt": [1, 1], "on_play": [True, False], "won": [True, True]}
            ),
            card_columns,
        )

        result = restored.finalize()[uuid]
        assert result.value == 0.5
        assert result.sample_size == 2

    def test_state_uses_string_keys_for_json_compatibility(self) -> None:
        job = OnPlayWinRateDeltaMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(
            pd.DataFrame(
                {"deck_Bolt": [1, 1], "on_play": [True, False], "won": [True, False]}
            ),
            [_card_columns(uuid)],
        )

        state = job.save_state()

        for key in (
            "wins_on_play",
            "games_on_play",
            "wins_on_draw",
            "games_on_draw",
        ):
            assert all(isinstance(k, str) for k in state[key])
