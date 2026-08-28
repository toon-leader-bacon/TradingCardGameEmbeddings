from uuid import uuid4

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.win_rate import (
    WinRateMetric,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult


def _card_columns(nocab_uuid, name: str = "Bolt") -> CardColumnSet:
    return CardColumnSet(
        nocab_uuid=nocab_uuid,
        opening_hand=f"opening_hand_{name}",
        drawn=f"drawn_{name}",
        tutored=f"tutored_{name}",
        deck=f"deck_{name}",
        sideboard=f"sideboard_{name}",
    )


def _chunk(deck_values: list[int], won_values: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"deck_Bolt": deck_values, "won": won_values})


class TestAccumulate:
    def test_computes_wins_and_games_from_one_chunk(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        chunk = _chunk([1, 1, 1, 0], [True, True, False, True])

        job.accumulate(chunk, card_columns)

        result = job.finalize()[uuid]
        assert result.sample_size == 3
        assert result.value == 2 / 3

    def test_multiple_calls_accumulate_not_overwrite(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(_chunk([1, 1], [True, False]), card_columns)
        job.accumulate(_chunk([1, 1], [True, True]), card_columns)

        result = job.finalize()[uuid]
        assert result.sample_size == 4
        assert result.value == 3 / 4

    def test_ignores_games_where_card_not_in_deck(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]

        job.accumulate(_chunk([0, 0, 1], [True, True, False]), card_columns)

        result = job.finalize()[uuid]
        assert result.sample_size == 1
        assert result.value == 0.0

    def test_card_never_in_any_deck_is_absent_from_results(self) -> None:
        # card_columns is the invariant full list of every resolved
        # card in the file, not just cards present in a given chunk —
        # a card whose deck column is always 0 across the whole run
        # must never appear in finalize()'s output (not appear with a
        # 0/0 division). Regression test for a bug the file-critic
        # caught: unconditionally touching self._games on every
        # accumulate() call inserted a 0-count entry for such cards,
        # causing finalize() to raise ZeroDivisionError.
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        never_played_uuid = uuid4()
        played_uuid = uuid4()
        card_columns = [
            _card_columns(never_played_uuid, name="Never Played"),
            _card_columns(played_uuid, name="Played"),
        ]
        chunk = pd.DataFrame(
            {
                "deck_Never Played": [0, 0],
                "deck_Played": [1, 1],
                "won": [True, False],
            }
        )

        job.accumulate(chunk, card_columns)

        results = job.finalize()
        assert never_played_uuid not in results
        assert played_uuid in results


class TestFinalize:
    def test_produces_one_metric_result_per_card_seen(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(_chunk([1], [True]), [_card_columns(uuid)])

        results = job.finalize()

        assert results == {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name="win_rate",
                value=1.0,
                sample_size=1,
                expansion="MSH",
                format="PremierDraft",
            )
        }

    def test_no_accumulation_produces_no_results(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")

        assert job.finalize() == {}


class TestSaveAndLoadState:
    def test_round_trips_wins_and_games(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(_chunk([1, 1], [True, False]), [_card_columns(uuid)])

        state = job.save_state()
        restored = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        restored.load_state(state)

        assert restored.finalize() == job.finalize()

    def test_restored_job_continues_accumulating_correctly(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        card_columns = [_card_columns(uuid)]
        job.accumulate(_chunk([1, 1], [True, False]), card_columns)

        restored = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        restored.load_state(job.save_state())
        restored.accumulate(_chunk([1], [True]), card_columns)

        result = restored.finalize()[uuid]
        assert result.sample_size == 3
        assert result.value == 2 / 3

    def test_state_uses_string_keys_for_json_compatibility(self) -> None:
        job = WinRateMetric(expansion="MSH", format_code="PremierDraft")
        uuid = uuid4()
        job.accumulate(_chunk([1], [True]), [_card_columns(uuid)])

        state = job.save_state()

        assert all(isinstance(k, str) for k in state["wins"])
        assert all(isinstance(k, str) for k in state["games"])
