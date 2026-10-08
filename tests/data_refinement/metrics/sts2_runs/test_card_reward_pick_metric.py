from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import (
    CardRewardPickMetric,
)
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import Sts2Run
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.sts2_runs._runs import binder, raw_run


def _uuid(tmp_path: Path, card_id: str) -> str:
    card = binder(tmp_path).get_by_alias(
        GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id
    )
    assert card is not None
    return str(card.nocab_uuid)


def _rows(tmp_path: Path, run: Sts2Run) -> list[dict]:
    metric = CardRewardPickMetric(binder(tmp_path), tmp_path / "out.parquet")
    metric.accumulate(run)
    return pq.read_table(metric.finalize()).to_pylist()


def _parse(tmp_path: Path, raw: dict) -> Sts2Run:
    parser = Sts2RunParser(binder(tmp_path), SpireCodexRunsDeckExtractionStage())
    return parser.parse(raw)


class TestCardRewardPickMetric:
    def test_one_row_per_card_reward_with_its_deck_offer_and_pick(
        self, tmp_path: Path
    ) -> None:
        run = _parse(tmp_path, raw_run())

        first, second = _rows(tmp_path, run)

        neutralize = _uuid(tmp_path, "NEUTRALIZE")
        strike = _uuid(tmp_path, "STRIKE_SILENT")
        assert (first["floor"], second["floor"]) == (1, 3)
        assert first["run_id"] == "abc123"
        assert first["deck_uuids"] == []
        assert first["offered_uuids"] == [neutralize, strike]
        assert first["picked_uuid"] == neutralize
        assert second["deck_uuid"] == str(run.players[0].deck_uuid)

    def test_the_deck_drops_unaliased_cards_and_keeps_copies(
        self, tmp_path: Path
    ) -> None:
        _, second = _rows(tmp_path, _parse(tmp_path, raw_run()))

        # Strike and Defend from floor 1; the unaliased card (floor 2) is
        # dropped; Neutralize is added by this reward
        assert second["deck_uuids"] == [
            _uuid(tmp_path, "STRIKE_SILENT"),
            _uuid(tmp_path, "DEFEND_SILENT"),
        ]

    def test_a_skipped_reward_writes_a_null_pick(self, tmp_path: Path) -> None:
        raw = raw_run()
        for stats in raw["map_point_history"][0][0]["player_stats"]:
            for option in stats["card_choices"]:
                option["was_picked"] = False

        first, _ = _rows(tmp_path, _parse(tmp_path, raw))

        assert first["picked_uuid"] is None

    def test_an_offer_with_an_unaliased_card_writes_no_row(
        self, tmp_path: Path
    ) -> None:
        raw = raw_run()
        stats = raw["map_point_history"][0][0]["player_stats"][0]
        stats["card_choices"][1]["card"]["id"] = "CARD.NOT_IN_THE_BINDER"

        rows = _rows(tmp_path, _parse(tmp_path, raw))

        assert [row["floor"] for row in rows] == [3]

    def test_the_output_stamps_the_binder_without_requiring_a_deck_box(
        self, tmp_path: Path
    ) -> None:
        metric = CardRewardPickMetric(binder(tmp_path), tmp_path / "out.parquet")
        metric.finalize()

        metadata = read_version_metadata(tmp_path / "out.parquet")

        assert metadata.game is GameId.SLAY_THE_SPIRE_2
        assert not metadata.requires_deck_box

    @pytest.mark.parametrize("calls", [1, 2])
    def test_finalize_is_idempotent(self, tmp_path: Path, calls: int) -> None:
        metric = CardRewardPickMetric(binder(tmp_path), tmp_path / "out.parquet")

        paths = {metric.finalize() for _ in range(calls)}

        assert paths == {tmp_path / "out.parquet"}
