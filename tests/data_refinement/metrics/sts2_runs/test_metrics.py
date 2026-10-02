import dataclasses
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.sts2_runs import card_average_metrics as cards
from src.data_refinement.metrics.sts2_runs import deck_label_metrics as decks
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import Sts2Run
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.sts2_runs._runs import binder, raw_run


@pytest.fixture
def parse(tmp_path: Path):
    parser = Sts2RunParser(binder(tmp_path), SpireCodexRunsDeckExtractionStage())
    return lambda **overrides: parser.parse(raw_run(**overrides))


def _deck_rows(metric_cls, runs: list[Sts2Run], tmp_path: Path) -> list[dict]:
    metric = metric_cls(binder(tmp_path), tmp_path / "out.parquet")
    for run in runs:
        metric.accumulate(run)
    return pq.read_table(metric.finalize()).to_pylist()


def _card_labels(metric_cls, runs: list[Sts2Run], tmp_path: Path) -> dict[str, float]:
    lookup = binder(tmp_path)
    metric = metric_cls(lookup, tmp_path / "out.parquet")
    for run in runs:
        metric.accumulate(run)
    frame = pd.read_parquet(metric.finalize())
    names = {}
    for card_id in ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE"):
        card = lookup.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id
        )
        assert card is not None
        names[str(card.nocab_uuid)] = card_id
    return {
        names[row.nocab_uuid]: getattr(row, metric_cls.LABEL_COLUMN)
        for row in frame.itertuples()
    }


class TestDeckLabelMetrics:
    @pytest.mark.parametrize(
        "metric_cls,expected",
        [
            (decks.AscensionPredictionMetric, 3),
            (decks.CharacterPredictionMetric, "CHARACTER.SILENT"),
            (decks.WinMetric, False),
            (decks.KilledByMetric, "ENCOUNTER.THE_KIN_BOSS"),
            (decks.RelicCountMetric, 2),
            (decks.TotalDamageTakenMetric, 52),
            (decks.TotalCardsPickedMetric, 2),
            (decks.TotalCardsSkippedMetric, 2),
            (decks.TotalTurnsMetric, 14),
            (decks.ElitesKilledMetric, 1),
            (decks.FloorsClearedMetric, 3),
            (decks.TotalCombatsMetric, 4),
        ],
    )
    def test_one_row_per_player_pointing_at_its_published_deck(
        self, metric_cls, expected, parse, tmp_path: Path
    ) -> None:
        run = parse()

        (row,) = _deck_rows(metric_cls, [run], tmp_path)

        assert row == {
            "run_id": "abc123",
            "deck_uuid": str(run.players[0].deck_uuid),
            metric_cls.LABEL_COLUMN: expected,
        }

    def test_output_requires_the_deck_box(self, parse, tmp_path: Path) -> None:
        metric = decks.WinMetric(binder(tmp_path), tmp_path / "out.parquet")
        metric.accumulate(parse())
        metadata = read_version_metadata(metric.finalize())
        assert metadata is not None and metadata.requires_deck_box

    def test_killed_by_writes_no_row_for_a_win_and_other_for_a_rare_killer(
        self, parse, tmp_path: Path
    ) -> None:
        runs = [
            parse(win=True, killed_by_encounter="NONE.NONE"),
            parse(killed_by_encounter="ENCOUNTER.SOMETHING_RARE"),
        ]
        rows = _deck_rows(decks.KilledByMetric, runs, tmp_path)
        assert [row["killed_by"] for row in rows] == [OTHER_LABEL]

    def test_a_co_op_run_writes_each_players_own_label(
        self, parse, tmp_path: Path
    ) -> None:
        run = parse()
        partner = dataclasses.replace(
            run.players[0], deck_uuid=uuid4(), character="CHARACTER.IRONCLAD"
        )
        co_op = dataclasses.replace(run, players=(run.players[0], partner))
        rows = _deck_rows(decks.CharacterPredictionMetric, [co_op], tmp_path)
        assert [row["character"] for row in rows] == [
            "CHARACTER.SILENT",
            "CHARACTER.IRONCLAD",
        ]


class TestCardAverageMetrics:
    def test_unaliased_cards_get_no_row(self, parse, tmp_path: Path) -> None:
        labels = _card_labels(cards.CardRelicCountMetric, [parse()], tmp_path)
        assert labels == {"STRIKE_SILENT": 2, "DEFEND_SILENT": 2, "NEUTRALIZE": 2}

    def test_win_rate_averages_over_runs(self, parse, tmp_path: Path) -> None:
        labels = _card_labels(
            cards.CardWinRateMetric, [parse(), parse(win=True)], tmp_path
        )
        assert labels["STRIKE_SILENT"] == 0.5

    def test_upgrade_rate_reads_each_copy(self, parse, tmp_path: Path) -> None:
        labels = _card_labels(cards.CardUpgradeRateMetric, [parse()], tmp_path)
        assert labels == {"STRIKE_SILENT": 0.0, "DEFEND_SILENT": 1.0, "NEUTRALIZE": 0.0}

    def test_act_2_win_rate_counts_copies_added_before_act_2(
        self, parse, tmp_path: Path
    ) -> None:
        # NEUTRALIZE joined on floor 3, act 2's first floor: not counted
        labels = _card_labels(
            cards.CardWinRateAtAct2Metric, [parse(win=True)], tmp_path
        )
        assert labels == {"STRIKE_SILENT": 1.0, "DEFEND_SILENT": 1.0}

    def test_act_2_win_rate_skips_runs_that_never_reached_act_2(
        self, parse, tmp_path: Path
    ) -> None:
        one_act = parse(map_point_history=[raw_run()["map_point_history"][0]])
        assert _card_labels(cards.CardWinRateAtAct2Metric, [one_act], tmp_path) == {}

    def test_deck_size_counts_every_copy(self, parse, tmp_path: Path) -> None:
        labels = _card_labels(cards.CardDeckSizeMetric, [parse()], tmp_path)
        assert labels["NEUTRALIZE"] == 4
