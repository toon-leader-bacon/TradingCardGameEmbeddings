import json
import logging
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.card_win_rate_at_act2_metric import (
    CardWinRateAtAct2Metric,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _binder_from_rows(rows: list[dict], tmp_path: Path) -> CardBinder:
    binder = CardBinder()
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps(rows), encoding="utf-8")
    SpireCodexCardIngestionStage().ingest(cards_path, binder)
    return binder


def _uuid_for(binder: CardBinder, card_id: str) -> UUID:
    card = binder.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id)
    assert card is not None
    return card.nocab_uuid


def _run_row(
    run_id: str,
    win: bool,
    deck_entries: list[tuple[str, int]],
    hp_per_floor: list[tuple[int, int]],
) -> dict:
    return {
        "id": run_id,
        "win": win,
        "deck": [
            {"id": f"CARD.{card_id}", "upgraded": False, "floor": floor}
            for card_id, floor in deck_entries
        ],
        "hpPerFloor": [
            {"floor": floor, "actIdx": act_idx} for floor, act_idx in hp_per_floor
        ],
    }


# actIdx 0 = act 1 (floors 1-17), actIdx 1 = act 2 starting floor 18 - a
# fixed, simple boundary used across these tests (the metric itself must
# never assume this fixed boundary - see the real derivation tests below).
_STANDARD_HP_PER_FLOOR = [(floor, 0) for floor in range(1, 18)] + [
    (floor, 1) for floor in range(18, 25)
]


def test_accepts_and_ignores_a_deck_box(tmp_path: Path) -> None:
    # deck_box exists only for constructor-shape consistency with this
    # container's multi-card metrics - this metric is per-card and
    # must not read from or write into it.
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = CardWinRateAtAct2Metric(
        binder, deck_box=deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        _run_row("run1", True, [("STRIKE_SILENT", 1)], _STANDARD_HP_PER_FLOOR)
    )
    metric.finalize()

    assert list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2)) == []


class TestAccumulate:
    def test_tallies_win_for_card_present_by_act2_start(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        metric.accumulate(
            _run_row("run1", True, [("STRIKE_SILENT", 5)], _STANDARD_HP_PER_FLOOR)
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        row = df.iloc[0]
        assert row["nocab_uuid"] == str(_uuid_for(binder, "STRIKE_SILENT"))
        assert row["win_rate"] == 1.0
        assert row["sample_count"] == 1

    def test_card_acquired_after_act2_start_is_excluded(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "GRAND_FINALE", "name": "Grand Finale"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        # Acquired at floor 20, after this run's act-2 start (floor 18).
        metric.accumulate(
            _run_row("run1", True, [("GRAND_FINALE", 20)], _STANDARD_HP_PER_FLOOR)
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 0

    def test_run_that_never_reached_act2_contributes_nothing(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        act_1_only_hp = [(floor, 0) for floor in range(1, 10)]
        metric.accumulate(
            _run_row("run1", False, [("STRIKE_SILENT", 1)], act_1_only_hp)
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 0

    def test_act2_boundary_is_derived_per_run_not_hardcoded(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        # This run's act 2 starts at floor 17, one earlier than the
        # "standard" boundary used elsewhere in this file - a card
        # acquired at floor 17 must still count as present by act 2's
        # start for THIS run.
        shifted_hp_per_floor = [(floor, 0) for floor in range(1, 17)] + [
            (floor, 1) for floor in range(17, 24)
        ]
        metric.accumulate(
            _run_row("run1", True, [("STRIKE_SILENT", 17)], shifted_hp_per_floor)
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1

    def test_multiple_copies_in_one_run_are_independent_samples(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        metric.accumulate(
            _run_row(
                "run1",
                True,
                [("STRIKE_SILENT", 1), ("STRIKE_SILENT", 5)],
                _STANDARD_HP_PER_FLOOR,
            )
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert df.iloc[0]["sample_count"] == 2

    def test_win_rate_across_multiple_runs(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        metric.accumulate(
            _run_row("run1", True, [("STRIKE_SILENT", 1)], _STANDARD_HP_PER_FLOOR)
        )
        metric.accumulate(
            _run_row("run2", False, [("STRIKE_SILENT", 1)], _STANDARD_HP_PER_FLOOR)
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        row = df.iloc[0]
        assert row["win_rate"] == 0.5
        assert row["sample_count"] == 2

    def test_unresolved_card_is_excluded_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        with caplog.at_level(logging.ERROR):
            metric.accumulate(
                _run_row(
                    "run1",
                    True,
                    [("STRIKE_SILENT", 1), ("NOT_A_REAL_CARD", 1)],
                    _STANDARD_HP_PER_FLOOR,
                )
            )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1
        assert any(
            record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
            for record in caplog.records
        )


class TestFinalize:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = CardWinRateAtAct2Metric(binder, output_path=nested_path)
        metric.accumulate(
            _run_row("run1", True, [("STRIKE_SILENT", 1)], _STANDARD_HP_PER_FLOOR)
        )

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_no_cards_seen_writes_empty_file(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([], tmp_path)
        metric = CardWinRateAtAct2Metric(binder, output_path=tmp_path / "out.parquet")

        output_path = metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 0


def test_default_output_path() -> None:
    assert CardWinRateAtAct2Metric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/sts_gg/card_win_rate_at_act2.parquet"
    )
