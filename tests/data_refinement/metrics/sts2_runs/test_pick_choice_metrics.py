import dataclasses
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.metrics.sts2_runs.card_removal_pick_metric import (
    CardRemovalPickMetric,
)
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import (
    CardRewardPickMetric,
)
from src.data_refinement.metrics.sts2_runs.card_upgrade_pick_metric import (
    CardUpgradePickMetric,
)
from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice, PickKind
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import Sts2Run
from src.data_refinement.metrics.sts2_runs.shop_purchase_pick_metric import (
    ShopPurchasePickMetric,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.sts2_runs._runs import binder, raw_run

_METRICS = [
    (CardRewardPickMetric, PickKind.CARD_REWARD),
    (ShopPurchasePickMetric, PickKind.SHOP_PURCHASE),
    (CardRemovalPickMetric, PickKind.CARD_REMOVAL),
    (CardUpgradePickMetric, PickKind.CARD_UPGRADE),
]


def _run_with(tmp_path: Path, kind: PickKind, choice: PickChoice) -> Sts2Run:
    parser = Sts2RunParser(binder(tmp_path), SpireCodexRunsDeckExtractionStage())
    run = parser.parse(raw_run())
    player = run.players[0]
    pick_choices = {k: () for k in PickKind} | {kind: (choice,)}
    return dataclasses.replace(
        run, players=(dataclasses.replace(player, pick_choices=pick_choices),)
    )


def _rows(metric: PickChoiceMetric, run: Sts2Run) -> list[dict]:
    metric.accumulate(run)
    return pq.read_table(metric.finalize()).to_pylist()


@pytest.mark.parametrize("metric_class, kind", _METRICS)
class TestPickChoiceMetrics:
    def test_writes_only_its_own_kind_of_choice(
        self, tmp_path: Path, metric_class: type[PickChoiceMetric], kind: PickKind
    ) -> None:
        run = _run_with(tmp_path, kind, _choice(tmp_path))
        metric = metric_class(binder(tmp_path), tmp_path / "out.parquet")

        (row,) = _rows(metric, run)

        assert row["floor"] == 4
        assert row["picked_uuid"] == row["offered_uuids"][1]
        assert metric_class.KIND is kind

    def test_a_choice_of_another_kind_writes_nothing(
        self, tmp_path: Path, metric_class: type[PickChoiceMetric], kind: PickKind
    ) -> None:
        other = next(k for k in PickKind if k is not kind)
        run = _run_with(tmp_path, other, _choice(tmp_path))
        metric = metric_class(binder(tmp_path), tmp_path / "out.parquet")

        # The raw_run's own parsed rewards are replaced, so nothing is left
        assert _rows(metric, run) == []

    def test_an_unaliased_option_writes_no_row(
        self, tmp_path: Path, metric_class: type[PickChoiceMetric], kind: PickKind
    ) -> None:
        choice = _choice(tmp_path, unaliased_option=True)
        run = _run_with(tmp_path, kind, choice)
        metric = metric_class(binder(tmp_path), tmp_path / "out.parquet")

        assert _rows(metric, run) == []


def _choice(tmp_path: Path, unaliased_option: bool = False) -> PickChoice:
    cards = binder(tmp_path)
    uuids = []
    for name in ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE"):
        card = cards.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, name)
        assert card is not None
        uuids.append(card.nocab_uuid)
    offered = (uuids[0], None if unaliased_option else uuids[1])
    return PickChoice(4, (uuids[2],), offered, picked_index=1)


def test_each_metric_has_its_own_output_file() -> None:
    paths = {metric.DEFAULT_OUTPUT_PATH for metric, _ in _METRICS}

    assert len(paths) == len(_METRICS)
