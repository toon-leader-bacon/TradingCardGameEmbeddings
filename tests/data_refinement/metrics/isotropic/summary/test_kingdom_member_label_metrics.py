from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.kingdom_member_label_metrics import (
    WinningDeckCountMetric,
    WinningDeckMembershipMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def _row() -> dict:
    return summary_row(
        ["Witch", "Village", "Moat"],
        [player_entry("winner", 1, {"Witch": 3, "Copper": 7})],
    )


def test_membership_metric_flags_cards_in_winning_deck(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village", "Moat", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckMembershipMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row())
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("card_uuid")
    assert bool(df.loc[str(card_uuid(binder, "Witch")), "in_winning_deck"])
    assert not bool(df.loc[str(card_uuid(binder, "Village")), "in_winning_deck"])
    assert len(df) == 3


def test_count_metric_reports_copies_in_winning_deck(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village", "Moat", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckCountMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row())
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("card_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "winning_deck_count"] == 3
    assert df.loc[str(card_uuid(binder, "Village")), "winning_deck_count"] == 0


def test_no_resolvable_winner_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckMembershipMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(summary_row(["Witch"], [player_entry("solo", 1, resigned=True)]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0
