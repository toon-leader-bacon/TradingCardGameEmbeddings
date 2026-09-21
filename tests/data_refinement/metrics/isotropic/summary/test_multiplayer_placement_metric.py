from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.multiplayer_placement_metric import (
    MultiplayerPlacementMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    player_entry,
    summary_row,
)


def test_writes_row_for_full_multiplayer_game(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = MultiplayerPlacementMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [
                player_entry("p1", 2, {"Copper": 7}),
                player_entry("p2", 1, {"Witch": 1, "Copper": 7}),
                player_entry("p3", 3, {"Copper": 7}),
            ],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    row = df.iloc[0]
    assert len(row["deck_uuids"]) == 3
    assert len(row["ranks"]) == 3
    assert sorted(row["ranks"]) == [1, 2, 3]
    # deck_uuids must be ascending-sorted (the canonical ordering).
    assert list(row["deck_uuids"]) == sorted(row["deck_uuids"])


def test_two_player_game_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = MultiplayerPlacementMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [player_entry("a", 1, {"Copper": 7}), player_entry("b", 2, {"Copper": 7})],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_any_resignation_discards_the_whole_game(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = MultiplayerPlacementMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [
                player_entry("a", 1, {"Copper": 7}),
                player_entry("b", 2, {"Copper": 7}),
                player_entry("c", 3, resigned=True),
            ],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert MultiplayerPlacementMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/multiplayer_placement.parquet"
    )
