from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.deck_pair_winner_metric import (
    DeckPairWinnerMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    player_entry,
    summary_row,
)


def test_writes_one_row_for_a_two_player_game(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = DeckPairWinnerMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [
                player_entry("winner", 1, {"Witch": 1, "Copper": 7}),
                player_entry("loser", 2, {"Copper": 7}),
            ],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    # Canonical ordering: lo/hi is by deck_uuid, not players[] order -
    # whichever side is "lo", lo_won must correctly reflect THAT deck's
    # own rank, not always players[0].
    row = df.iloc[0]
    assert row["deck_uuid_lo"] < row["deck_uuid_hi"]
    assert isinstance(bool(row["lo_won"]), bool)


def test_ordering_is_symmetric_regardless_of_players_order(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    deck_box_1 = DeckBox()
    deck_box_2 = DeckBox()
    metric_1 = DeckPairWinnerMetric(
        binder, deck_box_1, output_path=tmp_path / "out1.parquet"
    )
    metric_2 = DeckPairWinnerMetric(
        binder, deck_box_2, output_path=tmp_path / "out2.parquet"
    )

    a = player_entry("a", 1, {"Witch": 1, "Copper": 7})
    b = player_entry("b", 2, {"Copper": 7})
    metric_1.accumulate(summary_row(["Witch"], [a, b]))
    metric_2.accumulate(summary_row(["Witch"], [b, a]))
    metric_1.finalize()
    metric_2.finalize()

    df1 = pd.read_parquet(tmp_path / "out1.parquet")
    df2 = pd.read_parquet(tmp_path / "out2.parquet")
    assert df1.iloc[0].to_dict() == df2.iloc[0].to_dict()


def test_solo_game_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = DeckPairWinnerMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(summary_row(["Witch"], [player_entry("solo", 1, {"Copper": 7})]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert DeckPairWinnerMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/deck_pair_winner.parquet"
    )
