from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.full_deck_win_prediction_metric import (
    FullDeckWinPredictionMetric,
)
from src.schema.game_id import GameId
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    player_entry,
    summary_row,
)


def test_writes_one_row_per_eligible_player_and_fills_deck_box(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = FullDeckWinPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [
                player_entry("winner", 1, {"Witch": 1, "Copper": 7}),
                player_entry("loser", 2, {"Copper": 7}),
                player_entry("resigned", 3, resigned=True),
            ],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 2
    assert df["won"].tolist() == [True, False]
    assert len(list(deck_box.all_decks(GameId.DOMINION))) == 2


def test_default_output_path() -> None:
    assert FullDeckWinPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/full_deck_win_prediction.parquet"
    )
