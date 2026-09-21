from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.kingdom_game_length_metric import (
    KingdomGameLengthMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    player_entry,
    summary_row,
)


def test_writes_winner_turns_for_natural_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomGameLengthMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(["Witch"], [player_entry("winner", 1, {"Copper": 7}, turns=25)])
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["winner_turns"] == 25


def test_no_resolvable_winner_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomGameLengthMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(summary_row(["Witch"], [player_entry("solo", 1, resigned=True)]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert KingdomGameLengthMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/kingdom_game_length.parquet"
    )
