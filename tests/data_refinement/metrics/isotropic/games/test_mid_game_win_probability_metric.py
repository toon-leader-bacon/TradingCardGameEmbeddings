from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.mid_game_win_probability_metric import (  # noqa: E501
    EventualWinProbabilityMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_quantity,
    game_header,
    game_header_player,
    game_log,
    turn,
)

_NAMES = ["Copper", "Estate", "Silver", "Witch"]


def test_one_row_per_turn_labeled_by_eventual_winner(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    metric = EventualWinProbabilityMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 2), game_header_player("b", 30, 1)),
    )
    log = game_log(
        header,
        (
            turn("a", 1, bought=(card_quantity("Silver"),)),
            turn("b", 1),
            turn("a", 2),
        ),
    )

    metric.accumulate(log)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert list(df["player_wins"]) == [True, False, True]
    assert df["kingdom_uuid"].nunique() == 1


def test_default_output_path() -> None:
    assert EventualWinProbabilityMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/mid_game_win_probability.parquet"
    )
