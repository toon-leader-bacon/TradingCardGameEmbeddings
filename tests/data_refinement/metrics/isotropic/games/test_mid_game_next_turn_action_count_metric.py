from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.mid_game_next_turn_action_count_metric import (  # noqa: E501
    NextTurnActionCountMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_quantity,
    game_header,
    game_header_player,
    game_log,
    turn,
)

_NAMES = ["Copper", "Estate", "Silver", "Village", "Smithy"]
_TYPES = {"Copper": ["Treasure"], "Silver": ["Treasure"], "Estate": ["Victory"]}


def test_counts_only_action_cards_played_on_the_next_own_turn(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path, card_types=_TYPES)
    metric = NextTurnActionCountMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )
    header = game_header(
        "a",
        ("Village",),
        (game_header_player("a", 40, 2), game_header_player("b", 30, 2)),
    )
    log = game_log(
        header,
        (
            turn("a", 1),
            turn("b", 1, played=(card_quantity("Smithy"),)),
            turn(
                "a",
                2,
                played=(
                    card_quantity("Village", 2),
                    card_quantity("Smithy"),
                    card_quantity("Coppers", 4),
                ),
            ),
        ),
    )

    metric.accumulate(log)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert list(df["next_turn_action_count"]) == [3]


def test_last_turn_of_a_player_has_no_row(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path, card_types=_TYPES)
    metric = NextTurnActionCountMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )
    header = game_header("a", ("Village",), (game_header_player("a", 40, 1),))

    metric.accumulate(game_log(header, (turn("a", 1),)))
    metric.finalize()

    assert len(pd.read_parquet(tmp_path / "o.parquet")) == 0


def test_default_output_path() -> None:
    assert NextTurnActionCountMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/mid_game_next_turn_action_count.parquet"
    )
