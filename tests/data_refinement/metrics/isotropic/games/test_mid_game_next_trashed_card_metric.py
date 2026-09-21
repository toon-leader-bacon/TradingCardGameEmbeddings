from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.mid_game_next_trashed_card_metric import (  # noqa: E501
    NextTrashedCardMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_quantity,
    card_uuid,
    game_header,
    game_header_player,
    game_log,
    turn,
)

_NAMES = ["Copper", "Estate", "Silver", "Chapel"]


def test_only_turns_with_a_trash_contribute_a_row(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    metric = NextTrashedCardMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )
    header = game_header("a", ("Chapel",), (game_header_player("a", 40, 2),))
    log = game_log(
        header,
        (
            turn("a", 1, bought=(card_quantity("Chapel"),)),
            turn(
                "a", 2, trashed=(card_quantity("Coppers", 3), card_quantity("Estate"))
            ),
        ),
    )

    metric.accumulate(log)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert len(df) == 1
    assert set(df.iloc[0]["next_trashed_card_uuids"]) == {
        str(card_uuid(binder, "Copper")),
        str(card_uuid(binder, "Estate")),
    }


def test_default_output_path() -> None:
    assert NextTrashedCardMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/mid_game_next_trashed_card.parquet"
    )
