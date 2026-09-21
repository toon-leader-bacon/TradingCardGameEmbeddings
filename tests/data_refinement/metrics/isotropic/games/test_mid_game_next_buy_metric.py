from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.mid_game_next_buy_metric import (
    NextBuyPredictionMetric,
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

_NAMES = ["Copper", "Estate", "Silver", "Gold", "Witch"]


def _header():
    return game_header(
        "a",
        ("Witch", "Gold"),
        (game_header_player("a", 40, 2), game_header_player("b", 30, 2)),
    )


def test_writes_one_row_per_turn_with_a_buy(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    deck_box = DeckBox()
    metric = NextBuyPredictionMetric(
        binder, deck_box, output_path=tmp_path / "o.parquet"
    )
    log = game_log(
        _header(),
        (
            turn("a", 1, bought=(card_quantity("Silver"),)),
            turn("b", 1),
            turn("a", 2, bought=(card_quantity("Golds", 2), card_quantity("Witch"))),
        ),
    )

    metric.accumulate(log)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert len(df) == 2
    assert list(df.iloc[0]["next_buy_card_uuids"]) == [str(card_uuid(binder, "Silver"))]
    assert set(df.iloc[1]["next_buy_card_uuids"]) == {
        str(card_uuid(binder, "Gold")),
        str(card_uuid(binder, "Witch")),
    }


def test_partial_deck_grows_and_kingdom_is_shared(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    deck_box = DeckBox()
    metric = NextBuyPredictionMetric(
        binder, deck_box, output_path=tmp_path / "o.parquet"
    )
    log = game_log(
        _header(),
        (
            turn("a", 1, bought=(card_quantity("Silver"),)),
            turn("a", 2, bought=(card_quantity("Gold"),)),
        ),
    )

    metric.accumulate(log)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert df.iloc[0]["partial_deck_uuid"] != df.iloc[1]["partial_deck_uuid"]
    assert df.iloc[0]["kingdom_uuid"] == df.iloc[1]["kingdom_uuid"]


def test_default_output_path() -> None:
    assert NextBuyPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/mid_game_next_buy.parquet"
    )
