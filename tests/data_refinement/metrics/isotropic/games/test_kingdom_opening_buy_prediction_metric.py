from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.kingdom_opening_buy_prediction_metric import (  # noqa: E501
    KingdomOpeningBuyPredictionMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_uuid,
    game_header,
    game_header_player,
)


def test_predicts_winners_opening_only(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Silver", "Moat"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomOpeningBuyPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    header = game_header(
        "winner",
        ("Witch", "Moat"),
        (
            game_header_player("winner", 40, 20, ("Witch", "Silver")),
            game_header_player("loser", 10, 20, ("Moat", None)),
        ),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert set(df.iloc[0]["opening_card_uuids"]) == {
        str(card_uuid(binder, "Witch")),
        str(card_uuid(binder, "Silver")),
    }


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomOpeningBuyPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    header = game_header(
        "winner",
        ("Witch",),
        (game_header_player("winner", 40, 20, ("Witch", None)),),
        is_natural_kingdom=False,
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert KingdomOpeningBuyPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/kingdom_opening_buy_prediction.parquet"
    )
