from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.kingdom_ending_pile_prediction_metric import (  # noqa: E501
    KingdomEndingPilePredictionMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_uuid,
    game_header,
    game_header_player,
)


def test_flags_exhausted_kingdom_card(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomEndingPilePredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    header = game_header(
        "a",
        ("Witch", "Village"),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Witches",),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("card_uuid")
    assert bool(df.loc[str(card_uuid(binder, "Witch")), "exhausted"])
    assert not bool(df.loc[str(card_uuid(binder, "Village")), "exhausted"])
    assert len(df) == 2


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomEndingPilePredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Witches",),
        is_natural_kingdom=False,
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert KingdomEndingPilePredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/kingdom_ending_pile_prediction.parquet"
    )
