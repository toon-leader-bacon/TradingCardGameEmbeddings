from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.isotropic.games.pile_exhaustion_rate_metric import (
    PileExhaustionRateMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_uuid,
    game_header,
    game_header_player,
)


def test_flags_exhausted_kingdom_card(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village"], tmp_path)
    metric = PileExhaustionRateMetric(binder, output_path=tmp_path / "out.parquet")

    header = game_header(
        "a",
        ("Witch", "Village"),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Witches",),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "pile_exhaustion_rate"] == 1.0
    assert df.loc[str(card_uuid(binder, "Village")), "pile_exhaustion_rate"] == 0.0


def test_basic_pile_exhaustion_does_not_match_any_kingdom_card(
    tmp_path: Path,
) -> None:
    binder = binder_from_card_names(["Witch", "Province"], tmp_path)
    metric = PileExhaustionRateMetric(binder, output_path=tmp_path / "out.parquet")

    # Province exhaustion never matches a kingdom card (Province isn't
    # one of the ten kingdom piles) - confirmed live.
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Provinces",),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "pile_exhaustion_rate"] == 0.0


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = PileExhaustionRateMetric(binder, output_path=tmp_path / "out.parquet")

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
    assert PileExhaustionRateMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/pile_exhaustion_rate.parquet"
    )
