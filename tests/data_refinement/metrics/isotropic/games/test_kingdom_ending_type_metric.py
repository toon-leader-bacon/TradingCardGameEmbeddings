from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.kingdom_ending_type_metric import (
    KingdomEndingTypeMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    game_header,
    game_header_player,
)


def _metric(tmp_path: Path, extra_cards: list[str] | None = None):
    binder = binder_from_card_names(
        ["Witch", "Province", "Colony"] + (extra_cards or []), tmp_path
    )
    deck_box = DeckBox()
    metric = KingdomEndingTypeMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )
    return binder, metric


def test_province_ending(tmp_path: Path) -> None:
    binder, metric = _metric(tmp_path)
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Provinces",),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["ending_type"] == "province"


def test_colony_ending(tmp_path: Path) -> None:
    binder, metric = _metric(tmp_path)
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Colonies",),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["ending_type"] == "colony"


def test_multi_pile_ending(tmp_path: Path) -> None:
    binder, metric = _metric(tmp_path, extra_cards=["Duke", "Monument", "Duchy"])
    header = game_header(
        "a",
        ("Witch", "Duke", "Monument"),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Duke", "Monument", "Duchy"),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["ending_type"] == "multi_pile"


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder, metric = _metric(tmp_path)
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Provinces",),
        is_natural_kingdom=False,
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert KingdomEndingTypeMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/kingdom_ending_type.parquet"
    )
