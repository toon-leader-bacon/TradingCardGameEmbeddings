from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.isotropic.games.opening_buy_rate_metric import (
    OpeningBuyRateMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_uuid,
    game_header,
    game_header_player,
)


def test_tallies_every_players_opening_not_just_winner(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village", "Silver"], tmp_path)
    metric = OpeningBuyRateMetric(binder, output_path=tmp_path / "out.parquet")

    header = game_header(
        "a",
        ("Witch", "Village"),
        (
            game_header_player("a", 40, 20, ("Witch", "Silver")),
            game_header_player("b", 30, 20, ("Village", None)),
        ),
    )
    metric.accumulate(header)
    metric.finalize()

    # Every kingdom card gets one sample PER PLAYER (2 players here), so
    # a card opened by exactly one of the two players has rate 0.5, not
    # 1.0 - this is what "tallied across every player" means in
    # practice, confirming both players' openings are counted (a
    # winner-only design would only ever see player "a"'s own opening
    # at all, so Village - which only "b" opened - would show 0/1, not
    # 1/2).
    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "opening_buy_rate"] == 0.5
    assert df.loc[str(card_uuid(binder, "Witch")), "sample_count"] == 2
    assert df.loc[str(card_uuid(binder, "Village")), "opening_buy_rate"] == 0.5
    assert df.loc[str(card_uuid(binder, "Village")), "sample_count"] == 2


def test_kingdom_card_never_opened_has_zero_rate(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Moat"], tmp_path)
    metric = OpeningBuyRateMetric(binder, output_path=tmp_path / "out.parquet")

    header = game_header(
        "a",
        ("Witch", "Moat"),
        (game_header_player("a", 40, 20, ("Witch", None)),),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Moat")), "opening_buy_rate"] == 0.0


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = OpeningBuyRateMetric(binder, output_path=tmp_path / "out.parquet")

    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20, ("Witch", None)),),
        is_natural_kingdom=False,
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert OpeningBuyRateMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/opening_buy_rate.parquet"
    )
