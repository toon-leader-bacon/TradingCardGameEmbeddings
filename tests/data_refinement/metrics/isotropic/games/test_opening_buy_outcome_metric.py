from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.opening_buy_outcome_metric import (
    OpeningBuyOutcomeMetric,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    game_header,
    game_header_player,
)


def test_writes_one_row_per_player_win_and_loss(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Moat", "Silver"], tmp_path)
    deck_box = DeckBox()
    metric = OpeningBuyOutcomeMetric(
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
    assert len(df) == 2
    assert sorted(df["won"].tolist()) == [False, True]
    # Both rows share the same kingdom_uuid.
    assert df["kingdom_uuid"].nunique() == 1


def test_player_with_no_resolvable_opening_is_skipped(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = OpeningBuyOutcomeMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    header = game_header(
        "winner",
        ("Witch",),
        (
            game_header_player("winner", 40, 20, ("Witch", None)),
            game_header_player("loser", 10, 20, ("Not A Card", None)),
        ),
    )
    metric.accumulate(header)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert df.iloc[0]["won"] == True  # noqa: E712


def test_skips_generator_constrained_kingdom(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = OpeningBuyOutcomeMetric(
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
    assert OpeningBuyOutcomeMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/opening_buy_outcome.parquet"
    )
