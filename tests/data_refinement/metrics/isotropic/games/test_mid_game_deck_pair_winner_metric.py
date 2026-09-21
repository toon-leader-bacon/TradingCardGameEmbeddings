from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.mid_game_deck_pair_winner_metric import (  # noqa: E501
    MidGameDeckPairWinnerMetric,
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


def _log(winner: str, first: str = "a", second: str = "b"):
    header = game_header(
        winner,
        ("Witch",),
        (game_header_player(first, 40, 2), game_header_player(second, 30, 2)),
    )
    return game_log(
        header,
        (
            turn(first, 1, bought=(card_quantity("Silver"),)),
            turn(second, 1, bought=(card_quantity("Witch"),)),
            turn(first, 2),
            turn(second, 2),
        ),
    )


def test_skips_identical_starting_decks_and_pairs_later_checkpoints(
    tmp_path: Path,
) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    metric = MidGameDeckPairWinnerMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )

    metric.accumulate(_log("a"))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "o.parquet")
    assert len(df) == 1
    assert df.iloc[0]["deck_uuid_lo"] < df.iloc[0]["deck_uuid_hi"]


def test_pair_row_is_independent_of_players_order(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    out_1, out_2 = tmp_path / "1.parquet", tmp_path / "2.parquet"
    metric_1 = MidGameDeckPairWinnerMetric(binder, DeckBox(), output_path=out_1)
    metric_2 = MidGameDeckPairWinnerMetric(binder, DeckBox(), output_path=out_2)

    # Same game, same decks, same winner - only the header's players[]
    # order (and which nick is "first") differs.
    log_1 = _log("a")
    header_swapped = game_header(
        "a",
        ("Witch",),
        (game_header_player("b", 30, 2), game_header_player("a", 40, 2)),
    )
    log_2 = game_log(header_swapped, log_1.turns)
    metric_1.accumulate(log_1)
    metric_2.accumulate(log_2)
    metric_1.finalize()
    metric_2.finalize()

    pd.testing.assert_frame_equal(pd.read_parquet(out_1), pd.read_parquet(out_2))


def test_lo_wins_label_follows_the_winner(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    out_a, out_b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    metric_a = MidGameDeckPairWinnerMetric(binder, DeckBox(), output_path=out_a)
    metric_b = MidGameDeckPairWinnerMetric(binder, DeckBox(), output_path=out_b)

    metric_a.accumulate(_log("a"))
    metric_b.accumulate(_log("b"))
    metric_a.finalize()
    metric_b.finalize()

    lo_wins_a = pd.read_parquet(out_a).iloc[0]["lo_wins"]
    lo_wins_b = pd.read_parquet(out_b).iloc[0]["lo_wins"]
    assert lo_wins_a != lo_wins_b


def test_skips_non_two_player_games(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    metric = MidGameDeckPairWinnerMetric(
        binder, DeckBox(), output_path=tmp_path / "o.parquet"
    )
    header = game_header("a", ("Witch",), (game_header_player("a", 40, 2),))

    metric.accumulate(game_log(header, (turn("a", 1), turn("a", 2))))
    metric.finalize()

    assert len(pd.read_parquet(tmp_path / "o.parquet")) == 0


def test_default_output_path() -> None:
    assert MidGameDeckPairWinnerMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/mid_game_deck_pair_winner.parquet"
    )
