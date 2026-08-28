"""Phase 2 — a saved CardBinder + a raw 17lands game_data CSV -> per-card metrics.

Loads a CardBinder already built by ingest_scryfall_cards.py, streams a
17lands game_data CSV through every registered Metric, writes the
results as parquet, then prints a win-rate leaderboard as a sanity
check. This parquet output is what `training` later joins against the
CardBinder (on nocab_uuid) to build labeled examples — see
src/data_refinement/seventeenlands/game_data_metrics/README.md.

Usage (from the project root):

    python3 scripts/compute_seventeenlands_metrics.py \\
        --binder-path data/final/cards/mtg.jsonl \\
        --game-data-csv "data/tmp/game_data_public.MSH.PremierDraft(1).csv" \\
        --expansion MSH \\
        --format-code PremierDraft

All flags default to the values above (matching what's already on disk
in this checkout), so `python3 scripts/compute_seventeenlands_metrics.py`
with no arguments works as-is — PROVIDED ingest_scryfall_cards.py has
already been run at least once to produce --binder-path. Rerunning is
safe: the metric scan always overwrites its own output.

Writes:
    data/final/metrics/17lands/<expansion>.<format_code>.parquet
    data/final/metrics/17lands/<expansion>.<format_code>.checkpoint.json (transient)

To run:
  cd /Users/nocab/Projects/Personal/MTG_AI_2
  PYTHONPATH=. python3 scripts/compute_seventeenlands_metrics.py
"""

import argparse
from pathlib import Path
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.metrics.average_copies_when_included import (
    AverageCopiesWhenIncludedMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.average_game_length_with_card import (
    AverageGameLengthWithCardMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.inclusion_rate import (
    InclusionRateMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.on_play_win_rate_delta import (
    OnPlayWinRateDeltaMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.drawn_win_rate import (
    DrawnWinRateMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.opening_hand_win_rate import (
    OpeningHandWinRateMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.win_rate import (
    WinRateMetric,
)
from src.data_refinement.seventeenlands.game_data_metrics.metric_scanner import (
    MetricScanner,
)
from src.schema.game_id import GameId


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binder-path",
        type=Path,
        default=Path("data/final/cards/mtg.jsonl"),
        help="CardBinder to resolve card columns against (see ingest_scryfall_cards.py).",
    )
    parser.add_argument(
        "--game-data-csv",
        type=Path,
        default=Path("data/tmp/game_data_public.MSH.PremierDraft(1).csv"),
        help="Raw 17lands game_data CSV to stream.",
    )
    parser.add_argument("--expansion", default="MSH")
    parser.add_argument("--format-code", default="PremierDraft")
    parser.add_argument(
        "--top-n", type=int, default=20, help="How many cards to show in the leaderboard."
    )
    return parser.parse_args()


def print_win_rate_leaderboard(
    binder: CardBinder, output_path: Path, top_n: int, min_sample_size: int = 100
) -> None:
    """Print the top_n cards by win_rate (with at least min_sample_size games)."""
    metrics = pd.read_parquet(output_path)
    win_rate = metrics[metrics["metric_name"] == "win_rate"]
    win_rate = win_rate[win_rate["sample_size"] >= min_sample_size].copy()

    # nocab_uuid values here always came from this same binder's own
    # metric scan, so get_by_uuid() is guaranteed non-None in practice.
    win_rate["name"] = win_rate["nocab_uuid"].apply(
        lambda uuid_str: binder.get_by_uuid(UUID(uuid_str)).name  # type: ignore[union-attr]
    )
    win_rate = win_rate.sort_values("value", ascending=False).head(top_n)

    print(f"Top {top_n} cards by win_rate (min {min_sample_size} games):")
    print(win_rate[["name", "value", "sample_size"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    output_path = Path(
        f"data/final/metrics/17lands/{args.expansion}.{args.format_code}.parquet"
    )
    checkpoint_path = Path(
        f"data/final/metrics/17lands/{args.expansion}.{args.format_code}.checkpoint.json"
    )

    print(f"Loading CardBinder from {args.binder_path}")
    binder = CardBinder.load([args.binder_path])

    print(f"Streaming {args.game_data_csv} (this reads the whole file)")
    exp = args.expansion
    fc = args.format_code
    scanner = MetricScanner(
        raw_csv_path=args.game_data_csv,
        card_binder=binder,
        metrics=[
            WinRateMetric(expansion=exp, format_code=fc),
            DrawnWinRateMetric(expansion=exp, format_code=fc),
            OpeningHandWinRateMetric(expansion=exp, format_code=fc),
            InclusionRateMetric(expansion=exp, format_code=fc),
            AverageCopiesWhenIncludedMetric(expansion=exp, format_code=fc),
            AverageGameLengthWithCardMetric(expansion=exp, format_code=fc),
            OnPlayWinRateDeltaMetric(expansion=exp, format_code=fc),
        ],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        source_game=GameId.MTG,
    )
    result = scanner.scan()

    print(f"Wrote {result.output_path}")
    if result.unresolved_column_names:
        print(
            f"{len(result.unresolved_column_names)} unresolved card names: "
            f"{result.unresolved_column_names}"
        )

    print_win_rate_leaderboard(binder, output_path, args.top_n)


if __name__ == "__main__":
    main()
