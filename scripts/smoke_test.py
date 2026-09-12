"""End-to-end smoke test of the data pipeline, up to (not including) model training.

Runs the full chain a real training run would go through first, for
whichever sts_gg metric --metric selects:

    1. spire-codex cards.json -> CardBinder (SpireCodexCardIngestionStage,
       same shape as any other source's Phase 1).
    2. sts_gg's runs.jsonl -> that metric's parquet output
       (scan_runs_jsonl, same as compute_seventeenlands_metrics.py's
       spiritual successor would for this game).
    3. That metric's dojo wired up against the parquet output (this
       makes train/test/validation splits under data/splits/ via
       FileManagerParquet, exactly as a real trainer would).
    4. Iterates dojo.training_data()/test_data()/validation_data() like
       a training loop would, but instead of feeding batches to a
       model, just echoes each (card input, label) datum to stdout.

No model is constructed or trained here - this only proves the pipeline
wiring (ingestion -> metric -> dojo -> batches) works end to end.

Two metrics are wired up, the same single-card vs. multi-card duality
this smoke test has always demonstrated (formerly via 17lands'
average_pick_number/pick_prediction - see below):
    card_win_rate - single card in, float label out (P(win | card in
                    final deck), CardWinRateMetric/CardWinRateDojo).
    win           - multi-card (whole final deck) in, bool label out
                    (WinMetric/WinDojo).

RETIRED 17LANDS/MTG CHAIN: this script used to run a Scryfall + 17lands
draft-CSV chain (average_pick_number/pick_prediction metrics against
src.data_refinement.seventeenlands/src.dojos.seventeenlands). The dojo
v2 refactor (commit 1616b6d) removed both those packages without a
replacement, leaving this script importing modules that no longer
exist - sts_gg is the current architecture's most complete game
(src/README.md), so this rewrite moved the smoke test onto it rather
than resurrecting 17lands support.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/smoke_test.py
    PYTHONPATH=. python3 scripts/smoke_test.py --metric win
"""

import argparse
from pathlib import Path
from typing import Generator, Protocol

from src.data_refinement.card_binder.build import build_or_update_card_binder
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.card_average_metrics import CardWinRateMetric
from src.data_refinement.metrics.sts_gg.deck_label_metrics import WinMetric
from src.data_refinement.metrics.sts_gg.scanner import scan_runs_jsonl
from src.dojos.batch import Batch
from src.dojos.sts_gg.card_average_dojos import CardWinRateDojo
from src.dojos.sts_gg.deck_label_dojos import WinDojo


class Dojo(Protocol):
    """Structural shape this script needs from a dojo - just the three
    split generators, since CardWinRateDojo and WinDojo take different
    constructor arguments (WinDojo also needs a DeckBox) and are built
    directly in each run_*() function below rather than through a
    shared factory signature.
    """

    def training_data(self) -> Generator[Batch, None, None]: ...

    def test_data(self) -> Generator[Batch, None, None]: ...

    def validation_data(self) -> Generator[Batch, None, None]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metric",
        choices=["card_win_rate", "win"],
        default="card_win_rate",
        help="Which sts_gg metric (and its dojo) to smoke test.",
    )
    parser.add_argument(
        "--cards-path",
        type=Path,
        default=Path("data/raw/spire_codex/cards.json"),
        help="spire-codex cards.json dump to build the CardBinder from.",
    )
    parser.add_argument(
        "--binder-path",
        type=Path,
        default=Path("data/tmp/smoke_test/cards/slay_the_spire_2.jsonl"),
        help="Where to write the smoke test's own CardBinder.",
    )
    parser.add_argument(
        "--runs-jsonl",
        type=Path,
        default=Path("data/raw/sts_gg/runs.jsonl"),
        help="sts_gg run history to scan.",
    )
    parser.add_argument(
        "--metric-output-dir",
        type=Path,
        default=Path("data/tmp/smoke_test/metrics"),
        help="Where the selected metric writes its parquet output.",
    )
    parser.add_argument(
        "--card-embedding-size",
        type=int,
        default=16,
        help="Dummy embedding size the dojo is configured with (no model is built).",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    return parser.parse_args()


def build_card_binder(cards_path: Path, binder_path: Path) -> CardBinder:
    print(f"[1/4] Building CardBinder from {cards_path} -> {binder_path}")
    build_or_update_card_binder(
        raw_path=cards_path,
        ingestion_stage=SpireCodexCardIngestionStage(),
        binder_path=binder_path,
    )
    return CardBinder.load([binder_path])


def run_card_win_rate(
    card_binder: CardBinder,
    runs_jsonl: Path,
    metric_output_dir: Path,
    card_embedding_size: int,
    rng_seed: int,
) -> None:
    output_path = metric_output_dir / "card_win_rate.parquet"
    print(f"[2/4] Scanning {runs_jsonl} with CardWinRateMetric -> {output_path}")
    metric = CardWinRateMetric(card_binder, output_path=output_path)
    scan_runs_jsonl(runs_jsonl, [metric])

    print("[3/4] Setting up CardWinRateDojo")
    dojo = CardWinRateDojo(
        card_binder=card_binder,
        card_embedding_size=card_embedding_size,
        path_to_training_data=output_path,
        rng_seed=rng_seed,
    )
    echo_dojo_data(dojo)


def run_win(
    card_binder: CardBinder,
    runs_jsonl: Path,
    metric_output_dir: Path,
    card_embedding_size: int,
    rng_seed: int,
) -> None:
    output_path = metric_output_dir / "win.parquet"
    print(f"[2/4] Scanning {runs_jsonl} with WinMetric -> {output_path}")
    # Shared with WinDojo below rather than saved/reloaded - a
    # metric-scan's DeckBox is only ever persisted for real production
    # runs (src/data_refinement/deck_box/README.md), not required here.
    deck_box = DeckBox()
    metric = WinMetric(card_binder, deck_box, output_path=output_path)
    scan_runs_jsonl(runs_jsonl, [metric])

    print("[3/4] Setting up WinDojo")
    dojo = WinDojo(
        card_binder=card_binder,
        deck_box=deck_box,
        card_embedding_size=card_embedding_size,
        path_to_training_data=output_path,
        rng_seed=rng_seed,
    )
    echo_dojo_data(dojo)


METRICS = {
    "card_win_rate": run_card_win_rate,
    "win": run_win,
}


def _describe_card_input(card_input):
    """Render a TrainingInput (a card, a list of cards, or a list of lists
    of cards - see src/schema/type_hints.py) down to just its card names,
    recursing through however many list levels this datum's shape has.
    """
    if isinstance(card_input, list):
        return [_describe_card_input(item) for item in card_input]
    return card_input.name


def echo_dojo_data(dojo: Dojo) -> None:
    print(
        "[4/4] Iterating dojo data (training -> test -> validation), echoing each datum"
    )
    for split_name, data_generator in (
        ("train", dojo.training_data()),
        ("test", dojo.test_data()),
        ("validation", dojo.validation_data()),
    ):
        datum_count = 0
        batch: Batch
        for batch in data_generator:
            for card_input, label in zip(batch.inputs, batch.labels):
                description = _describe_card_input(card_input)
                print(f"      [{split_name}] {description!r} -> {label}")
                datum_count += 1
        print(f"      {split_name}: {datum_count} datums")


def main() -> None:
    args = parse_args()

    card_binder = build_card_binder(args.cards_path, args.binder_path)

    run = METRICS[args.metric]
    run(
        card_binder,
        args.runs_jsonl,
        args.metric_output_dir,
        args.card_embedding_size,
        args.rng_seed,
    )
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
