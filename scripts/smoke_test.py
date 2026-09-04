"""End-to-end smoke test of the data pipeline, up to (not including) model training.

Runs the full chain a real training run would go through first, for
whichever draft_game_metrics metric --metric selects:

    1. Scryfall oracle-cards dump -> CardBinder (ingest_scryfall_cards.py's
       build_or_update_card_binder, same as Phase 1).
    2. Raw 17lands draft CSV -> that metric's parquet output (streamed in
       chunks, same shape as Phase 2 / compute_seventeenlands_metrics.py).
    3. That metric's dojo wired up against the parquet output (this makes
       train/test/validation splits under data/splits/ via
       FileManagerParquet, exactly as the real trainer would).
    4. Iterates dojo.training_data()/test_data()/validation_data() like a
       training loop would, but instead of feeding batches to a model, just
       echoes each (card input, label) datum to stdout.

No model is constructed or trained here - this only proves the pipeline
wiring (ingestion -> metric -> dojo -> batches) works end to end.

Two metrics are wired up:
    average_pick_number  - single card in, float label out.
    pick_prediction       - multi-group in ([pack cards, pool cards]),
                             int label out (index of the picked card).

The raw draft CSV can be several GB, so every stage here streams it in
chunks rather than loading it into memory at once. --max-rows caps how
many raw draft-CSV rows are scanned, so the smoke test finishes quickly
by default; pass --max-rows 0 to scan the whole file.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/smoke_test.py
    PYTHONPATH=. python3 scripts/smoke_test.py --metric pick_prediction

To run against the full multi-million-row draft CSV instead of the
default capped sample:

    PYTHONPATH=. python3 scripts/smoke_test.py --max-rows 0
"""

import argparse
import glob
from pathlib import Path
from typing import Generator, Protocol, Type

import pandas as pd

from src.data_refinement.card_binder.build import build_or_update_card_binder
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.scryfall.ingestion_stage import (
    ScryfallCardIngestionStage,
)
from src.data_refinement.seventeenlands.draft_game_metrics.average_pick_number_metric import (
    AveragePickNumberMetric,
)
from src.data_refinement.seventeenlands.draft_game_metrics.picked_v_held_v_pack_metric import (
    PickedVHeldVPackMetric,
)
from src.data_refinement.seventeenlands.metric import Metric as MetricInstance
from src.dojos.batch import Batch
from src.dojos.seventeenlands.draft_game_metrics.average_pick_number.dojo import (
    AveragePickNumberDojo,
)
from src.dojos.seventeenlands.draft_game_metrics.pick_prediction.dojo import PickPredictionDojo
from src.schema.game_id import GameId


class Dojo(Protocol):
    """Structural shape every draft_game_metrics dojo satisfies - each is
    constructed from (path_to_training_data, card_binder,
    card_embedding_size, rng_seed) and yields Batch generators. Only the
    subset this script actually drives is declared here.
    """

    def __init__(self, path_to_training_data: Path, card_binder: CardBinder,
                 card_embedding_size: int, rng_seed: int | None = None) -> None: ...

    def training_data(self) -> Generator[Batch, None, None]: ...

    def test_data(self) -> Generator[Batch, None, None]: ...

    def validation_data(self) -> Generator[Batch, None, None]: ...


class MetricClass(Protocol):
    """Constructor shape every draft_game_metrics Metric class satisfies -
    src.data_refinement.seventeenlands.metric.Metric only declares the
    already-constructed instance's shape (accumulate/finalize/name), not
    this.
    """

    def __call__(self, card_binder: CardBinder, expansion: str, format_code: str,
                 output_dir: Path | None = None) -> MetricInstance: ...


# Every metric this smoke test can drive: raw-CSV-scanning Metric class
# paired with the dojo that consumes its parquet output.
METRICS: dict[str, tuple[MetricClass, Type[Dojo]]] = {
    "average_pick_number": (AveragePickNumberMetric, AveragePickNumberDojo),
    "pick_prediction": (PickedVHeldVPackMetric, PickPredictionDojo),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metric", choices=sorted(METRICS), default="average_pick_number",
        help="Which draft_game_metrics metric (and its dojo) to smoke test.",
    )
    parser.add_argument(
        "--scryfall-glob",
        default="data/raw/scryfall/oracle-cards-*.jsonl",
        help="Glob for the Scryfall oracle-cards dump to build the CardBinder from.",
    )
    parser.add_argument(
        "--binder-path",
        type=Path,
        default=Path("data/tmp/smoke_test/cards/mtg.jsonl"),
        help="Where to write the smoke test's own CardBinder.",
    )
    parser.add_argument(
        "--draft-csv",
        type=Path,
        default=Path("data/tmp/draft_data/MSH.PremierDraft.csv"),
        help="Raw 17lands draft-pick CSV to scan.",
    )
    parser.add_argument("--expansion", default="MSH")
    parser.add_argument("--format-code", default="PremierDraft")
    parser.add_argument(
        "--metric-output-dir",
        type=Path,
        default=Path("data/tmp/smoke_test/metrics"),
        help="Where the selected metric writes its parquet output.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50_000,
        help="Rows read from --draft-csv per chunk while scanning.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=200_000,
        help="Cap on how many rows of --draft-csv to scan (0 = the whole file).",
    )
    parser.add_argument(
        "--card-embedding-size", type=int, default=16,
        help="Dummy embedding size the dojo is configured with (no model is built).",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    return parser.parse_args()


def build_card_binder(scryfall_glob: str, binder_path: Path) -> CardBinder:
    matches = sorted(glob.glob(scryfall_glob))
    if not matches:
        raise FileNotFoundError(f"No Scryfall dump matched {scryfall_glob!r}")
    scryfall_path = Path(matches[0])

    print(f"[1/4] Building CardBinder from {scryfall_path} -> {binder_path}")
    build_or_update_card_binder(
        raw_path=scryfall_path,
        source_game=GameId.MTG,
        ingestion_stage=ScryfallCardIngestionStage(),
        binder_path=binder_path,
    )
    return CardBinder.load([binder_path])


def build_metric_data(metric_name: str, metric_cls: MetricClass, card_binder: CardBinder,
                      draft_csv: Path, expansion: str, format_code: str,
                      output_dir: Path, chunk_size: int,
                      max_rows: int) -> Path:
    """Stream draft_csv in chunks through metric_cls, capping the total
    rows scanned at max_rows (0 means the whole file).

    This mirrors CsvScanner.scan()'s own chunk-then-finalize algorithm
    (src/data_refinement/seventeenlands/csv_scanner.py) rather than
    calling it directly, since CsvScanner always scans a CSV to
    completion and a smoke test needs a fast, capped default run.
    """
    row_limit_note = "whole file" if max_rows == 0 else f"capped at {max_rows} rows"
    print(f"[2/4] Scanning {draft_csv} in {chunk_size}-row chunks ({row_limit_note}) "
          f"with {metric_name}")
    metric = metric_cls(card_binder, expansion, format_code, output_dir=output_dir)

    rows_scanned = 0
    for chunk in pd.read_csv(draft_csv, chunksize=chunk_size):
        metric.accumulate(chunk)
        rows_scanned += len(chunk)
        if max_rows > 0 and rows_scanned >= max_rows:
            break

    output_path = metric.finalize()
    print(f"      Scanned {rows_scanned} rows, wrote {output_path}")
    return output_path


def _describe_card_input(card_input):
    """Render a TrainingInput (a card, a list of cards, or a list of lists
    of cards - see src/schema/type_hints.py) down to just its card names,
    recursing through however many list levels this datum's shape has.
    """
    if isinstance(card_input, list):
        return [_describe_card_input(item) for item in card_input]
    return card_input.name


def echo_dojo_data(dojo: Dojo) -> None:
    print("[4/4] Iterating dojo data (training -> test -> validation), echoing each datum")
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
    metric_cls, dojo_cls = METRICS[args.metric]

    card_binder = build_card_binder(args.scryfall_glob, args.binder_path)

    metric_output_path = build_metric_data(
        args.metric, metric_cls, card_binder, args.draft_csv, args.expansion, args.format_code,
        args.metric_output_dir, args.chunk_size, args.max_rows,
    )

    print(f"[3/4] Setting up {dojo_cls.__name__} against {metric_output_path}")
    dojo = dojo_cls(
        path_to_training_data=metric_output_path,
        card_binder=card_binder,
        card_embedding_size=args.card_embedding_size,
        rng_seed=args.rng_seed,
    )

    echo_dojo_data(dojo)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
