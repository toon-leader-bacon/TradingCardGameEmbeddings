"""Runs src/data_refinement/card_binder ingestion stages — one at a
time, or all of them — each converting one raw card dump into (or
onto) that game's CardBinder file under data/final/cards/.

Every source here is driven identically through
build_or_update_card_binder(raw_path, ingestion_stage, binder_path):
see src/data_refinement/card_binder/README.md's "How to run" section
for the same recipes this script turns into runnable subcommands.
Idempotent/resumable the same way every stage's own ingest() is — safe
to rerun, only content-changed cards are reported.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py --list
    PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py --source scryfall
    PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py --all
    PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py \\
        --source spire_codex --raw-path data/raw/spire_codex/cards.json

`--all` runs every source in this process, one after another (unlike
run_data_retrieval.py's `--all`, which spawns a terminal per
downloader for multi-hour crawls — every source here reads data
already on disk, so there's nothing long-running to isolate). One
source's failure is logged and does not stop the rest — see
run_all()'s own docstring.
"""

import argparse
import glob
import sys
import traceback
from pathlib import Path
from typing import Callable

from src.data_refinement.card_binder.build import build_or_update_card_binder
from src.data_refinement.card_binder.cardvault_fabtcg.ingestion_stage import (
    CardVaultFabtcgCardIngestionStage,
)
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.gwent_one.ingestion_stage import (
    GwentOneCardIngestionStage,
)
from src.data_refinement.card_binder.ingestion import CardIngestionStage
from src.data_refinement.card_binder.pokemon_tcg.ingestion_stage import (
    PokemonTcgCardIngestionStage,
)
from src.data_refinement.card_binder.scryfall.ingestion_stage import (
    ScryfallCardIngestionStage,
)
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_retrieval.cardvault_fabtcg.card_downloader import (
    CardVaultFabtcgCardDownloader,
)
from src.data_retrieval.gwent_one.downloader import GwentOneDownloader
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader
from src.data_retrieval.scryfall.downloader import ScryfallOracleDownloader
from src.data_retrieval.spire_codex.card_downloader import SpireCodexCardDownloader


def _latest_scryfall_dump() -> Path:
    """Newest oracle-cards-*.jsonl under ScryfallOracleDownloader's raw
    dir — Scryfall's dump filename is dated and changes every release
    (see run_data_retrieval.py's own note on this), so there is no
    single fixed default the way every other source here has."""
    candidates = sorted(
        glob.glob(
            str(ScryfallOracleDownloader.DEFAULT_RAW_DATA_DIR / "oracle-cards-*.jsonl")
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No oracle-cards-*.jsonl found under "
            f"{ScryfallOracleDownloader.DEFAULT_RAW_DATA_DIR} — run "
            f"scripts/run_data_retrieval.py --source scryfall first."
        )
    return Path(candidates[-1])


# One (ingestion_stage, default raw_path) pair per source — the default
# raw_path matches what run_data_retrieval.py's own downloader for that
# source writes.
_STAGES: dict[str, tuple[CardIngestionStage, Callable[[], Path]]] = {
    "scryfall": (ScryfallCardIngestionStage(), _latest_scryfall_dump),
    "pokemon_tcg": (
        PokemonTcgCardIngestionStage(),
        lambda: PokemonTcgDataDownloader.DEFAULT_RAW_DATA_DIR / "cards",
    ),
    "gwent_one": (
        GwentOneCardIngestionStage(),
        lambda: GwentOneDownloader.DEFAULT_RAW_DATA_DIR,
    ),
    "spire_codex": (
        SpireCodexCardIngestionStage(),
        lambda: SpireCodexCardDownloader.DEFAULT_RAW_DATA_DIR / "cards.json",
    ),
    "cardvault_fabtcg": (
        CardVaultFabtcgCardIngestionStage(),
        lambda: CardVaultFabtcgCardDownloader.DEFAULT_RAW_DATA_DIR
        / "public_card_data.csv",
    ),
}


def run_one(name: str, raw_path: Path | None) -> None:
    """Ingest one source into its game's CardBinder file, printing how
    many cards were created or content-changed."""
    stage, default_raw_path = _STAGES[name]
    resolved_raw_path = raw_path if raw_path is not None else default_raw_path()
    binder_path = CardBinder.default_output_path(stage.SOURCE_GAME)

    print(f"=== {name}: {resolved_raw_path} -> {binder_path} ===")
    changed_uuids = build_or_update_card_binder(
        raw_path=resolved_raw_path,
        ingestion_stage=stage,
        binder_path=binder_path,
    )
    print(f"created or content-changed {len(changed_uuids)} cards")


def run_all() -> None:
    """Run every source in this process, isolating one source's
    failure (logged via traceback, not raised) from the rest — matches
    this project's existing scanner-isolation convention (e.g.
    src/data_refinement/metrics/sts_gg/scanner.py) rather than letting
    one bad/missing raw dump abort every other source's run."""
    failed: list[str] = []
    for name in sorted(_STAGES):
        try:
            run_one(name, raw_path=None)
        except Exception:
            traceback.print_exc()
            failed.append(name)
        print()

    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="List available source names and exit."
    )
    parser.add_argument(
        "--source",
        choices=sorted(_STAGES),
        help="Ingest just this one source into its game's CardBinder.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Override --source's default raw input path (ignored without --source).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest every source, one after another, in this process (the default "
        "when no other flag is given).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for name in sorted(_STAGES):
            print(name)
        return

    if args.source:
        run_one(args.source, raw_path=args.raw_path)
        return

    run_all()


if __name__ == "__main__":
    main()
