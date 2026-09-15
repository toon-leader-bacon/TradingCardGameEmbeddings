"""Runs src/data_refinement/deck_box extraction stages — one at a
time, or all of them — each converting one raw deck source into (or
onto) that game's DeckBox file under data/final/decks/.

Every source here is driven identically through
build_or_update_deck_box(raw_path, extraction_stage, box_path,
card_lookup): see src/data_refinement/deck_box/README.md's "How to
run" section for the same recipe this script turns into runnable
subcommands. Every stage also REQUIRES its game's CardBinder to
already carry the "Unknown" sentinel card (see card_binder/README.md's
"The Unknown sentinel card" section) before extract() runs — this
script seeds it (idempotent, cheap to redo) rather than leaving that as
a separate manual step. Run scripts/run_card_binder_ingestion.py for a
game before running this script against a source of that game.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py --list
    PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py --source sts_gg
    PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py --all
    PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py \\
        --source fabtcg_decklists --raw-path data/raw/fabtcg_decklists/decklists

`--all` runs every source in this process, one after another (same
reasoning as run_card_binder_ingestion.py's own `--all` — see its
module docstring). One source's failure is logged and does not stop
the rest.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.build import build_or_update_deck_box
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.extraction import DeckExtractionStage
from src.data_refinement.deck_box.fabtcg_decklists.extraction_stage import (
    FabtcgDecklistsExtractionStage,
)
from src.data_refinement.deck_box.play_gwent.extraction_stage import (
    PlayGwentDeckExtractionStage,
)
from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.data_refinement.deck_box.sts_gg.extraction_stage import (
    StsGgDeckExtractionStage,
)

# One (extraction_stage, default raw_path) pair per source — the
# default raw_path is each concrete stage's own DEFAULT_RAW_PATH.
_STAGES: dict[str, tuple[DeckExtractionStage, Callable[[], Path]]] = {
    "sts_gg": (
        StsGgDeckExtractionStage(),
        lambda: StsGgDeckExtractionStage.DEFAULT_RAW_PATH,
    ),
    "fabtcg_decklists": (
        FabtcgDecklistsExtractionStage(),
        lambda: FabtcgDecklistsExtractionStage.DEFAULT_RAW_PATH,
    ),
    "play_gwent": (
        PlayGwentDeckExtractionStage(),
        lambda: PlayGwentDeckExtractionStage.DEFAULT_RAW_PATH,
    ),
    "sts2runs": (
        Sts2RunsDeckExtractionStage(),
        lambda: Sts2RunsDeckExtractionStage.DEFAULT_RAW_PATH,
    ),
}


def run_one(name: str, raw_path: Path | None) -> None:
    """Extract one source into its game's DeckBox file, printing how
    many decks were created or content-changed.

    Requires that game's CardBinder to already exist on disk (built via
    run_card_binder_ingestion.py) — raises SystemExit with a pointer to
    that script otherwise, rather than silently extracting against an
    empty binder where every card reference would resolve to Unknown.
    """
    stage, default_raw_path = _STAGES[name]
    resolved_raw_path = raw_path if raw_path is not None else default_raw_path()
    binder_path = CardBinder.default_output_path(stage.SOURCE_GAME)
    if not binder_path.exists():
        raise SystemExit(
            f"{binder_path} does not exist — run "
            f"'python3 scripts/run_card_binder_ingestion.py' for "
            f"{stage.SOURCE_GAME.value} before ingesting decks for it."
        )

    binder = CardBinder.load([binder_path])
    # Bootstrap precondition every DeckExtractionStage requires — see
    # module docstring. Idempotent, so always safe to redo; only
    # actually changes binder_path's content the first time it's seeded
    # for this game.
    binder.ensure_unknown_card(stage.SOURCE_GAME)
    binder.save(binder_path, stage.SOURCE_GAME)

    box_path = DeckBox.default_output_path(stage.SOURCE_GAME)
    print(f"=== {name}: {resolved_raw_path} -> {box_path} ===")
    changed_uuids = build_or_update_deck_box(
        raw_path=resolved_raw_path,
        extraction_stage=stage,
        box_path=box_path,
        card_lookup=binder,
    )
    print(f"created or content-changed {len(changed_uuids)} decks")


def run_all() -> None:
    """Run every source in this process, isolating one source's
    failure (logged via traceback, not raised) from the rest — same
    convention as run_card_binder_ingestion.py's own run_all()."""
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
        help="Extract just this one source into its game's DeckBox.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Override --source's default raw input path (ignored without --source).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract every source, one after another, in this process (the default "
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
