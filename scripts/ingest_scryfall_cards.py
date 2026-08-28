"""Phase 1 — Scryfall raw dump -> CardBinder.

Ingests a Scryfall oracle-cards .jsonl dump into a CardBinder, writing
the binder's card store plus its sibling alias-ledger file. Safe to
rerun: build_or_update_card_binder() is idempotent, and reports what
actually changed rather than succeeding silently (see IngestSummary).

Usage (from the project root):

    python3 scripts/ingest_scryfall_cards.py \\
        --scryfall-jsonl data/raw/scryfall/oracle-cards-20260820090157.jsonl \\
        --binder-path data/final/cards/mtg.jsonl

Both flags default to the values above (matching what's already on
disk in this checkout), so `python3 scripts/ingest_scryfall_cards.py`
with no arguments works as-is.

Writes:
    <binder-path>                      (e.g. data/final/cards/mtg.jsonl)
    <binder-path's alias-ledger sibling> (e.g. data/final/cards/mtg.alias_ledger.jsonl)

To run:
  PYTHONPATH=. python3 scripts/ingest_scryfall_cards.py
"""

import argparse
from pathlib import Path

from src.data_refinement.card_binder.build import build_or_update_card_binder
from src.data_refinement.card_binder.scryfall.ingestion_stage import (
    ScryfallCardIngestionStage,
)
from src.schema.game_id import GameId


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scryfall-jsonl",
        type=Path,
        default=Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
        help="Scryfall oracle-cards bulk dump to ingest.",
    )
    parser.add_argument(
        "--binder-path",
        type=Path,
        default=Path("data/final/cards/mtg.jsonl"),
        help="Where to write the CardBinder (created if missing, updated if present).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Ingesting {args.scryfall_jsonl} -> {args.binder_path}")
    summary = build_or_update_card_binder(
        raw_path=args.scryfall_jsonl,
        source_game=GameId.MTG,
        ingestion_stage=ScryfallCardIngestionStage(),
        binder_path=args.binder_path,
    )
    print(
        f"inserted={summary.inserted} "
        f"content_updated={summary.content_updated} "
        f"kept_existing={summary.kept_existing}"
    )


if __name__ == "__main__":
    main()
