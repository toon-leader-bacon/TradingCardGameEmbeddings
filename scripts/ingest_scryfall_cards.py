"""Phase 1 — Scryfall raw dump -> CardBinder.

Ingests a Scryfall oracle-cards .jsonl dump into a CardBinder, writing
the binder's card store plus its sibling alias-ledger file. Safe to
rerun: build_or_update_card_binder() is idempotent, and returns the
nocab_uuid of every card it created or content-changed rather than
succeeding silently.

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
    changed_uuids = build_or_update_card_binder(
        raw_path=args.scryfall_jsonl,
        ingestion_stage=ScryfallCardIngestionStage(),
        binder_path=args.binder_path,
    )
    print(f"created or content-changed {len(changed_uuids)} cards")


if __name__ == "__main__":
    main()
