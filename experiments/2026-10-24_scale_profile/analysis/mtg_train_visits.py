"""Measure, on a sample of MTG decks, the share of (deck, distinct card)
rows that a TRAIN contrastive pass can anchor: rows of TRAIN-split decks
whose card is a known (in the binder), non-Unknown card of a tier the
TRAIN view shows. Writes the factor (anchorable rows of TRAIN decks /
all rows of TRAIN decks) and the TRAIN deck share to
data/runs/2026-10-24_scale_profile/mtg_train_card_factor.txt, which
profile_report.py reads.

The sample is every deck whose rowid falls in evenly spaced blocks of
the decks table (cheap: no full scan of the 40 GB file).

Usage (from the project root; run it when no training run is going):
    PYTHONPATH=. python experiments/2026-10-24_scale_profile/analysis/mtg_train_visits.py
"""

import sqlite3
from pathlib import Path
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.training.dojo_recipes.card_shelf import CardShelf

DECKS = "data/final/decks/mtg.db"
OUT = Path("data/runs/2026-10-24_scale_profile/mtg_train_card_factor.txt")
BLOCKS = 200
DECKS_PER_BLOCK = 100
# The run's holdout (r2 multi_mixed): seed 0, tiers 8/1/1, deck split default
HOLDOUT = HoldoutSpec(
    seed=0, tier_ratios=(8.0, 1.0, 1.0), held_out_games=frozenset({GameId.GWENT})
)


def main() -> None:
    binder = CardShelf().card_binder(GameId.MTG)
    known = {card.nocab_uuid for card in binder.all_cards(GameId.MTG)}
    unknown = CardBinder.unknown_card_uuid(GameId.MTG)
    visible = HOLDOUT.visible_tiers(Split.TRAIN)
    connection = sqlite3.connect(f"file:{DECKS}?mode=ro", uri=True)
    (max_rowid,) = connection.execute("select max(rowid) from decks").fetchone()
    decks = train_decks = rows = anchorable = 0
    tier_cache: dict[UUID, bool] = {}
    for block in range(BLOCKS):
        first = 1 + block * (max_rowid // BLOCKS)
        deck_ids = [r[0] for r in connection.execute(
            "select nocab_uuid from decks where rowid >= ? limit ?",
            (first, DECKS_PER_BLOCK))]
        for deck_id in deck_ids:
            decks += 1
            if HOLDOUT.deck_split_of(UUID(deck_id), GameId.MTG) is not Split.TRAIN:
                continue
            train_decks += 1
            for (card_id,) in connection.execute(
                    "select card_uuid from deck_cards where deck_uuid = ?", (deck_id,)):
                card = UUID(card_id)
                rows += 1
                if card == unknown or card not in known:
                    continue
                if card not in tier_cache:
                    tier_cache[card] = HOLDOUT.tier_of(card, GameId.MTG) in visible
                anchorable += tier_cache[card]
    factor = anchorable / rows
    OUT.write_text(
        f"{factor:.4f} card factor: anchorable rows / rows of sampled TRAIN decks\n"
        f"{train_decks / decks:.4f} TRAIN deck share of sampled decks\n"
        f"sampled decks {decks}, TRAIN decks {train_decks}, their rows {rows}, "
        f"anchorable {anchorable}\n",
        encoding="utf-8",
    )
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
