"""Project training time for one pass over TRAIN decks -> results/projection.csv.

Usage (from project root):
    python experiments/2026-10-29_throughput_shakeout/analysis/projection.py

Rates are results/runs.csv steady_cards_per_s (TEST included). A
single-card contrastive sample costs 2 cards. Deck sets:
- three_games_train: the TRAIN decks of this experiment's dojos (run.log
  preflight: MTG 5,529,170 + FaB 635,615 + Dominion 425,658);
- all_games_train_estimate: 80% of about 13M decks across every trainable
  game once the Limitless and HearthstoneTopDecks ingests land (deck
  snapshots plus ingestion logs, 2026-10-29; Gwent held out). A rough
  figure, as the human asked for an order of magnitude.
Samples per deck: 1 (one anchor per deck), 23 (about every distinct card
of an MTG deck once: 161.3M deck_cards rows / 6.91M decks,
scale_profile's mtg_deck_card_totals.txt), 60 (the human's figure).
"""

import csv
from pathlib import Path

_RESULTS = Path("experiments/2026-10-29_throughput_shakeout/results")
_DECK_SETS = {
    "three_games_train": 5_529_170 + 635_615 + 425_658,
    "all_games_train_estimate": int(0.8 * 13_000_000),
}
_SAMPLES = (1, 23, 60)
_RATES = ("three_baked_n0", "three_baked_n4", "three_baked_n0_omp1")
_CARDS_PER_SAMPLE = 2


def main() -> None:
    rates = {
        row["run"]: float(row["steady_cards_per_s"])
        for row in csv.DictReader(open(_RESULTS / "runs.csv", encoding="utf-8"))
    }
    rows = []
    for deck_set, decks in _DECK_SETS.items():
        for samples in _SAMPLES:
            for run in _RATES:
                hours = decks * samples * _CARDS_PER_SAMPLE / rates[run] / 3600
                rows.append(
                    {"deck_set": deck_set, "train_decks": decks, "samples_per_deck": samples,
                     "rate_from": run, "cards_per_s": round(rates[run]),
                     "hours": round(hours, 2), "days": round(hours / 24, 2)}
                )
    with open(_RESULTS / "projection.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
