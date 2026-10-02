# final_decks

Metrics over the published deck boxes themselves
(`data/final/decks/<game>.db`), not over a raw source. The boxes are
only read. Today this is one family: the held-out card metric,
`HeldOutDeckCardMetric` (see
[`../generic/README.md`](../generic/README.md) for its rows and sampling
rules). `held_out_card_metrics.py` holds one ClassVar-only subclass per
box:

| Metric | Box | Decks sampled | Targets per deck | Output |
|---|---|---|---|---|
| `PokemonHeldOutCardMetric` | `pokemon.db` (188 decks) | all | 8 | `data/metrics/final_decks/held_out_card_pokemon.parquet` |
| `FleshAndBloodHeldOutCardMetric` | `flesh_and_blood.db` (4.2k) | all | 8 | `.../held_out_card_flesh_and_blood.parquet` |
| `GwentHeldOutCardMetric` | `gwent.db` (60k) | all | 4 | `.../held_out_card_gwent.parquet` |
| `DominionHeldOutCardMetric` | `dominion.db` (532k) | all | 1 | `.../held_out_card_dominion.parquet` |
| `SlayTheSpire2HeldOutCardMetric` | `slay_the_spire_2.db` (2.8M) | 500k | 1 | `.../held_out_card_slay_the_spire_2.parquet` |
| `MtgHeldOutCardMetric` | `mtg.db` (4.8M) | 500k | 1 | `.../held_out_card_mtg.parquet` |

Every metric uses 7 decoys (8 candidates), half of them co-occurrence
decoys, with staple threshold 0.05 and seed 0. Each row points at a
deck in the published box, so a re-ingested box makes these outputs
stale: `scan()` refuses a box whose stamped binder version differs from
the binder's, and the dojo's version check refuses a stale pairing.

Decoys match the targets' frequency only approximately (targets are
weighted within each deck, decoys across the sample). On the subset
outputs, an input-ignoring "pick the most frequent candidate" predictor
scores 0.14 (Gwent), 0.16 (FaB) and 0.17 (Pokemon) against a chance
level of 0.125.

Pokemon, FaB and Gwent take more than one target per deck, so their
dojos split by `deck_uuid` rather than by row: all of a deck's rows land
in one split, and a deck never shows up in both TRAIN and TEST.

Dojos: `src/dojos/final_decks/held_out_card_dojos.py`, catalog keys
`final_decks.held_out_card_<game>`.

## How to run

One box per command; these families run only with `--source` (not
`--all`), after deck-box ingestion and after `play_gwent`, which writes
into the Gwent box. Smallest box first:

```
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_pokemon
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_flesh_and_blood
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_gwent
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_dominion
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_slay_the_spire_2
PYTHONPATH=. python3 scripts/run_metrics.py --source final_decks_mtg
```

Measured on subsets (2026-10-01): Pokemon (all decks) 2 s, FaB (all)
8 s, Gwent (all) 50 s with under 0.7 GB peak working set; 20k-deck
samples of Dominion, StS2 and MTG take 18-26 s each. Reading decks one
by one from SQLite dominates (about 0.3-0.75 ms per deck), so a full
Dominion run takes roughly 5-8 minutes, and StS2 or MTG (500k decks)
roughly 8-15 minutes each, under about 1.5 GB.
