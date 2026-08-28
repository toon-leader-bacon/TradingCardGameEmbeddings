# nocab_card_embedding

A personal/portfolio project to build a generic card embedding model
that represents a card from any trading card game (MTG, Pokemon,
Yu-Gi-Oh, Hearthstone, etc.) in a shared embedding space, trained via a
suite of pluggable auxiliary tasks ("dojos") rather than any single
task. The embedding model itself is the primary artifact of value —
breadth of dojos/data sources is a deliberate feature, not scope creep.

See [`src/README.md`](src/README.md) for the full container map. This
file is just the entry point: what the project is, and how data moves
through it.

## The two data flows

Everything in this project reduces to two flows, both handing data off
between containers as files on disk rather than in-memory objects:

1. **Card ingestion** — raw data → a `CardIngestionStage` → a
   `CardBinder`, saved to `data/final/cards/<game>.jsonl`. This is the
   one thing every other flow depends on: it's how a raw card, from
   any source, gets this project's own stable identity
   (`nocab_uuid`).
2. **Metric → dojo** — raw data → a metric scanner → a parquet file of
   per-card metric values → a `Dojo`, which reads that parquet file as
   its label source and the `CardBinder` file (flow 1's output) to
   resolve each label back to an actual card. [`docs/metric_dojo_inventory.csv`](docs/metric_dojo_inventory.csv)
   lists every metric ↔ dojo pairing that exists today.

`training/` drives one encoder model against one (or more) `Dojo`s,
never touching either flow's raw data directly — see
[`src/README.md`](src/README.md#how-the-pieces-fit-together) for the
precise class names and [`src/training/README.md`](src/training/README.md)
for how a training run is actually assembled.
