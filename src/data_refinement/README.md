# data_refinement

Converts raw data written by `data_retrieval` (and `image_processing`,
for image-only sources) into the standardized `GenericCard`/
`GenericDeck` schema (`src/schema/`), and computes derived per-card
metrics from 17lands' raw data. Outputs land under `data/final/`.

Internally, this is expected to be a pipeline of small, individually
typed stages — each stage has an explicit input schema and output
schema, so a stage can be tested and reasoned about in isolation. The
order stages get chained in is a per-experiment/per-source concern,
not something fixed by this container's structure.

## Containers

- **`card_binder/`** *(stable)* — the standardized, multi-game card
  store every other stage (and `training`, downstream) reads from and
  writes into. See [`card_binder/README.md`](card_binder/README.md).
- **`seventeenlands/`** *(in progress)* — everything specific to the
  17lands raw source: three sibling metric-engine pipelines over
  17lands' `draft_data`/`game_data`/`replay_data` CSVs. See
  [`seventeenlands/README.md`](seventeenlands/README.md).
- **`sts_gg/`** *(technology demonstration)* — a single streaming
  metric (`DeckOutcomeMetric`) converting sts_gg's raw Slay the Spire 2
  run data into a per-run deck/outcome row, resolving card references
  through the same `CardBinder` a *different* source (`spire_codex`)
  built — this project's first metric spanning two independent raw
  sources for one game. See [`sts_gg/README.md`](sts_gg/README.md).
- **`fabtcg_decklists/`** — a single streaming metric
  (`DecklistCardsMetric`) converting one fabtcg.com decklist HTML
  fragment (as saved by `data_retrieval/fabtcg_decklists/`) into a
  per-deck `(deck_slug, card_nocab_uuids)` row, resolving each card's
  name against an already-loaded `CardBinder` (populated from
  `cardvault_fabtcg`'s ingestion). Multiple copies of the same card
  appear as repeated uuids in `card_nocab_uuids`, not a count field.
  Structurally a sibling to `sts_gg/` (same streaming-metric-plus-
  directory-scanner shape: `DecklistCardsMetric.accumulate()`/
  `finalize()` plus a separate `scan_decklists_dir()`), not a
  `card_binder/<source>/ingestion_stage.py` — this stage never
  creates or replaces a `GenericCard`, only resolves existing ones.
  This first version deliberately flattens every card group (Hero /
  Weapon / Equipment, Pitch 1/2/3) into one flat list — see
  [`fabtcg_decklists/TODO.md`](fabtcg_decklists/TODO.md) for a
  deferred structured/grouped follow-up.

This file grows as more top-level stages get added.
