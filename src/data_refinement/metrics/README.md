# metrics

Converts raw per-source data (already ingested into a `CardBinder` -
`../card_binder/`) into per-card/per-deck training-data parquet files
under `data/metrics/<source>/`. Two independent metric shapes live
here, each behind its own structural Protocol - a metric class picks
whichever one matches its actual data source, not a single shape
forced onto both:

- **`Metric[RawRowT]`** (`metric.py`) - `accumulate(row)` /
  `finalize()`, driven per-row by an external scanner over a raw
  source too large to hold in memory at once (e.g. `sts_gg/`'s
  `scan_runs_jsonl` streaming `runs.jsonl`). See
  [`sts_gg/README.md`](sts_gg/README.md) for the fullest example of
  this shape - both its accumulation and streaming metric families.
  Raw-row-driven, so genuinely per-source - not a `generic/` family
  (see below for why).
- **`CorpusScanMetric`** (`generic/corpus_scan_metric.py`) - a single
  `scan() -> Path` call, for a metric whose entire source is already
  fully loaded by the time it runs (a `CardBinder`'s cards, read via
  `CardLookup.all_cards()`) - there's nothing to stream, so no
  `accumulate()`/`finalize()` split is needed.

## `generic/` - source-agnostic metric bases

[`generic/`](generic/README.md) holds two source-agnostic Template
Method bases, each consuming an already-normalized, game-agnostic
input rather than a raw per-source row - which is what makes them
genuinely shareable across data source containers:

- `MaskedFieldMetric` - the `CorpusScanMetric`-family base above,
  reading a `CardLookup`'s already-loaded `GenericCard`s. Consumer:
  `gwent_one/`'s eight masking metrics.
- `DeckCardMaskMetric` - `Metric[dict]`-shaped (not `CorpusScanMetric`
  - it's still driven by a raw per-source row), but delegates all
  row-parsing/card-resolution to an injected per-source
  `DeckExtractionStage` via abstract methods, so the base itself never
  touches raw row structure. Consumer: `play_gwent/`'s
  `LeaderMaskedFromDeckMetric`.

See [`generic/README.md`](generic/README.md) for the full per-class
description.

`sts_gg/`'s `DeckLabelMetric`/`CardAverageMetric` are NOT part of this
directory, even though they're also Template Method bases: they read
directly off a raw per-source row (`row["deck"]`'s shape, card-id
resolution), which genuinely varies by source, so forcing that through
injected hooks would trade a small amount of shared bookkeeping for a
larger amount of configuration surface - judged not worth it.

## Files

- `metric.py` - `Metric[RawRowT]`, the accumulator-family Protocol.
- `hash_utils.py` - content-addressed deck-id hashing shared across
  `Metric[RawRowT]`-family containers (see its own module docstring);
  not used by the `CorpusScanMetric` family, which has no deck concept.
- `generic/` - source-agnostic metric bases, described above.

## Containers

Every raw-source subdirectory here holds that source's own concrete
metric classes; most are still candidate-only (`BRAINSTORM.md`, no
implementation yet). Implemented today:

- **`sts_gg/`** - twenty-one `Metric[dict]` accumulator/streaming
  metrics over Slay the Spire 2 run data. See
  [`sts_gg/README.md`](sts_gg/README.md).
- **`gwent_one/`** - eight `MaskedFieldMetric` masking metrics over
  gwent.one card data - this project's first `CorpusScanMetric`-family
  consumer. See [`gwent_one/README.md`](gwent_one/README.md).
- **`play_gwent/`** - `LeaderMaskedFromDeckMetric`, this project's
  first `DeckCardMaskMetric` consumer, driven directly off
  playgwent.com's community deck guides. See
  [`play_gwent/README.md`](play_gwent/README.md).
- **`seventeenlands/`** - `Metric[dict]` metrics over 17lands' MTG
  draft/game/replay data exports; `draft_data/`, `game_data/`, and
  `replay_data/` are all implemented today. See
  [`seventeenlands/README.md`](seventeenlands/README.md).
- **`dominiontabs/`** - three single-card metrics over Dominion card
  data: `CostRegressionMetric` (this project's first
  `MaskedFieldRegressionMetric` consumer), `TypeMaskMetric`, and
  `SetMaskMetric` (a bespoke `CorpusScanMetric` reading raw
  `cards_db.json` directly, not `raw_content`). See
  [`dominiontabs/README.md`](dominiontabs/README.md).

This file grows as more raw sources get real metric implementations.
