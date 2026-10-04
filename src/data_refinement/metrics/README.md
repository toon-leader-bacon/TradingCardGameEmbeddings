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
- `HeldOutDeckCardMetric` - `CorpusScanMetric`-shaped, driven from a
  published `DeckBox`: one card held out of a deck, picked among K + 1
  candidates. Consumer: `final_decks/`.
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
- `isolated_call.py` - `call_isolated(logger, subject, step, call)`,
  which runs one metric step and logs (never raises) a failure as an
  ERROR line starting `METRIC FAILURE`, so one metric's bug never stops
  a scan. It is used by the 17lands game_data chunk scanner; the older
  scanners still carry their own copies of this logic.
- `version_metadata.py` - writes/reads the `CardBinder` (and, where
  needed, `DeckBox`) version a metric's parquet output was built from,
  as parquet schema metadata; dojos check it at construction.
- `parquet_builder.py` - `ParquetBuilder`, buffers per-row writes into
  bounded row groups so a metric over a tens-of-millions-row source
  keeps flat memory. `write_columns()` writes a whole batch of rows at
  once (e.g. one 17lands chunk), after any rows already buffered.
- `deck_ids.py` - content-addressed deck-id hashing shared across
  `Metric[RawRowT]`-family containers (see its own module docstring);
  not used by the `CorpusScanMetric` family, which has no deck concept.
- `generic/` - source-agnostic metric bases, described above.

## Containers

Every raw-source subdirectory here holds that source's own concrete
metric classes; most are still candidate-only (`BRAINSTORM.md`, no
implementation yet). Implemented today:

- **`sts_gg/`** - twenty-four `Metric[dict]` accumulator/streaming
  metrics over Slay the Spire 2 run data. See
  [`sts_gg/README.md`](sts_gg/README.md).
- **`sts2_runs/`** - twenty-three `Metric[Sts2Run]` metrics (sts_gg's
  deck-label and per-card families, recomputed) over spire_codex's
  ~1.7M-run export plus sts2runs' dump, which include losses. Its
  deck-level rows point into the published StS2 deck box. See
  [`sts2_runs/README.md`](sts2_runs/README.md).
- **`gwent_one/`** - eight `MaskedFieldMetric` masking metrics over
  gwent.one card data - this project's first `CorpusScanMetric`-family
  consumer. See [`gwent_one/README.md`](gwent_one/README.md).
- **`play_gwent/`** - `LeaderMaskedFromDeckMetric`, this project's
  first `DeckCardMaskMetric` consumer, driven directly off
  playgwent.com's community deck guides. Also two card inclusion-rate
  metrics and guide vote prediction, which read the published Gwent deck
  box read-only. See
  [`play_gwent/README.md`](play_gwent/README.md).
- **`fabtcg_decklists/`** - hero masked from deck (a third
  `DeckCardMaskMetric` consumer) and two card inclusion-rate metrics
  over fabtcg.com tournament decklists, reading the published FaB deck
  box read-only. See [`fabtcg_decklists/README.md`](fabtcg_decklists/README.md).
- **`seventeenlands/`** - vectorized chunk metrics over 17lands' MTG
  draft/game/replay data exports (`draft_data/`, `game_data/`,
  `replay_data/`), each writing one partition per CSV that dojos read
  as slices. See
  [`seventeenlands/README.md`](seventeenlands/README.md).
- **`dominiontabs/`** - three single-card metrics over Dominion card
  data: `CostRegressionMetric` (this project's first
  `MaskedFieldRegressionMetric` consumer), `TypeMaskMetric`, and
  `SetMaskMetric` (a bespoke `CorpusScanMetric` reading raw
  `cards_db.json` directly, not `raw_content`). See
  [`dominiontabs/README.md`](dominiontabs/README.md).
- **`scryfall/`, `pokemon_tcg/`, `cardvault_fabtcg/`, `spire_codex/`,
  `hearthstonejson/`** - single-card masking metrics (`MaskedFieldMetric`,
  `MaskedFieldRegressionMetric`, `MaskedFieldMultiLabelMetric`) over
  each game's `CardBinder`: 6 MTG, 5 Pokemon, 6 Flesh and Blood, 4 Slay
  the Spire 2, 8 Hearthstone. See each directory's `README.md`.
- **`isotropic/summary/`** - thirteen `Metric[dict]` metrics over
  isotropic.org's Wayback-salvaged Dominion gameplay data (real
  played games - kingdoms, vetoes, final decks, outcomes), resolving
  card names against `dominiontabs`' `CardBinder`; this project's
  second `DeckCardMaskMetric` consumer, after `play_gwent/`. See
  [`isotropic/summary/README.md`](isotropic/summary/README.md).
- **`final_decks/`** - one `HeldOutDeckCardMetric` per published deck
  box (Pokemon, FaB, Gwent, Dominion, StS2, MTG), reading
  `data/final/decks/<game>.db` directly. See
  [`final_decks/README.md`](final_decks/README.md).
- **`isotropic/games/`** - eleven metrics over isotropic's other raw
  flavor - turn-by-turn game-log HTML files, parsed via BeautifulSoup and
  sharing `isotropic/card_names.py`'s card-name resolution with
  `isotropic/summary/`. Six `Metric[GameHeader]` header metrics (opening
  buys, pile exhaustion/game-ending type) plus five `Metric[GameLog]`
  mid-game metrics (next buy, next trashed card, next-turn action count,
  eventual win probability, deck-pair winner) built on a full log parser
  and a per-player partial-deck reconstruction. See
  [`isotropic/games/README.md`](isotropic/games/README.md) (and its
  `TODO.md` for the known partial-deck accuracy gap). Resignation metrics
  from this same source are still unbuilt - see
  [`isotropic/BRAINSTORM.md`](isotropic/BRAINSTORM.md).

