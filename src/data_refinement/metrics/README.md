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
- **`CorpusScanMetric`** (`corpus_scan_metric.py`) - a single
  `scan() -> Path` call, for a metric whose entire source is already
  fully loaded by the time it runs (a `CardBinder`'s cards, read via
  `CardLookup.all_cards()`) - there's nothing to stream, so no
  `accumulate()`/`finalize()` split is needed. `masked_field_metric.py`
  builds on this - see below.

## `MaskedFieldMetric` (`masked_field_metric.py`)

A Template Method base for masking metrics: one output row per
(eligible card, masked field) - the true value of a field a `Dojo`
will later mask out of that card's `raw_content` and ask a model to
predict. A subclass fixes:

- `SOURCE_GAME`/`MASKED_FIELD` (a path into `raw_content`, e.g.
  `["faction"]`)/`LABEL_VALUES` (this metric's fixed, ordered label
  vocabulary)/`DEFAULT_OUTPUT_PATH` as `ClassVar`s.
- `_is_eligible(card)` - THE ELIGIBILITY DECISION LIVES HERE, not left
  to a `Dojo` to discover defensively at read time: whether a card is
  even a valid sample for this field (e.g. a power-value metric
  restricting to unit-type cards). Defaults to "every card is
  eligible"; only overridden where a real filter applies.
- `_label_for_card(card)` - the true label for an already-eligible
  card, expected to route through `_raw_field_value(card)` (walks
  `MASKED_FIELD` into `raw_content`) and, for a numeric field,
  `_label_or_other(raw_value)` (encodes to a classification label,
  falling back to the shared `OTHER_LABEL` for a value outside
  `LABEL_VALUES`).

`scan()` owns the shared sequence (iterate `card_lookup.all_cards()` →
filter via `_is_eligible()` → build a `_MaskedFieldRow` → write one
parquet file with columns `nocab_uuid: str`, `masked_field: list[str]`
(a native pyarrow list column, not a JSON-encoded string), `label:
str`) - a subclass only ever fixes the two decisions above, never
repeats the sequence itself.

Deliberately NOT a `Metric[RawRowT]`: a `CardBinder` is already fully
loaded by `scan()` time, so there's no external per-row source to
`accumulate()` over, and no shared scanner module plays
`sts_gg/scanner.py`'s role here - that module exists specifically to
avoid re-reading one large external file once per metric, and
`CardLookup.all_cards()` has no comparable I/O cost to amortize.

First (and currently only) consumer: [`gwent_one/README.md`](gwent_one/README.md)'s
eight masking metrics.

## `DeckCardMaskMetric` (`deck_card_mask_metric.py`)

A `Metric[dict]`-shaped Template Method base (mirrors `sts_gg`'s
accumulator/streaming shape, not `CorpusScanMetric`) for masking one
whole *card* out of a *deck*, chosen via game-specific logic - distinct
from `MaskedFieldMetric`'s "mask one field of one card" shape. Each
metric's `accumulate(row)` is driven directly off a raw source's rows
(e.g. `play_gwent/`'s guide objects), never off an already-built
`DeckBox` - `GenericDeck`/`DeckBox` are deliberately game-agnostic
(a plain card multiset, no role/slot info), so a stored deck alone
can't answer a game-specific question like "which card was the
leader." A subclass fixes `_deck_uuid_for_row()` (ensures the row's
deck exists in a `DeckBox`, as a side effect, by delegating to the
game's own `DeckExtractionStage`), `_target_card_uuid_for_row()` (the
game-aware masking target, read directly off the raw row), and
`_label_for_card()` (per-metric label semantics - deliberately not
standardized the way `MaskedFieldMetric`'s `MASKED_FIELD` path is).
Output columns: `deck_uuid: str`, `target_card_uuid: str`, `label: str`.

First (and currently only) consumer: [`play_gwent/README.md`](play_gwent/README.md)'s
`LeaderMaskedFromDeckMetric`.

## Files

- `metric.py` - `Metric[RawRowT]`, the accumulator-family Protocol.
- `corpus_scan_metric.py` - `CorpusScanMetric`, the corpus-scan-family
  Protocol.
- `masked_field_metric.py` - `MaskedFieldMetric`, described above.
- `deck_card_mask_metric.py` - `DeckCardMaskMetric`, described above.
- `hash_utils.py` - content-addressed deck-id hashing shared across
  `Metric[RawRowT]`-family containers (see its own module docstring);
  not used by the `CorpusScanMetric` family, which has no deck concept.

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

This file grows as more raw sources get real metric implementations.
