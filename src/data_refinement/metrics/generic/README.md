# generic

Source-agnostic metric bases: every class here consumes an already-
normalized, game-agnostic input (a `GenericCard` via `CardLookup`, or
a `DeckBox`) rather than a raw per-source row, so a per-source package
only supplies a Strategy hook (which field/card to target, what label
to assign) - no per-source raw-parsing lives here. This is what makes
these bases genuinely shareable across data source containers, unlike
`../metric.py`'s `Metric[RawRowT]` family (e.g. `sts_gg/deck_label_metric.py`,
`sts_gg/card_average_metric.py`): those operate directly on a raw
per-source row (a JSONL dict, a CSV chunk), which genuinely varies
shape by source, so they're deliberately NOT part of this directory -
see `../README.md`.

## `CorpusScanMetric` (`corpus_scan_metric.py`)

The shared Protocol both classes below satisfy structurally: a single
`scan() -> Path` call, for a metric whose entire source is already
fully loaded by the time it runs (a `CardBinder`'s cards, read via
`CardLookup.all_cards()`) - there's nothing to stream, so no
`accumulate()`/`finalize()` split is needed.

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

First (and currently only) consumer: [`../gwent_one/README.md`](../gwent_one/README.md)'s
eight masking metrics.

## `MaskedFieldRegressionMetric` (`masked_field_regression_metric.py`)

`MaskedFieldMetric`'s regression-flavored twin: same `scan()` sequence
and same `_is_eligible()` philosophy, but `_label_for_card()` becomes
`_value_for_card(card) -> float` and there's no `LABEL_VALUES`/
`OTHER_LABEL` - a fixed-set label vocabulary with a catch-all bucket is
a classification-only concept with no regression equivalent. A
subclass's only tool for excluding a card whose field value can't be
treated as a number is `_is_eligible()` itself. Output columns:
`nocab_uuid: str`, `masked_field: list[str]`, `label: float`.

Deliberately a separate class from `MaskedFieldMetric` rather than one
generic base parameterized over label dtype - unifying them would need
to thread the OTHER-bucket concept through a regression path that has
no use for it.

First (and currently only) consumer: [`../dominiontabs/README.md`](../dominiontabs/README.md)'s
`CostRegressionMetric`.

## `DeckCardMaskMetric` (`deck_card_mask_metric.py`)

A `Metric[dict]`-shaped Template Method base (mirrors `sts_gg`'s
accumulator/streaming shape, not `CorpusScanMetric`) for masking one
whole *card* out of a *deck*, chosen via game-specific logic - distinct
from `MaskedFieldMetric`'s "mask one field of one card" shape. Each
metric's `accumulate(row)` is driven directly off a raw source's rows
(e.g. `../play_gwent/`'s guide objects), never off an already-built
`DeckBox` - `GenericDeck`/`DeckBox` are deliberately game-agnostic
(a plain card_nocab_uuids multiset, no role/slot information), so a
stored deck alone can't answer a game-specific question like "which
card was the leader." A subclass fixes `_deck_uuid_for_row()` (ensures
the row's deck exists in a `DeckBox`, as a side effect, by delegating
to the game's own `DeckExtractionStage`), `_target_card_uuid_for_row()`
(the game-aware masking target, read directly off the raw row), and
`_label_for_card()` (per-metric label semantics - deliberately not
standardized the way `MaskedFieldMetric`'s `MASKED_FIELD` path is).
Output columns: `deck_uuid: str`, `target_card_uuid: str`, `label: str`.

First (and currently only) consumer: [`../play_gwent/README.md`](../play_gwent/README.md)'s
`LeaderMaskedFromDeckMetric`.

## Files

- `corpus_scan_metric.py` - `CorpusScanMetric`, the shared Protocol.
- `masked_field_metric.py` - `MaskedFieldMetric`, described above.
- `masked_field_regression_metric.py` - `MaskedFieldRegressionMetric`,
  described above.
- `deck_card_mask_metric.py` - `DeckCardMaskMetric`, described above.
