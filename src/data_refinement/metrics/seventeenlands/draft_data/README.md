# draft_data

Converts 17lands' raw per-pick draft data
(`data/raw/17lands/draft_data/<Set>.<EventType>.csv`) into training-data
parquet files under `data/metrics/seventeenlands/draft_data/`, matching
each CSV's `pack_card_<name>`/`pool_<name>` column suffixes and each
row's `pick` cell value against a `CardBinder` already populated for
`GameId.MTG` (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)).
Every metric here satisfies the shared `Metric[dict]` Protocol
([`../../metric.py`](../../metric.py)).

## Files

- `pack_pool_columns.py` — `DraftCardColumns`: built once per metric
  instance from that metric's own `(card_binder, header, source_game)`.
  Parses every `pack_card_<name>`/`pool_<name>` header column and
  matches its `<name>` suffix against `card_binder` (see "Card-name
  matching" below), exposing the results as `pack_columns`/
  `pool_columns` (`list[tuple[str, UUID]]`) plus `uuid_for_name(name)`
  and `present_uuids(row, columns)` so no per-row string parsing or
  repeat `CardBinder` query is ever needed again for that metric's scan.
- `scanner.py` — `scan_draft_csv(raw_csv_path, metrics, chunk_size)`
  drives every metric in a list over one chunked read pass of a
  draft_data CSV, handing each metric one row at a time and isolating
  one metric's `accumulate()`/`finalize()` failure (logged, not raised)
  from every other metric in the list. Never touches `DraftCardColumns`
  or `CardBinder` itself — each metric already built its own
  `DraftCardColumns` before reaching this function.
- `pack_card_tally_metric.py` — `PackCardTallyMetric` (Template Method,
  abstract): shared per-`(card, key)` running `(times_in_pack,
  times_picked)` tally, driven off every `pack_card_<name)` column
  present (> 0) on an eligible row. A subclass fixes `KEY_COLUMNS`/
  `DEFAULT_OUTPUT_PATH` and implements `_tally_key()`, optionally
  overriding `_is_eligible_row()` (default: every row eligible).
- `pack_card_tally_metrics.py` — three concrete `PackCardTallyMetric`
  subclasses: `CardTakeRateMetric` (`P(picked | in pack, pick_number,
  pack_number)`, `KEY_COLUMNS = ("pack_number", "pick_number")`),
  `FirstPickRateMetric` (`P(pick == card | pack_number == 0,
  pick_number == 0, card in pack)`, `KEY_COLUMNS = ()` — eligibility
  alone does the conditioning), `RankStratifiedTakeRateMetric` (Card
  Take Rate additionally stratified by rank bucket, `KEY_COLUMNS =
  ("pack_number", "pick_number", "rank")`).
- `pick_number_decay_curve_metric.py` — `PickNumberDecayCurveMetric`, a
  fourth `PackCardTallyMetric` subclass (`KEY_COLUMNS =
  ("pick_number",)`) kept in its own file since its `finalize()` is
  overridden entirely: one output row per **card**, carrying
  `take_rate_by_pick_number`/`sample_count_by_pick_number` as parallel
  pyarrow-list columns indexed by `pick_number` bucket (bucket count =
  `max(pick_number) + 1` actually observed across every tallied key,
  never a hardcoded pack size).
- `pack_to_pick_choice_set_metric.py` — `PackToPickChoiceSetMetric`
  (streaming): one output row per input row — `draft_id`,
  `pack_number`, `pick_number`, `pack_option_uuids` (every matched pack
  option present on the row), `pick_uuid` (nullable — null if the
  row's `pick` name doesn't match a card).
- `pool_conditioned_pick_metric.py` — `PoolConditionedPickMetric`
  (streaming), the same shape as `PackToPickChoiceSetMetric` plus one
  extra column: `pool_uuids` (the drafter's already-committed pool,
  read as-is from `pool_<name>` — those columns already exclude the
  row's own pick, so no extra "exclude the current pick" logic is
  needed).
- `BRAINSTORM.md` — candidate metrics from this raw source not yet
  built (this container currently implements six of the seventeen
  ideas catalogued there).

## Card-name matching

`DraftCardColumns._match_uuid()` (private — the only place this policy
lives) matches a bare card name against `card_binder`: an exact
`card_binder.get_by_name(source_game, name)` match first; on 0 or 2+
matches, a regex fallback via
`card_binder.get_by_name_regex(source_game, f"^{re.escape(name)}( //.*)?$")`
(treating `name` as a split/MDFC card's front face, since 17lands'
column-name convention uses only the front face while Scryfall's own
`name` field is `"A // B"`); on 0 or 2+ matches from that fallback, the
name is unmatched — ambiguity is never guessed at. `uuid_for_name()`
caches every lookup (hit or miss) so the same name is never queried
against `card_binder` twice; `unmatched_names` exposes every name this
instance's cache has no uuid for, for a caller to log.

A card's `pack_card_<name>`/`pool_<name>` header suffix and a row's
`pick` cell value are drawn from the same per-set card pool, so
`uuid_for_name()` serves both header-driven and value-driven lookups
through the one cache. An unmatched `pack_card_<name>`/`pool_<name>`
column is simply absent from `pack_columns`/`pool_columns` — it can
never reach a metric's `accumulate()` at all.

## Card-binder access shape

Every metric's constructor takes `(card_binder, header, source_game,
output_path=None)` directly — the same shape every metric in
[`../../sts_gg/`](../../sts_gg/README.md) uses — and builds its own
private `DraftCardColumns` internally via `DraftCardColumns.from_header()`.
Each metric re-parsing the same CSV header independently is a
header-sized cost (hundreds of columns), not a row-sized one, so
paying it once per metric is negligible next to the scan itself. A
driver scanning multiple metrics over the same CSV reads the header
once itself and passes the same `header` object to every metric's
constructor — no shared `DraftCardColumns` instance ever crosses a
metric boundary.

## How it works

`PackCardTallyMetric` and its three `pack_card_tally_metrics.py`
subclasses share one `accumulate()` sequence: look up the row's pick,
iterate every `pack_card_<name>` column present on an eligible row, and
tally `times_in_pack`/`times_picked` per `_tally_key(row, card_uuid)`.
`finalize()` writes one output row per key seen at least once
(`nocab_uuid`, `*KEY_COLUMNS`, `take_rate`, `sample_count`).
`PickNumberDecayCurveMetric` reuses that same `accumulate()`/
`_tally_key()` unchanged but overrides `finalize()` to collect every
tallied `pick_number` bucket into one row per card instead — the same
"parallel lists, one row per card" convention
[`../../sts_gg/card_character_prediction_metric.py`](../../sts_gg/card_character_prediction_metric.py)
established for a similarly-shaped per-card frequency table.

`PackToPickChoiceSetMetric`/`PoolConditionedPickMetric` are streaming
instead — one row already carries a complete example (the full pack
option set, and for the latter, the pool so far, plus which option was
picked), so `accumulate()` writes it immediately via an open
`pyarrow.parquet.ParquetWriter` and `finalize()` only closes that
writer, mirroring
[`../../sts_gg/deck_label_metric.py`](../../sts_gg/deck_label_metric.py)'s
shape.

## How to run

```python
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (
    PickNumberDecayCurveMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (
    PackToPickChoiceSetMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (
    PoolConditionedPickMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.scanner import (
    scan_draft_csv,
)
from src.schema.game_id import GameId

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
raw_csv_path = Path("data/raw/17lands/draft_data/MSH.PremierDraft.csv")
header = pd.read_csv(raw_csv_path, nrows=0).columns

metrics = [
    CardTakeRateMetric(binder, header, GameId.MTG),
    FirstPickRateMetric(binder, header, GameId.MTG),
    RankStratifiedTakeRateMetric(binder, header, GameId.MTG),
    PickNumberDecayCurveMetric(binder, header, GameId.MTG),
    PackToPickChoiceSetMetric(binder, header, GameId.MTG),
    PoolConditionedPickMetric(binder, header, GameId.MTG),
]
scan_draft_csv(raw_csv_path, metrics)
# each metric's DEFAULT_OUTPUT_PATH now exists
```
