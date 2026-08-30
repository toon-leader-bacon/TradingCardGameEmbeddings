# v2

A second, additive metric-building architecture for the 17lands
pipelines, living alongside `game_data_metrics/`/`draft_data_metrics/`/
`replay_data_metrics/` (the sibling directories one level up) rather
than replacing them. Nothing in those three pipelines is affected by
anything in this directory.

The core difference from the v1 architecture: every `Metric` here
resolves its own card identity and writes its own output file
directly, instead of a shared scanner resolving column identity once
and a shared writer merging every metric's `finalize()` result into
one file. This removes two things the v1 architecture needs that a
metric like "one JSON record per raw row" (as opposed to "one
aggregate scalar per card") can't actually satisfy: a `finalize()`
return type fixed to `dict[UUID, MetricResult]`, and a checkpoint
model whose resumability depends on every metric's state being a
small, cheap-to-snapshot accumulator. v2 metrics have neither
constraint, at the cost of dropping checkpoint/resume entirely — a
failed scan simply reruns from row 0.

## Files

- `metric.py` — `Metric`, the Strategy interface every v2 metric
  implements (`typing.Protocol`, matching every other Strategy
  interface in this project — `CardIngestionStage`, `SingleCardModel`,
  `MultiCardModel`, `Dojo`, `Loss`):
  ```python
  class Metric(Protocol):
      name: str
      DEFAULT_OUTPUT_DIR: ClassVar[Path]
      DEFAULT_OUTPUT_NAME: ClassVar[str]

      def accumulate(self, chunk: pd.DataFrame) -> None: ...
      def finalize(self) -> Path: ...
  ```
  `accumulate()` takes only the raw chunk — no scanner-supplied
  resolved-column argument; a metric that needs card identity resolves
  it itself. `finalize()` writes this metric's own output (in whatever
  shape it wants) and returns the path it wrote to — nothing calling
  it ever inspects what's inside. `DEFAULT_OUTPUT_DIR`/
  `DEFAULT_OUTPUT_NAME` name this metric's conventional output
  location, same `ClassVar` pair plus `default_output_path(...)`
  staticmethod convention used by `CardBinder` and every v1
  `*Scanner` — not itself part of the Protocol, since each concrete
  metric's `default_output_path()` signature is parameterized however
  that metric needs.
- `csv_scanner.py` — `CsvScanner`, a thin driver with no knowledge of
  column resolution, checkpointing, or output writing (every one of
  those is now a `Metric`'s own concern). `CsvScanner(raw_csv_path,
  chunk_size, metrics).scan() -> list[Path]` streams `raw_csv_path` in
  chunks, calls every metric's `accumulate(chunk)` per chunk (so N
  active metrics still cost one pass over the file, not N — the one
  property carried forward from v1), then calls every metric's
  `finalize()` once the file is exhausted, returning each metric's
  reported path in `metrics` order. Warns (does not raise) if a
  metric's `finalize()` reports a path that doesn't actually exist.
- `game_data_metrics/card_column_cache.py` — `CardColumnCache`, a
  small shared component both `DeckOutcomeMetric` and `WinRateMetric`
  hold privately: `card_columns(chunk) -> list[CardColumnSet]` looks
  up `chunk`'s header against a `CardBinder` once, on the first call
  (via the pre-existing, unchanged `column_lookup.find_card_columns()`
  — reused directly, not duplicated), and caches the result for every
  later call regardless of which chunk is passed. `unresolved_column_names`
  exposes any card names the lookup couldn't resolve; a non-empty
  result also emits a `RuntimeWarning` at lookup time.
- `game_data_metrics/deck_outcome_metric.py` — `DeckOutcomeMetric`, a
  streaming metric: for every row of every chunk, reconstructs that
  game's deck (repeating each resolved card's `nocab_uuid` once per
  copy in its `deck_<name>` count — duplicates included) and appends
  one JSON line — `{"deck_card_uuids": [<uuid str>, ...], "won":
  <bool>, "expansion": <str>, "format": <str>}` — to its own output
  file, opened lazily on the first `accumulate()` call. `finalize()`
  closes that file (opening an empty one first if `accumulate()` was
  never called, so the output path always exists) and returns its
  path. An unresolved card name is simply excluded from that game's
  `deck_card_uuids` — it never causes the row itself to be dropped.
- `game_data_metrics/win_rate_metric.py` — `WinRateMetric`, an
  accumulator metric: win rate = fraction of games where a card was in
  the deck that were also won. Its `accumulate()` is a structural port
  of `game_data_metrics/metrics/win_rate/base.py`'s
  `_BinaryTriggerWinRateMetric.accumulate()` (same running
  `self._wins`/`self._games` dicts, same vectorized
  `chunk[deck_column] > 0` / `& chunk["won"]` boolean-mask counting) —
  the only change is reading card columns from a held
  `CardColumnCache` instead of a scanner-supplied list.  `finalize()`
  writes one JSON line per card with at least one observed game
  (`{"nocab_uuid": <uuid str>, "metric_name": "win_rate", "value":
  <float>, "sample_size": <int>, "expansion": <str>, "format": <str>}`)
  and returns its path.

## How it works

`CsvScanner` streams `raw_csv_path` in fixed-size chunks and, for each
chunk, calls `accumulate(chunk)` on every metric in order — one shared
read of the file regardless of how many metrics are attached. Neither
metric here needs a scanner-supplied card-column list: each holds its
own `CardColumnCache`, which resolves once (from whichever chunk
happens to be first) and answers every later call from cache.
`DeckOutcomeMetric` does its entire job inside `accumulate()`, writing
as it goes; `WinRateMetric` only accumulates in `accumulate()` and
does its actual writing once, in `finalize()`. `CsvScanner` doesn't
need to know which shape a given metric is — both satisfy the same
`accumulate()`/`finalize()` contract.

`v1`'s `game_data_metrics/deck_outcome_scanner.py` (unaffected by this
directory, still shipping) is a standalone class rather than a
`Metric` implementation, because neither v1's `finalize() ->
dict[UUID, MetricResult]` per-card return shape nor its checkpoint
model (every metric's state assumed to be a compact, cheap-to-snapshot
accumulator) can express "one independent JSON record per raw row,
with no aggregate to accumulate at all" — the shape `DeckOutcomeMetric`
needs. `WinRateMetric`'s output is diffable against the already-shipped
v1 `WinRateMetric`'s output on the same input CSV — the two produce
identical per-card win rates, verified by this container's own tests.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.deck_outcome_metric import (
    DeckOutcomeMetric,
)
from src.data_refinement.seventeenlands.v2.game_data_metrics.win_rate_metric import WinRateMetric
from src.schema.game_id import GameId

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

deck_outcome_metric = DeckOutcomeMetric(binder, GameId.MTG, "MSH", "PremierDraft")
win_rate_metric = WinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft")

scanner = CsvScanner(
    raw_csv_path=Path("data/raw/17lands/game_data/MSH.PremierDraft.csv"),
    chunk_size=100_000,
    metrics=[deck_outcome_metric, win_rate_metric],
)
output_paths = scanner.scan()
# One shared pass over the CSV drove both metrics; each wrote its own
# file under data/final/metrics/17lands/v2/game/ (a distinct namespace
# from v1's data/final/metrics/17lands/game/ — the two architectures'
# outputs are never guaranteed byte-identical, so they never share a
# path).
```
