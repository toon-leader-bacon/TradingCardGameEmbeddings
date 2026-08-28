# metric_regression

`MetricRegressionDojo`, a `Dojo` that regresses a single card's
embedding against a numeric per-card 17lands draft metric (e.g.
`average_pick_number`, `pick_sideboard_rate` —
`src/data_refinement/seventeenlands/draft_data_metrics/metrics/`).
`CardCount(1, 1)`, same arity as `GameClassificationDojo`. Generic
over *which* metric it reads — `metric_name`/`metrics_path` are
constructor parameters, not hardcoded — so one class is instantiated
once per metric rather than duplicated per metric; the two metrics
above differ only by which `metric_name` to read, not by any
different logic this dojo needs to special-case.

## Files

- `average_pick_number_dojo.py` / `pick_sideboard_rate_dojo.py` —
  `AveragePickNumberDojo` / `PickSideboardRateDojo`, thin
  `MetricRegressionDojo` subclasses that exist purely for
  discoverability — collecting a `Dojo` per metric class (see
  `docs/metric_dojo_inventory.csv`) makes the metric↔dojo pairing
  visible without reading constructor arguments. Each only overrides
  `__init__(expansion, format_code, train_ratio, validate_ratio,
  rng_seed, metrics_path=None, loss=None)`: `metric_name` is fixed to
  the paired metric class's own `.name` attribute (never a
  re-declared string literal, so the pairing can't drift),
  `metrics_path` defaults to
  `DraftMetricScanner.default_output_path(expansion, format_code)`
  when not given, and `loss` defaults to `MSELoss()`. Neither
  subclass declares its own `DEFAULT_OUTPUT_*` — a `Dojo` never
  authors files, only reads them.
- `metric_regression_dojo.py` — `MetricRegressionDojo`.
  `prepare_splits` reads a `MetricResult` parquet file
  (`src/data_refinement/seventeenlands/metric_result.py`, written by
  `write_metric_results`) via `pandas.read_parquet`, filters rows to
  `metric_name`, and builds its label map from the filtered rows —
  **this label map, not the full `CardLookup`, is this dojo's entire
  example population**, since a metrics file is typically a strict
  subset of a full card binder. Raises `ValueError` if any uuid
  present in the metrics file has no card in the given `CardLookup`
  (a metrics-file/corpus version mismatch). Split/cursor mechanics
  (train sampling with replacement, exhaustive reshuffling
  TEST/VALIDATE cursors, `held_out_cards` exclusion from train) are
  otherwise identical to `GameClassificationDojo`'s — not shared via a
  base class, following this project's "rule of three, not
  preemptive" convention for two-instance similarity. `build_decoder_head`
  returns a thin `torch.nn.Linear(embedding_dim, 1)`; `compute_loss`
  squeezes its `(batch_size, 1)` output to `(batch_size,)` via
  `.squeeze(dim=1)` (never a bare `.squeeze()`, which would collapse a
  short final batch of size 1 to a 0-d tensor) before delegating to
  this dojo's injected `Loss` (e.g. `MSELoss`).

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.losses.mse_loss import MSELoss
from src.dojos.metric_regression.metric_regression_dojo import MetricRegressionDojo

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

dojo = MetricRegressionDojo(
    metrics_path=Path("data/final/metrics/17lands/draft/MSH.PremierDraft.parquet"),
    metric_name="average_pick_number",
    train_ratio=0.8,
    validate_ratio=0.1,
    loss=MSELoss(),
    rng_seed=0,
)
dojo.prepare_splits(binder, held_out_cards=set())
batch = dojo.next_batch(Split.TRAIN, batch_size=32)
```

A second instance, `MetricRegressionDojo(metrics_path=..., metric_name="pick_sideboard_rate", ...)`,
regresses against the other metric — same class, same `metrics_path`
(the same parquet file can hold both metrics' rows in one output),
different `metric_name`.

The thin per-metric subclasses wrap exactly this, defaulting
`metric_name`/`metrics_path`:

```python
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.metric_regression.average_pick_number_dojo import AveragePickNumberDojo

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

dojo = AveragePickNumberDojo(
    expansion="MSH",
    format_code="PremierDraft",
    train_ratio=0.8,
    validate_ratio=0.1,
    rng_seed=0,
)  # metrics_path defaults to DraftMetricScanner.default_output_path("MSH", "PremierDraft")
dojo.prepare_splits(binder, held_out_cards=set())
batch = dojo.next_batch(Split.TRAIN, batch_size=32)
```
