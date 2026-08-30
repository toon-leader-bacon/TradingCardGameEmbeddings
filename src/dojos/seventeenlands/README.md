# seventeenlands

`MetricRegressionDojo`, a `Dojo` that regresses a single card's
embedding against a numeric per-card 17lands metric, plus one thin
per-metric subclass for every currently-implemented metric across all
three `src/data_refinement/seventeenlands/` pipelines
(`draft_data_metrics/`, `game_data_metrics/`, `replay_data_metrics/`).
`CardCount(1, 1)`, same arity as `GameClassificationDojo`.
`MetricRegressionDojo` is generic over *which* metric it reads —
`metric_name`/`metrics_path` are constructor parameters, not
hardcoded — so one class is instantiated once per metric rather than
duplicated per metric.

## Directory layout

Mirrors `data_refinement/seventeenlands/`'s own per-source-pipeline
split, so a metrics pipeline and its dojos live at a predictable,
parallel path:

- `draft_data_dojo/` — thin subclasses over `draft_data_metrics/`'s
  metrics: `AveragePickNumberDojo`, `PickSideboardRateDojo`.
- `game_data_dojo/` — thin subclasses over `game_data_metrics/`'s
  metrics: `WinRateDojo`, `DrawnWinRateDojo`, `OpeningHandWinRateDojo`,
  `AverageCopiesWhenIncludedDojo`, `OnPlayWinRateDeltaDojo`,
  `AverageGameLengthWithCardDojo`, `InclusionRateDojo`.
- `replay_data_dojo/` — thin subclasses over `replay_data_metrics/`'s
  currently-implemented metrics: `OpeningHandWinRateDojo`,
  `CandidateHandMulliganRateDojo`, `AverageTurnCastDojo`,
  `AverageTurnsRemainingPostCastDojo`, `OpponentResponseRateDojo`.
  `replay_data_metrics/` also has 6 metrics that are still bare
  placeholders (see that container's own README) — no dojo exists for
  those yet; one gets added once its metric is implemented.

**`OpeningHandWinRateDojo` exists twice, once per directory above** —
`game_data_metrics` and `replay_data_metrics` each have their own,
genuinely distinct `OpeningHandWinRateMetric` (same `metric_name`
string, different classes, different raw source CSVs). The directory
split is what disambiguates the two dojo classes by import path
(`game_data_dojo.opening_hand_win_rate_dojo.OpeningHandWinRateDojo` vs.
`replay_data_dojo.opening_hand_win_rate_dojo.OpeningHandWinRateDojo`) —
the same precedent already set by the two source `Metric` classes
themselves living in different modules under the same name.

See `docs/metric_dojo_inventory.csv` for the full metric-to-dojo
pairing, including each metric's raw/output data locations and a
`label_kind` hint (`count`, `probability`, or `delta`).

## Files

- `metric_regression_dojo.py` — `MetricRegressionDojo`, shared by
  every subclass in all three subdirectories above. `prepare_splits`
  reads a `MetricResult` parquet file
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
- `draft_data_dojo/`, `game_data_dojo/`, `replay_data_dojo/` — one
  file per thin `MetricRegressionDojo` subclass (see "Directory
  layout" above for the full roster). Each exists purely for
  discoverability — collecting a `Dojo` per metric class (see
  `docs/metric_dojo_inventory.csv`) makes the metric↔dojo pairing
  visible without reading constructor arguments. Every one only
  overrides `__init__(expansion, format_code, train_ratio,
  validate_ratio, rng_seed, metrics_path=None, loss=None)`:
  `metric_name` is fixed to the paired metric class's own `.name`
  attribute (never a re-declared string literal, so the pairing can't
  drift), `metrics_path` defaults to that pipeline's own scanner's
  `default_output_path(expansion, format_code)`
  (`DraftMetricScanner`/`MetricScanner`/`ReplayMetricScanner`
  respectively) when not given, and `loss` defaults to `MSELoss()`.
  No subclass declares its own `DEFAULT_OUTPUT_*` — a `Dojo` never
  authors files, only reads them.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.losses.mse_loss import MSELoss
from src.dojos.seventeenlands.metric_regression_dojo import MetricRegressionDojo

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

Any of this container's other metric_name/metrics_path pairs works
the same way — same class, different `(metric_name, metrics_path)`,
e.g. `metric_name="win_rate"` against a `game_data_metrics` output
parquet, or `metric_name="opponent_response_rate"` against a
`replay_data_metrics` output parquet.

The thin per-metric subclasses wrap exactly this, defaulting
`metric_name`/`metrics_path`:

```python
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.seventeenlands.draft_data_dojo.average_pick_number_dojo import AveragePickNumberDojo

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

Every other subclass in `game_data_dojo/`/`replay_data_dojo/` follows
the identical shape, defaulting through its own pipeline's scanner
instead of `DraftMetricScanner` — e.g.
`WinRateDojo(expansion="MSH", format_code="PremierDraft", ...)`
defaults `metrics_path` via `MetricScanner.default_output_path(...)`.
