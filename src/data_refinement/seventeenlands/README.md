# seventeenlands

Everything specific to the 17lands raw source within `data_refinement`,
organized source-first rather than under one shared cross-source
engine: unlike `card_binder` (a genuinely shared, cross-source
capability), there's no shared store or resolution logic another
17lands data category — or another source entirely — plugs into as-is.
Each of 17lands' three CSV types gets its own sibling pipeline
directory instead.

What *is* genuinely shared across all three pipelines lives directly
in this directory (extracted once all three needed the identical
logic, not duplicated per pipeline):

## Files

- `metric_result.py` — `MetricResult`, the one output-row shape all
  three pipelines write (`nocab_uuid`, `metric_name`, `value`,
  `sample_size`, `expansion`, `format`).
- `scannable_metric.py` — `ScannableMetric`, the Protocol subset
  (`name`/`finalize`/`save_state`/`load_state`) the shared checkpoint
  and writer code below need, independent of each pipeline's own
  `accumulate()` signature.
- `metric_checkpoint.py` — `MetricCheckpoint`: checkpoint restore /
  periodic write / delete-on-completion, shared by all three
  `*MetricScanner`s.
- `metric_scan_loop.py` — `accumulate_over_chunks()`: the shared
  per-chunk streaming/checkpointing loop every `*MetricScanner.scan()`
  composes.
- `metric_writer.py` — `write_metric_results()`: the shared
  finalize/dtype-cast/parquet-write sequence.
- `csv_row_count.py` — `count_data_rows()`: cheap newline-based row
  estimate, used only to size a progress bar (never load-bearing for
  checkpoint correctness).
- `lookup_cache.py` — `LookupCache`, the shared cache-first
  "resolve a key, remember misses" primitive.
- `name_lookup.py` — `find_uuid_by_name()`, the shared exact-match →
  `get_by_name_regex` fallback → ambiguous-is-unresolved policy, used
  by both `game_data_metrics/column_lookup.py` and
  `draft_data_metrics/pick_name_cache.py`.

## Pipelines

- **`game_data_metrics/`** *(stable)* — metric engine over 17lands'
  `game_data` CSVs (one row per game, card identity resolved from
  header columns). See
  [`game_data_metrics/README.md`](game_data_metrics/README.md).
- **`draft_data_metrics/`** *(stable)* — metric engine over 17lands'
  `draft_data` CSVs (one row per pick, card identity resolved from a
  per-row `pick` value). See
  [`draft_data_metrics/README.md`](draft_data_metrics/README.md).
- **`replay_data_metrics/`** *(in progress)* — metric engine over
  17lands' `replay_data` CSVs (one row per game, card identity
  resolved from pipe-delimited Arena IDs). 5 metrics implemented, 6
  still bare placeholders. See
  [`replay_data_metrics/README.md`](replay_data_metrics/README.md).
- **`v2/`** *(in progress)* — a second, additive metric-building
  architecture, living alongside the three pipelines above rather than
  replacing them. Every metric resolves its own card identity and
  writes its own output file directly, instead of going through a
  shared scanner-driven checkpoint/write pipeline. Proven so far by two
  `game_data_metrics`-sourced metrics; none of the metrics above have
  been migrated to it yet. See [`v2/README.md`](v2/README.md).

**Output-path convention:** all three pipelines now write under their
own subdirectory of `data/final/metrics/17lands/` —
`game_data_metrics` → `.../game/<expansion>.<format>.parquet`,
`draft_data_metrics` → `.../draft/<expansion>.<format>.parquet`,
`replay_data_metrics` → `.../replay/<expansion>.<format>.parquet` —
resolving the three-different-conventions inconsistency this section
used to document (bare filename / filename suffix / subdirectory).
Each `*MetricScanner` names its own convention as a
`DEFAULT_OUTPUT_DIR`/`DEFAULT_OUTPUT_NAME` pair plus a
`default_output_path(...)` staticmethod (see each pipeline's own
README) — a recommended default any caller may use, not an enforced
requirement; `output_path` stays a required constructor parameter on
every scanner regardless, and a future pipeline is free to organize
its own output differently as long as its own consumer understands
that convention.
