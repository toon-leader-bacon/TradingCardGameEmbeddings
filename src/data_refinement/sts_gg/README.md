# sts_gg

Converts sts_gg's raw Slay the Spire 2 run data
(`data/raw/sts_gg/runs.jsonl`) into a per-run metric. Currently a
single file, `deck_outcome_metric.py`.

## Why this container exists

This is this project's first metric built from two independent raw
data sources for the same game: `spire_codex` supplies Slay the Spire
2's card identities into the `CardBinder` (see
[`../card_binder/spire_codex/`](../card_binder/spire_codex/)), and
`sts_gg` supplies run data whose decks reference those same cards
under a different id scheme (`"CARD.<id>"` vs. spire_codex's own bare
`"<id>"`). The point of this container is demonstrating that a second
source's card references resolve through the *same* `CardBinder` a
different source's ingestion stage built — see
`deck_outcome_metric.py`'s own module docstring for the exact
resolution rule (a deterministic prefix strip + alias lookup, not a
fuzzy name match).

`DeckOutcomeMetric` is deliberately a weak metric, not a considered
training signal: it emits one row per run (`run_id`, every resolvable
card in the deck as a `nocab_uuid`, and the run's win/loss) with no
attempt to correct for the obvious confound that a player often
abandons a bad run early rather than playing it out, which likely lets
a model infer a lot of the label just from whether starting
Strikes/Defends are still in the deck. Stronger versions of this idea
(e.g. `P(wins | survived to floor Y)`) are future work.

## Streaming vs. accumulation

`DeckOutcomeMetric` is a *streaming* metric: each run produces its
output row independently of every other run, so `accumulate()` writes
that row immediately (via an already-open
`pyarrow.parquet.ParquetWriter`) instead of buffering and writing
everything in `finalize()` the way an *accumulation* metric would (see
[`../seventeenlands/README.md`](../seventeenlands/README.md)'s
Accumulation vs. Streaming metric builders section for the general
distinction). `finalize()` here is a true no-op relative to data —
it only closes the writer, and is idempotent so a caller and
`scan_runs_jsonl`'s own cleanup can't conflict.

This container does not implement `seventeenlands/metric.py`'s
`Metric` Protocol — that protocol is `CsvScanner`-driven and expects
`pd.DataFrame` chunks; `sts_gg`'s raw data is JSONL, one dict per line,
with no equivalent scanner today.
