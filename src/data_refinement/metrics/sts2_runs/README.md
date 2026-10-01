# sts2_runs

Slay the Spire 2 run metrics over the two run sources that share one
raw schema: spire_codex's run export (`data/raw/spire_codex/runs/`,
~1.67M runs in 34 gzip pages) and sts2runs' dump
(`data/raw/sts2runs/*.json.gz`, 6,796 runs). Card references resolve
against the `spire_codex` `CardBinder`. Unlike sts_gg (a leaderboard of
wins only, see [`../sts_gg/README.md`](../sts_gg/README.md)'s "Wins
only"), these runs include losses, so the win and killed-by labels
carry signal here.

The metrics are sts_gg's two families recomputed from this schema: 12
deck-label metrics and 11 per-card averages, each writing to
`data/metrics/sts2_runs/<same stem as sts_gg>.parquet` with its sts_gg
counterpart's label column. No new dojo classes exist for them: the
`sts2_runs.*` keys in `src/training/dojo_catalog.py` reuse the sts_gg
wrappers (`src/dojos/sts_gg/`) with a `metric_output` override.

## Files

- `run_record.py` - the typed run: `Sts2Run` (outcome, killer,
  ascension, floors per act, turns, combats, elites), `PlayerRun` (deck
  uuid, character, relics, damage, card picks/skips, final deck),
  `CardSlot` (card uuid or None, floor added, upgraded), `RunOutcome`.
- `run_parser.py` - `Sts2RunParser`: raw run dict -> `Sts2Run`, the only
  code that reads the raw schema. Derives the stats from
  `map_point_history` (the raw run has no stats block).
- `scanner.py` - `scan_sts2_runs(sources, card_lookup, metrics)`: one
  pass over every source, `scored_run()` (the conditioning policy),
  failure isolation; `default_run_sources()`.
- `deck_label_metric.py` / `deck_label_metrics.py` - `DeckLabelMetric`
  (streaming Template Method: one row per (run, player), `_label_for()`
  varies) and its 12 subclasses: ascension, character, win, killed_by,
  relic count, damage taken, cards picked, cards skipped, turns, elites
  killed, floors cleared, combats.
- `card_average_metric.py` / `card_average_metrics.py` -
  `CardAverageMetric` (accumulation Template Method: per card, the mean
  of `_value_for(run, player, slot)` over its copies) and its 11
  subclasses: sts_gg's nine run-level averages plus upgrade rate and
  act-2 win rate.
- `BRAINSTORM.md` - candidate metrics from this schema.

## How it works

- **Reading** reuses the deck box extraction stages
  (`deck_box/sts2runs/`, `deck_box/spire_codex_runs/`): `raw_files()`
  lists a source's files, `runs()` streams its parsed runs, and
  `deck_uuid(run_id, player_index)` names the deck the stage stored.
- **No deck box is written.** Every (run, player) final deck is already
  in the published `data/final/decks/slay_the_spire_2.db`, so a
  deck-label row's `deck_uuid` points there and the file is stamped
  `requires_deck_box`: the dojo checks that box was built from the same
  CardBinder version. Re-ingesting that box with a new binder version
  means re-running these metrics.
- **Conditioning** (`scored_run()`): only standard-mode, not cheated,
  not abandoned runs are scored. Abandoned runs (~15%) are neither wins
  nor real losses and early quits leave near-starter decks. A player
  whose deck has no aliased card is dropped. A run the parser cannot
  read (a few modded runs per page) is counted as malformed and skipped.
- **Unknown cards**: a card id with no spire_codex alias is a
  `CardSlot` with `card_uuid` None (the box holds the Unknown sentinel
  there). Per-card metrics never tally it; deck-level inputs keep the
  slot, since it is really in the deck.
- **Labels**: run-level labels repeat for every player of a co-op run;
  character, relics, damage and card picks/skips are per player.
  `KilledByMetric` writes losses only, into sts_gg's closed encounter
  vocabulary plus OTHER. `CardWinRateAtAct2Metric` counts copies added
  strictly before act 2's first floor (per run, from the act lengths).
- The 868 runs present in both sources are scored twice (0.05%).

## Per-floor metrics (not built)

Card-reward pick, shop purchase, card removal, upgrade target and rest
site choice (BRAINSTORM #15, #16, #23, #24, #26) need the deck as it
was at a given floor. The parser already walks `map_point_history` per
player; a per-floor metric would add a `FloorSnapshot` record (deck so
far, from `floor_added` plus the floor's `cards_gained`/`cards_removed`,
and the floor's choices) to `run_record.py`, emitted by the parser, and
a new metric base over (run, player, snapshot). Those decks are partial,
so they would go into a metrics-private DeckBox, not the published one.

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source sts2_runs
```

About 60 s per spire_codex page (~50k runs) plus the sts2runs dump, so
roughly 35-40 minutes in all, under 0.5 GB of memory.

```python
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.deck_label_metrics import WinMetric
from src.data_refinement.metrics.sts2_runs.scanner import (
    default_run_sources,
    scan_sts2_runs,
)
from src.schema.game_id import GameId

binder = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
tally = scan_sts2_runs(default_run_sources(), binder, [WinMetric(binder)])
```
