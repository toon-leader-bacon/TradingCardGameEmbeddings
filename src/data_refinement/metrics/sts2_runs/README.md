# sts2_runs

Slay the Spire 2 run metrics over spire_codex's run export
(`data/raw/spire_codex/runs/`, ~1.67M runs in gzip pages). A second
source, sts2runs' dump, was dropped when the site died (code in
`archive/sts2runs/`); its raw schema matched. Card references resolve
against the `spire_codex` `CardBinder`. Unlike sts_gg (a leaderboard of
wins only, see [`../sts_gg/README.md`](../sts_gg/README.md)'s "Wins
only"), these runs include losses, so the win and killed-by labels
carry signal here.

The metrics are sts_gg's two families recomputed from this schema: 12
deck-label metrics and 11 per-card averages, plus four per-floor pick
metrics (below). Each of the 23 sts_gg-derived metrics
writes to
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
- `pick_choice.py` - `PickChoice` (a deck, the options, the one taken)
  and `PickKind`; `PlayerRun.pick_choices` holds each kind's choices.
- `player_history.py` - `PlayerHistory`: a raw player's points plus the
  deck timeline. `deck_timeline.py` - `DeckTimeline`: every copy a
  player held, and the deck on arrival at a floor.
- `pick_choice_reader.py` - `PickChoiceReader` (Template Method) and its
  four subclasses, the only code that reads the per-floor raw fields:
  `card_reward_reader.py`, `shop_purchase_reader.py`,
  `card_removal_reader.py`, `card_upgrade_reader.py`.
- `pick_choice_metric.py` - `PickChoiceMetric`, the shared streaming
  writer; its subclasses `card_reward_pick_metric.py`,
  `shop_purchase_pick_metric.py`, `card_removal_pick_metric.py` and
  `card_upgrade_pick_metric.py` set only a `PickKind` and an output path.
- `BRAINSTORM.md` - candidate metrics from this schema.

## How it works

- **Reading** reuses the deck box extraction stages
  (`deck_box/spire_codex_runs/`): `raw_files()`
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

## Per-floor metrics

Four metrics share one row layout (`PickChoiceMetric`): `run_id`,
`deck_uuid`, `floor`, `deck_uuids` (the deck on arrival, one uuid per
copy), `offered_uuids` (the options the pick was made among) and
`picked_uuid` (NULL = declined; only the card reward can be). The deck is
inline, not in a deck box, and only the CardBinder version is stamped. A
dojo splits by `run_id`. An option with no alias voids the row; the
dojos' `Sts2RunPickDojo` base reads all four.

| Metric | One row per | Options | Dojo key |
|---|---|---|---|
| `CardRewardPickMetric` (BRAINSTORM #15) | card-reward room | cards offered, plus a skip | `sts2_runs.card_reward_pick` |
| `ShopPurchasePickMetric` (#16) | card bought | cards still for sale | `sts2_runs.shop_purchase_pick` |
| `CardRemovalPickMetric` (#23) | shop removal | distinct deck cards | `sts2_runs.card_removal_pick` |
| `CardUpgradePickMetric` (#24) | smith at a rest site | distinct deck cards with an un-upgraded copy | `sts2_runs.card_upgrade_pick` |

**Shop purchase is a noisy target.** A purchase turns on gold, prices and
the rest of the basket, none of which is an input, so expect a loss close
to the baseline. It is modelled as a series of draft picks: a shop opens
with 7 card slots; `card_choices` lists those still unsold, `cards_gained`
those bought (in purchase order, assumed). Purchase k picks from the stock
less the k-1 earlier buys, and its deck includes them. Options are sorted
by raw id so their order cannot give the label away. Visits with a
`was_picked` flag or whose stock does not add up to 7 are skipped. Card
removal pays gold too (milder); the upgrade does not.

Removal and upgrade options are the *distinct* deck cards (sorted), so
three Strikes are one option; the deck context shows the copies. A removal
is the shop service only (exactly one removed card); an upgrade is a
`SMITH` rest-site choice that upgraded exactly one card. The upgrade's
options leave out cards whose copies are all upgraded, counted from every
earlier upgrade (an upgraded copy later removed is miscounted). On 3,000
runs, 98% of smith visits and 96% of shop removals became rows.

- **Card reward rooms:** monster, elite and boss rooms that offered cards
  and had at most one pick. Shops (several purchases a visit), events and
  multi-pick rewards are different decisions and write nothing.
- **The deck on arrival** is rebuilt from the final deck plus the copies
  that left it (shop removals, event transforms), each with the floor it
  entered or left on. Checked on 3,000 spire_codex runs: after floor 1,
  every copy gained was later held or recorded as removed. A card is its
  id alone; upgrades (final state only) and enchantments are ignored.
- **Unaliased cards** (the binder lacks some newer cards: about 1.5% of
  rewards offer one) are dropped from the deck, and a reward offering one
  writes no row.
- **Size:** about 29M card-reward rows over the full corpus. The writer streams
  10,000-row groups, so memory stays flat.

Each dojo (`src/dojos/sts2_runs/`) is a `MultiGroupOptionSelectionDojo`
over `[options, deck]`; only the card reward sets `can_skip=True` (a
learned "none" option). The rest-site choice (BRAINSTORM #26) is not
built.

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source sts2_runs
```

About 60 s per spire_codex page (~50k runs) so
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
