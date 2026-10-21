# diet

Decides which dojo `Trainer` trains on next, and when a dojo (or a whole
phase) is finished.

## Files

- [diet_sampler.py](diet_sampler.py): `DietSampler` Protocol and the
  temperature-weighted sampler that covers proportional, uniform and
  temperature diets. `diet_sampler_for(rule)` builds the right sampler
  for any `DietRule`.
- [table_diet.py](table_diet.py): `TableDietSampler`, the sampler for a
  `TableDiet` (a nested drop table of dojos), and `build_dojo_drop_table`,
  which turns a table into a `DropTable` over the dojos that can be drawn.
- [count_weighting.py](count_weighting.py): TRAIN-count weighting shared by
  both samplers (`TrainCountCache`, `alpha_of`, `count_weight`).
- [diet_shares.py](diet_shares.py): each phase's expected share of steps
  per dojo (`plan_diet_shares`), printed by `scripts/run_training.py` (also
  under `--check`) and written to the run directory as `diet_shares.csv`.
- [saturation_tracker.py](saturation_tracker.py): per-dojo ACTIVE/SATURATED
  state driven by each round's normalized TEST loss (loss / baseline);
  reports which dojos are still active and when the phase is done.
- [dojo_batch_stream.py](dojo_batch_stream.py): turns a dojo's finite TRAIN
  pass into an endless batch supply.
- [round_schedule.py](round_schedule.py): `RoundSchedule`, the dojo name
  of every step of a round, and `draw_round_schedule(sampler, active,
  steps, rng)`, which makes the round's sampler draws up front (the same
  draws, in the same order, as one per step). Also `require_next_step`,
  the check both feeds make in `next_batch`.
- [batch_feed_errors.py](batch_feed_errors.py): `BatchFeedError` and its
  subclasses `ScheduleMismatchError` (a call out of schedule order) and
  `PrefetchStoppedError` (a worker that will not build). Neither is ever
  a dojo's fault.
- [batch_feed.py](batch_feed.py): `BatchFeed`, where the Trainer takes a
  phase's TRAIN batches from, one round's schedule at a time
  (`begin_round`, `next_batch(name)`, `reschedule`, `end_round`,
  `close`). `open_batch_feed` returns a `DirectBatchFeed` (builds on the
  training thread, the default) at `HardwareLimits.prefetch_depth` 0, or
  a started `PrefetchingBatchStreams` above it.
- [prefetching_batch_streams.py](prefetching_batch_streams.py):
  `PrefetchingBatchStreams` runs one worker thread that builds the
  round's scheduled steps, at most `depth` ahead of the trainer, and
  reserves each batch's cards in the model's feature cache
  (`BatchWarmer`: `build_model_warmer`, or `NO_BATCH_WARMING` without a
  cache). See "Prefetch" below.
- [prefetch_example.md](prefetch_example.md): one cold-started round,
  step by step, with what each cache tier holds.
- [dojo_fault_ledger.py](dojo_fault_ledger.py): counts consecutive step
  failures per dojo and overall; quarantines a dojo that keeps failing (in a row or in total) and
  signals when the whole run should stop.

## How it works

Each round, `Trainer` asks the tracker for the ACTIVE dojos and draws the
round's schedule from the sampler. It hands the schedule to the phase's
batch feed (`begin_round`), then takes each step's batch in schedule
order. After the round it ends the feed's round, and the normalized TEST
losses go back into the tracker.

When a step's failure quarantines a dojo, the trainer draws the rest of
the round again from the dojos still active and calls `reschedule`. The
redraw uses the same `rng`, so a run with a quarantine draws differently
from one without; a run without one draws exactly as one step at a time.

### Prefetch

Decoupled access/execute: the trainer decides the order of work, the
worker runs ahead through it, and the feature cache holds what the
worker fetched until it is used.

- **The worker follows the schedule.** It builds the next unbuilt step
  while fewer than `depth` built steps wait, and never past the round's
  last step. So it is idle while the trainer evaluates, and at the end
  of a round that ran to the end, every dojo's stream has advanced by
  exactly the batches the round used, at any depth. Runs are therefore
  reproducible across depths, including depth 0, except after a
  mid-round quarantine in a dojo that draws TRAIN and TEST from one RNG
  (OddOneOut).
- **Each built batch is reserved.** `BatchWarmer.reserve` pins its cards
  in the cache's memory tiers, then warms them, on the worker's own CUDA
  stream. `next_batch` releases the previous step's reservation before
  it returns or raises: by then that step has gathered its own copy of
  its rows. `end_round` and `close` release everything else. A
  reservation that fails to warm is counted (`warm_failures`, logged)
  and the batch still queued; its cards are encoded on the training
  thread. The model warmer warns once when pinned rows fill a tier
  (lower `prefetch_depth` or grow the tier).
- **Leftovers.** A `reschedule` keeps the built (and in-flight) steps
  that match the start of the new schedule. The rest go to their dojo's
  leftover pile, released and ordered by their place in the stream, as
  does anything still unused at `end_round`. When that dojo comes up
  again, the worker reserves its oldest leftover again instead of
  building, so no stream skips a batch. A quarantined dojo's leftovers
  are dropped at `close`.
- **Errors.** A stream's error is queued in its step's place and
  re-raised to the trainer in order, with its own type. A call out of
  schedule order raises `ScheduleMismatchError`, and a stopped or dead
  worker `PrefetchStoppedError`.
- **Threads.** Only the worker advances a stream, so each dojo keeps its
  batch order. A dojo's `compute_loss` runs on the training thread while
  the worker builds its next batch, so it must read nothing that
  `batches(TRAIN)` advances. `GenericDojo` and `ContrastiveDojo` read
  only their loss object and head.

```python
feed = open_batch_feed(streams, build_model_warmer(model, "fp16"), depth=4)
feed.begin_round(draw_round_schedule(sampler, active, steps, rng))
for name in schedule.dojo_names:
    batch = feed.next_batch(name)    # releases the previous step's pins
    ...
feed.end_round()                      # worker idle, every pin released
feed.close()
```

## Diet rules

A phase's `diet:` is one of:

- `{rule: uniform}`, `{rule: proportional}`, `{rule: temperature, alpha: a}`:
  each active dojo weighted by its TRAIN example count ** alpha (0, 1, a).
- `{rule: table, table: [<rows>]}`: a nested weighted table
  (`src/utils/drop_table.py`). Each row has a `weight` and exactly one of:
  - `dojo: <name>`: one dojo;
  - `dojos: [<patterns>]`: every run dojo matching any fnmatch pattern
    (`gwent_one.*`), split by `within:` (a flat rule above, default
    uniform);
  - `table: [<rows>]`: a sub-table.

  For example, half the steps to the contrastive dojos and half to the
  metric dojos:

  ```yaml
  diet:
    rule: table
    table:
      - weight: 1
        dojos: ["contrastive*"]
      - weight: 1
        dojos: ["gwent_one.*", "sts2_runs.*"]
        within: {rule: temperature, alpha: 0.3}
  ```

  The table names the phase's dojos, so a table-diet phase has no `dojos:`
  list. Patterns match only the run's non-held-out dojos; a pattern that
  matches nothing, or a dojo under two rows, is a config error.

A dojo with no TRAIN examples is never drawn under any rule. Under a table,
a dojo that saturates or is quarantined leaves its own sub-table, so its
share goes to its siblings (the contrastive half above stays half while
any contrastive dojo is active). A row left with no drawable dojo drops
out, its share going to the rows beside it, and is logged once as a
warning. Flat rules draw with `rng.choices` directly rather than through a
one-row table, so a flat-rule run's random stream is unchanged by tables.
