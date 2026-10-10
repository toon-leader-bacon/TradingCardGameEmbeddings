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
- [dojo_fault_ledger.py](dojo_fault_ledger.py): counts consecutive step
  failures per dojo and overall; quarantines a dojo that keeps failing (in a row or in total) and
  signals when the whole run should stop.

## How it works

Each round, `Trainer` asks the tracker for the ACTIVE dojos, then for each
step asks the sampler to pick one and its batch stream for the next batch.
After the round, the normalized TEST losses go back into the tracker.

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
