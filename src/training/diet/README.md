# diet

Decides which dojo `Trainer` trains on next, and when a dojo (or a whole
phase) is finished.

## Files

- [diet_sampler.py](diet_sampler.py): `DietSampler` Protocol and the
  temperature-weighted sampler that covers proportional, uniform and
  temperature diets. `diet_sampler_for(rule)` builds one from a `DietRule`.
- [saturation_tracker.py](saturation_tracker.py): per-dojo ACTIVE/SATURATED
  state driven by each round's TEST loss; reports which dojos are still
  active and when the phase is done.
- [dojo_batch_stream.py](dojo_batch_stream.py): turns a dojo's finite TRAIN
  pass into an endless batch supply.
- [dojo_fault_ledger.py](dojo_fault_ledger.py): counts consecutive step
  failures per dojo and overall; quarantines a dojo that keeps failing (in a row or in total) and
  signals when the whole run should stop.

## How it works

Each round, `Trainer` asks the tracker for the ACTIVE dojos, then for each
step asks the sampler to pick one and its batch stream for the next batch.
After the round, the TEST losses go back into the tracker.
