# training

Trains the card-embedding encoder (`../encoder_model/`) against a suite of
dojos (`../dojos/`). The encoder is the product; dojos are the teachers.
Status: implemented and tested; run end to end on CPU against two real
Gwent dojos (`scripts/smoke_test_training_loop.py`). No training driver or
config file yet (see `TODO.md`, section D).

Start with [`trainer.py`](trainer.py). It is the only file that knows the
whole story; every other file is a piece it delegates to.

## How it fits

```mermaid
flowchart LR
    Plan[TrainingPlan<br/>plan.py] --> Trainer
    Dojos["Dojo Protocol<br/>src/dojos/dojo.py"] --> Trainer
    Encoder["SingleCardModel / MultiCardModel<br/>src/encoder_model"] --> Trainer
    Trainer --> Ckpt[Checkpoint on disk<br/>encoder.pt = the artifact]
    Ckpt -.-> Eval["evaluation/<br/>(post-training, separate)"]
    Eval -.->|extrinsic runs construct| Trainer
```

The trainer sees dojos only through the `Dojo` Protocol, so generic and
contrastive dojos are interchangeable. It never calls `evaluation/`;
evaluation's extrinsic runs construct a `Trainer` of their own (one frozen
phase, no checkpointer) to train fresh dojo heads on a finished encoder.
Mixed precision (`Precision`, `autocast_for`) lives in
`../encoder_model/precision.py`, shared with evaluation's embedders.

## Files

Top level:

- [trainer.py](trainer.py): `Trainer.run()` walks phases -> rounds ->
  optimizer steps. One step trains on one batch from one dojo.
- [phase_run.py](phase_run.py): the per-phase machinery (optimizer,
  tracker, sampler, batch streams) bundled into one object.
- [plan.py](plan.py): `TrainingPlan` -> `Phase`s. A phase says which
  dojos, which `DietRule`, learning rates, whether the encoder is frozen,
  and when to stop (`SaturationSpec`). Also `HardwareLimits`. Frozen and
  validated on construction, so a run is described entirely by its plan.
- [round_evaluation.py](round_evaluation.py): `evaluate_split_losses`, a
  capped, deterministic scoring pass over one split for every dojo, at a
  given precision. `Trainer` runs it on TEST each round (diet or held-out
  dojos alike) as the tracker's input, the loop's own signal; evaluation's
  extrinsic runs call it once on VALIDATION after training.
- [trainable_encoder.py](trainable_encoder.py): the slice of the model the
  trainer needs.
- [preflight.py](preflight.py): `preflight_dojo()` exercises one already-
  built dojo (one TRAIN batch, `compute_loss` against random embeddings)
  without an encoder, so a stale split file or a `card_embedding_size`
  mismatch surfaces before an unattended run reaches it. Run over the
  first-run candidates by `scripts/preflight_dojos.py`.
- [TODO.md](TODO.md): the checklist to a first real training run.
  [Notes.md](Notes.md): the dated findings behind it (hardware, GPU
  measurements, serialization analysis).

Subdirectories:

- [diet/](diet/README.md): what to train on next (sampler, saturation
  tracker, batch stream).
- [recording/](recording/README.md): what comes out of training (reports,
  checkpointer, listeners).

## Vocabulary

From largest unit to smallest:

| Term | What it is | Where |
|---|---|---|
| **Run** | One `Trainer.run()`: a whole `TrainingPlan` executed against one model and a set of dojos. | `trainer.py`, `plan.py` (`TrainingPlan`) |
| **Phase** | One stage of the plan (e.g. pretraining with everything trainable, then a head-only stage with the encoder frozen). Each has its own diet, learning rates, frozen flag and stopping rule. Phases run in order. | `plan.py` (`Phase`) |
| **Round** | The loop's decision unit within a phase: `steps_per_round` optimizer steps, then one TEST pass over every dojo, after which the diet is re-decided, saturation checked and a checkpoint maybe written. A phase runs until enough dojos saturate or `max_rounds` is reached. | `Trainer._train_round` + `_evaluate_round` |
| **Step** | One optimizer update: pick a dojo from the diet, take its next batch, forward, loss, backward, update. `RoundReport.step` is the running total. | `Trainer._take_step` |
| **Batch** | What one step consumes: examples from **one** dojo, sized by a cost budget (`max_batch_cost`) rather than a fixed count. | `DojoBatch`, `BatchBudget` |
| **Epoch** | **Not a concept here.** Each dojo's TRAIN data is an endless stream that starts another pass whenever it runs out. Dojos differ in size and sampling rate, so "one pass over the data" means something different for each dojo and nothing for the run. Rounds are the unit of progress instead. | `diet/dojo_batch_stream.py` |

```
Run
 └─ Phase 1 … N              (in order)
     └─ Round 0 … ≤ max_rounds
         ├─ steps_per_round × Step   (each = one Batch from one Dojo)
         └─ one TEST pass → RoundReport → saturation, diet, checkpoint
```

**Saturation:** a dojo whose TEST loss has stopped improving (no gain
larger than `epsilon` for `patience_rounds` rounds) leaves the diet. It
re-enters if its loss rises again by `reactivation_delta`.

## How it works

```mermaid
sequenceDiagram
    participant T as Trainer
    participant S as DietSampler
    participant B as DojoBatchStream
    participant M as Encoder model
    participant D as Dojo
    participant E as evaluate_split_losses
    participant K as SaturationTracker
    participant C as Checkpointer / Listeners

    loop each Phase
        T->>T: open phase (optimizer, tracker, sampler, streams)
        loop each Round (until phase_done or max_rounds)
            T->>K: active_dojo_names()
            loop steps_per_round
                T->>S: next_dojo(active)
                S-->>T: dojo
                T->>B: next_batch()
                B->>D: batches(TRAIN, budget)
                B-->>T: batch
                T->>M: model(batch.inputs)
                M-->>T: embeddings
                T->>D: compute_loss(embeddings, batch)
                D-->>T: loss (backward + optimizer step)
            end
            T->>E: TEST losses, ALL dojos
            E-->>T: per-dojo loss
            T->>K: record_round(losses)
            K-->>T: statuses
            T->>C: on_round_end(report); save if best
        end
    end
```

Points worth knowing:

- The diet is re-decided only at round boundaries, from the ACTIVE dojos.
- One `BatchBudget` (from `HardwareLimits.max_batch_cost` and `cost_of`)
  is passed to every dojo, so no dojo owns a batch size.
- Held-out dojos (`TrainingPlan.held_out_dojos`) are evaluated every round
  but never trained on.
- A phase with `encoder_trainable=False` keeps the encoder in eval mode
  (no dropout) while the dojo heads train; the heads are not part of the
  model, so their mode is untouched.
- Every `RoundReport` carries `elapsed_seconds` (monotonic, since
  `run()` began), for loss-vs-time curves.
- The checkpointer is optional: with `checkpointer=None` nothing is saved
  and `TrainingResult.best_checkpoint` is `None` (evaluation's extrinsic
  runs, smoke tests, dry runs).
- `Trainer` raises if any dojo's `holdout` differs from the plan's, so
  card holdout is consistent across dojos.
- The best checkpoint is chosen per phase; scores are not comparable across
  phases because the diet changes.
- Nothing here is meant to crash an overnight run. A failing step
  (exception, out-of-memory, non-finite loss or gradient), evaluation,
  checkpoint save or listener is logged and skipped. Gradients are clipped
  to `Phase.max_grad_norm`, and a non-finite gradient skips the step before
  it can reach the weights. Under `HardwareLimits(precision="fp16")` the
  loss is scaled by a per-phase `GradScaler` and gradients are unscaled
  before clipping; an fp16 overflow only skips the step and halves the
  scale, and counts as a failure only once the scale is at its floor of 1
  (the gradients are then non-finite in true units). A dojo that fails
  `FaultPolicy.max_consecutive_dojo_failures` times in a row (or
  `max_total_dojo_failures` in total) is quarantined for the phase, and
  `max_consecutive_failures` in a row stops the run cleanly with the last
  good checkpoint. Only plan/dojo mismatches raise, and they do so in the
  constructor, before training starts.
- A round is "best" only when its mean TEST loss over the diet dojos scored
  in both it and the previous best is strictly lower, so a round in which a
  hard dojo failed to evaluate cannot win by omission.
- A checkpoint is a weights snapshot; it does not save tracker or RNG
  state, so a run cannot resume mid-way, and there is no automatic
  rollback to the best checkpoint after divergence.
- Not built: gradient accumulation, GradCache, several tasks per step,
  per-dojo step size on the shared encoder, staged partial unfreezing,
  cached card encodings while the LM is frozen, token-aware `cost_of`,
  saving `GradScaler` state for resume, Hugging Face export.

## How to run

Assumes `model` is a `SingleCardModel`/`MultiCardModel` and `dojos` is a
list of constructed `Dojo`s.

```python
plan = TrainingPlan(phases=(...), holdout=HoldoutSpec(...), held_out_dojos=frozenset(),
                    eval_examples_per_dojo=512, seed=0)
result = Trainer(model, dojos, plan, HardwareLimits(max_batch_cost=256),
                 DirectoryCheckpointer(Path("runs/a")), [LoggingRunListener()]).run()
```
