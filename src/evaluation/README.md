# evaluation

Post-training evaluation of the embedding model itself: does a checkpoint's
embedding space have the properties we want, and how useful is it for
tasks it wasn't trained on. Distinct from `training/round_evaluation.py`,
which is the loop's own cheap per-round TEST signal used for saturation
and checkpoint selection - `evaluation/` inspects a finished checkpoint,
offline, and is never called from `Trainer`.

Status: design only (this README + `TODO.md`); no code yet. See
`plans/evaluation_design.md` for the fuller discussion this was distilled
from.

## Two families

- **[`intrinsic/`](intrinsic/README.md)** - properties of the embedding
  space itself, computed with no trained downstream component: given
  embeddings and (usually) a set of known card labels, do they geometrically
  line up. Cheap, closed-form, no gradient descent.
- **[`extrinsic/`](extrinsic/README.md)** - how useful the frozen embedding
  is as input to a task it wasn't necessarily trained on: train a small
  decoder head on top and measure how well and how fast it learns. A real
  (if small) training run, with its own loss curve.

The dividing line is not "uses labels or not" (both families do) - it's
whether a new model is fit on top of the frozen embeddings. Intrinsic:
no. Extrinsic: yes.

## The shared building block: an evaluation corpus

Both families ultimately need the same underlying thing - a set of cards
(and, for intrinsic, a mapping from those cards to known labels) - so
that piece is shared rather than reimplemented per workflow:

- **Which cards.** A game, several games, or every game via
  `CardBinder.all_cards()`. For extrinsic, "which cards" is largely already
  handled for us: a `Dojo` is itself a cards-and-labels source with a lot
  of extra plumbing (splits, a loss, a decoder head) bolted on, so
  extrinsic evaluations mostly just consume a `Dojo` directly rather than
  building their own card selection.
- **Which labels** (intrinsic only). A metric's own output
  (`src/data_refinement/metrics/`'s `(nocab_uuid, label)` parquet, e.g.
  `ColorMaskMetric`) is the label source - not the `Dojo` that wraps it.
  The metric already carries exactly the eligibility-filtered
  `card -> label` mapping intrinsic evaluation needs, with none of the
  split/holdout/loss-head machinery it doesn't.
- **Domain drift ("how far from training").** Modeled two different ways,
  deliberately not unified, because they answer different questions:
  - *Mutated card* (same card identity, a field perturbed) -> a
    `Mod`/`ModPipeline` (`src/dojos/mods/`), reused as-is rather than
    reimplemented - it's already generic over `TrainingDatum`, nothing
    dojo-specific about it.
  - *Unseen card / unseen game* (different identity, never trained on) ->
    `HoldoutSpec` (`src/schema/holdout.py`), also reused as-is:
    `held_out_games` already gives whole-game holdout; a card's
    TEST/VALIDATION tier already gives per-card holdout.
  v1 scope: single-card mutation and whole-game holdout only. Held-out
  *sets/expansions within* a game needs a small `HoldoutSpec` extension
  (no predicate hook today, only whole-game or per-card hash) - real,
  deferred, tracked in `TODO.md`, not needed for a first pass.

## Subdirectories

- **[`intrinsic/`](intrinsic/README.md)** - two workflows: cluster-then-
  match-known-labels, and known-labels-then-measure-separation.
- **[`extrinsic/`](extrinsic/README.md)** - one workflow: fresh-decoder-
  head training curve on a dojo.
