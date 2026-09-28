# TODO (evaluation)

Started 2026-09-28 from a design conversation (see `README.md`,
`intrinsic/README.md`, `extrinsic/README.md`). Nothing here is built yet;
this is the ordered list toward a first working pass.

## First implementation pass

- [ ] **Shared eval-corpus builder.** `card_lookup + HoldoutSpec ->
  Iterable[GenericCard]`, optionally joined against a metric's
  `(nocab_uuid, label)` output, optionally piped through a `ModPipeline`.
  Both `intrinsic/` workflows and the corpus half of `extrinsic/` sit on
  top of this - build it first.
- [ ] **`intrinsic/`: cluster-then-match** (ARI / NMI against a metric's
  known labels).
- [ ] **`intrinsic/`: known-labels-then-tightness** (silhouette score /
  nearest-centroid accuracy against a metric's known labels).
- [ ] **`intrinsic/`: t-SNE / UMAP plot**, fixed seed and
  perplexity/n_neighbors, colored by a metric's labels - paired with, never
  a substitute for, the two quantitative workflows above.
- [ ] **`extrinsic/`: fresh-head training curve** on a `Dojo` (unseen, or
  seen-but-`reset_head()`'d), reusing
  `training/round_evaluation.py::evaluate_test_losses` for the TEST-side
  number. Report time-to-a-given-loss (wall clock + step count) and
  deepest loss reached.

## Known gaps / deferred, not blocking the first pass

- [ ] **Held-out sets/expansions within a game.** `HoldoutSpec` only
  supports whole-game holdout (`held_out_games`) or a per-card random
  hash - no predicate hook for "everything from set X is VALIDATION-tier".
  Needed for a finer-grained point on the distributional-distance curve
  than "whole new game"; a small, contained extension to `HoldoutSpec`
  (e.g. an optional `held_out_predicate: Callable[[GenericCard], bool]`),
  not a rewrite. v1 uses single-card mutation and whole-game holdout only.
- [ ] **Multi-card model comparison's context-list convention.** The
  multi-card model's embedding for a card depends on what else is in its
  input list at inference time (real deck? random pool? list of one?).
  Not decided; punt until a multi-card checkpoint actually needs
  evaluating.
- [ ] **A "new language" `Mod`.** Translate a card's `raw_content` text as
  a distributional-distance point past "new game" - architecturally just
  another `Mod` subclass, but needs translation tooling/data this project
  doesn't have yet. Reach, not scoped.
- [ ] **Relocate `Mod`/`ModPipeline`** out of `src/dojos/mods/` to a more
  neutral shared location, since `evaluation/` importing from `dojos/` is
  a real but slightly odd cross-container dependency (works fine per the
  existing `dojos -> training -> evaluation` layering; just a naming/home
  cleanup, not a blocker).
- [ ] **Extrinsic battery mode.** Run the fresh-head workflow over several
  dojos in one pass rather than one at a time, once the single-dojo path
  is proven.
