# evaluation TODO

## Card in contexts: identity retention vs. context sensitivity

Use the card-in-contexts contrastive dojo
(`contrastive_card_in_contexts.<game>`,
`src/dojos/contrastive/styles/card_in_contexts.py`) as an evaluation
probe, and build metrics and visualizations from what it finds.

The dojo places one anchor card in two disjoint slices of its deck and asks
whether the anchor's two contextual embeddings still match each other
better than other decks' anchors. Read alone, a good score is ambiguous:
a model that ignores context scores perfectly, and `SingleCardModel` does by
construction. So report it as one axis of a pair:

- **Identity retention**: card-in-contexts loss (normalized by its
  baseline) and top-1 accuracy on TEST batches. Does a card stay
  recognizably itself across decks?
- **Context sensitivity**: how far a card's contextual embedding moves
  away from its isolated one (cosine distance, averaged over contexts), and
  how much its two contexts disagree. Does context change anything at all?

A healthy multi-card model scores well on both. Collapse (context ignored)
shows high retention and near-zero sensitivity. Over-contextualization
(the card overwritten by its deck) shows the reverse. `SingleCardModel`
is the reference point: perfect retention, zero sensitivity.

Ideas:

- A scalar pair per encoder and game, written like the other analyses'
  `scalars.json`.
- A retention-vs-sensitivity scatter: one point per encoder (or per
  checkpoint over a run, to watch it drift), per game.
- Per-card breakdown: which cards lose their identity most in context
  (likely flexible staples), and which never move (likely vanilla or
  narrow cards).
- A small-multiples view of one anchor's contextual embeddings across many
  decks, projected next to its isolated embedding.

Card in contexts stays out of every training diet; see the dojo's
docstring.

## Deck AUC as an intrinsic analysis

Add same-deck retrieval AUC to `analyses/` as an `EmbeddingAnalysis`.
For a card in a sampled deck: P(cos to a random deck-mate > cos to a
random same-game card outside the deck), 0.5 = chance. It is the
contrastive dojos' task with no training, and the main signal they add,
but today it exists only in experiment scripts
(`experiments/2026-10-10_pooling_probe/analysis/probe.py`, wrapped by
`experiments/2026-10-11_model_by_diet_r2/analysis/probe_cells.py`).

Needs, beyond what the other analyses take:

- a deck source (decks from the deck boxes, sampled with a seed), since
  `EmbeddingTable` holds cards only; a card missing from the table is
  skipped and counted;
- scalars: overall AUC, AUC per game, and AUC split by the anchor card's
  holdout tier (TRAIN vs VALIDATION/TEST), so in-sample and out-of-sample
  readings are separate;
- a 95% bootstrap interval over decks, as `probe_cells.py` reports.

## run_evaluation.py: held-out dojos, cards and games

`scripts/run_evaluation.py` is a Gwent-only prototype with hard-coded
encoders, labels and dojos. Make it drive an experiment's own setup, so
evaluating a finished experiment is one command instead of new
`analysis/` scripts (the 2026-10-11_model_by_diet_r2 step 2 needed
`probe_cells.py`, `extrinsic_held_out.py` and `intrinsic_suite.py`):

- **Encoders:** build from a run's recorded `run_config.yaml`
  (`build_model`), any architecture, plus untrained controls under
  chosen seeds; not the fixed linear-projection reference models.
- **Held-out dojos:** run extrinsic fresh heads on a config's
  `held_out_dojos` (or a named list) via `build_dojos`, with the run's
  own holdout.
- **Held-out cards:** a corpus `TierFilter` option (e.g. only
  VALIDATION/TEST cards under the run's holdout), and per-tier splits of
  every analysis's scalars.
- **Held-out games:** choose the corpus games, and report games no dojo
  trained (e.g. Gwent) separately.
- **Label sources:** chosen per run (game, holdout tier, metric parquets
  for trained and held-out fields), not hard-coded.
- **Shared frozen features:** while the text encoder is frozen, compute
  its pooled output once per card and reuse it for every encoder (the
  experiment scripts' cache; exact against the real model).
- **Standard output:** one scalars CSV across encoders, so experiments
  compare without per-experiment report scripts.
