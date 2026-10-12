# TODO (training)

Open work on the trainer, its data and its run config. Runs and
ablations are in `experiments/TODO.md`. Background from 2026-09-21
(hardware, GPU measurements, the choice of ModernBERT-base, the ROCm
install) is in `Notes.md`.

## Data and dojo readiness

- [ ] **Build a real inventory.** The code half is done:
  `scripts/report_metric_dojo_inventory.py` regenerates
  `docs/metric_dojo_inventory.csv` (every metric class, its output path,
  its dojo and cell; a test keeps it current). Still to add, needs data:
  row count per parquet and whether its cards resolve in the binder.
- [ ] **Report card-loading health per game.** Unresolved names, alias
  collisions, 17lands-name -> Scryfall-UUID resolution rate for MSH and
  KTK. Check STS2: 578 cards against 7.8k decks.
- [ ] **Finish loss normalization across dojos.** Regression labels are
  z-scored from TRAIN statistics, every dojo reports a baseline and its
  normalized loss (loss / baseline, 1.0 = learned nothing), and each
  step's loss is divided by the dojo's baseline and scaled by a per-run
  `loss_weights:` entry (`loss_weighting.py`; `baseline_floor` 0.05 from
  `scripts/survey_dojo_baselines.py`). Still open:
  - a per-dojo grad-clip counter in the round log, and a look at the
    clip rate;
  - a diet whose weights shift over time;
  - two-level diet sampling (below);
  - much later, as experiments: learned per-dojo weights (uncertainty
    weighting, Kendall et al. 2018) and learning-progress task selection
    (Graves et al. 2017).
- [ ] **Outlier regression labels: what Huber does not cover.** z-scoring
  does not tame heavy tails. Built so far: `HuberLoss` in the three
  regression cells (`regression_loss: huber`), and label caps on
  `isotropic.kingdom_game_length` (50) and `NextTurnActionCountDojo`
  (30). The MSE-vs-Huber experiment is listed in `experiments/TODO.md`;
  when it runs, compare both arms with loss weighting on and off on
  RMSE and MAE in label units on TEST plus wall-clock time (normalized
  loss is not comparable across them), then decide the default, whether
  to keep the two caps, and whether the Huber baseline is the
  loss-weighting divisor (Huber baselines, 2026-10-08: 0.49 to 1.0 over
  57 regression keys; 26 more could not build on stale metrics). Still
  open besides:
  - every other regression dojo's label histogram is unchecked;
  - the `sts2_runs` deck labels are unclipped (e.g. total_cards_skipped
    max 209, elites_killed max 28);
  - label transforms (log) for the most extreme labels, and robust
    scaling (median/IQR) for the std inflation that tails cause.
- [ ] **Save each dojo's `LabelStats` with the run.** The TRAIN mean/std
  that z-score a regression dojo's labels are recomputed at every dojo
  construction and only appear in the construction log line. That is
  reproducible while the split files stay put, but a rebuilt split would
  silently pair a checkpoint's head with different stats. Write each
  dojo's stats into the run directory (manifest or `validation.json`) and
  read them back for evaluation, so predictions convert to label units
  reliably.
- [ ] **Cache calibrated baselines** (only if dojo startup becomes slow
  again; calibration reads a 20k-row TRAIN prefix today). Keyed by
  split-file identity, stored beside the splits, together with
  `LabelStats` (item above).
- [ ] **Two-level diet sampling: game, then dojo.** Today the diet
  sampler (`src/training/diet/diet_sampler.py`) picks one dojo per step
  with probability proportional to its TRAIN count^alpha. alpha = 0 means
  uniform over dojos; alpha = 1 means proportional. Two imbalances follow:
  - A game with many dojos gets proportionally more steps than a game
    with one (e.g. 20+ isotropic/dominiontabs dojos against Pokemon's
    single contrastive dojo).
  - With alpha > 0, huge dojos crowd out small ones (contrastive.mtg has
    3.85M train decks, a Dominion card-rate dojo about 130 cards).

  Fix: first sample a game (uniform, or temperature over the game's total
  data), then a dojo within that game (temperature over dojo size). This
  is the multilingual-model recipe applied twice, with games standing in
  for languages. See Arivazhagan et al. 2019, "Massively Multilingual
  Neural Machine Translation in the Wild", and Conneau et al. 2020,
  XLM-R, alpha = 0.3.

  Needs a game per dojo (the catalog recipes know it). The table diet
  (`rule: table`, `src/training/diet/README.md`, built 2026-10-09) already
  does this when the per-game sub-tables are written by hand, and already
  drops saturated dojos within their sub-table and an emptied sub-table
  as a whole. What remains is a `group_by: game` option on a `dojos:` row
  that builds one sub-table per game from the catalog.
- [ ] **Generate more metrics only where they pay off.** A few more 17lands
  sets (game and draft) via `scripts/run_metrics.py`; Play-Gwent and FaB
  decklists still need metrics. Defer replay data and Dominion (no raw data
  or binder on disk).
- [ ] **Augmentation tally overcounts on some mask dojos.** On a mask
  dojo whose target is also a row of its game's mask table (e.g.
  `gwent_one.faction_mask`, `spire_codex.color_mask`), that row only
  re-masks the target, so up to half the draws do nothing yet count as
  changed in the tally. Per-dojo mask tables would fix it if it matters.
- [ ] **Experiment: subsample common cards in contrastive pairs.**
  `SingleCardPairConstructor` (`src/dojos/contrastive/pair_constructor.py`)
  samples from a deck's full card multiset, so base cards (basic lands,
  Copper/Estate/Province) fill many positive pairs. They are low-signal
  but not worthless, so the default stays unchanged. Measured 2026-09-30
  on 3,000 decks per game:
  - Dominion: cards in >50% of decks fill 57% of slots, so only ~19% of
    2-card positive pairs have no such card.
  - MTG (17lands): basic lands fill ~20% of slots.
  - Gwent, Flesh and Blood, Slay the Spire 2 and Pokemon: negligible.

  Proposed fix, as an ablation: keep each card with probability
  `min(1, sqrt(t / n))`, where `n` is the card's share of that game's
  decks (document frequency, computed from the deck box). This is
  word2vec's frequent-word subsampling (Mikolov et al. 2013), which
  item2vec (Barkan & Koenigstein 2016) applied to unordered item sets.
  One knob replaces a per-game list of "basic" cards:
  - `t = inf`: no change (the baseline).
  - a moderate `t`: common cards are sampled less.
  - a tiny `t`: common cards are nearly always skipped.

  Optional extra arm: logQ correction of in-batch negatives (subtract the
  log sampling frequency from the logits; Yi et al. 2019). Until this
  runs, give Dominion's contrastive dojo a low weight in a mixed diet.

  **Knob built (2026-10-01, T5); the ablation itself has not run yet.**
  `staple_subsampling: {contrastive.dominion: 0.1}` in a run config sets
  `t`; a dojo left out keeps `t = inf`. See
  `src/dojos/contrastive/staple_subsampling.py`. Document frequency comes
  from 20,000 TRAIN decks, cached under `data/splits/contrastive/`. A deck
  thinned below `items_per_deck` is skipped. Measured on 3,000 Dominion
  VALIDATION decks, as the share of 2-card positive pairs with at least
  one staple (df > 0.5): t = inf 79.8%, 0.3 66.3%, 0.1 53.9%, 0.03 42.5%,
  0.01 42.2% (98.8% of decks kept). It levels off near 42% because decks
  hold many staple copies. Suggested arms: inf, 0.1, 0.01.
  `MultiCardPairConstructor` is not wired up, because its item sampler is
  still a stub.

## Training driver

- [ ] **Checkpoint size and contents.** Only the best checkpoint per
  phase plus `latest/` are kept, but each holds the frozen LM too (~1.2
  GB, twice: `state.pt` and `encoder.pt`). The manifest stores only
  `repr(plan)`, not the run config.
- [ ] **Set `max_batch_cost` sensibly.** It counts cards, not tokens, and a
  40-card deck counts as 40. Start small (~32) and raise it while watching
  VRAM.
- [ ] **Decide about resume.** Not supported (tracker and RNG state are not
  saved); accept that for the first run or scope it.
- [ ] **Learn pre-norm vs post-norm** (`group_attention.norm_first`).
  Defaulted to pre-norm (`true`): LayerNorm before attention and the
  feed-forward block rather than after, the usual choice now because it
  trains stably without a learning-rate warmup, which the trainer does
  not have. torch's own default is post-norm. Read up on it, or ask for an
  explanation, and decide whether it deserves an ablation.

## Housekeeping

- [ ] **Split `run_config.py` (913 lines) and `dojo_catalog.py` (895
  lines) into smaller files.** Neither violates a specific size
  heuristic badly today (no individual function is oversized, and the
  `_MOD_PARSERS`/`DOJO_CATALOG` dispatch-table style already avoids a
  long if/elif chain, which PRINCIPLES.md §5 wants), but per §3 (file
  organization) each file mixes a large flat literal/table with
  meaningful parsing/building logic, and is a candidate to move its
  private helpers (`run_config.py`'s `ConfigSection`/`_parse_*` family;
  `dojo_catalog.py`'s `_recipe_for_*` builders) into sibling
  implementation files, leaving the catalog/public API in a slimmer
  top-level file. Most of the bulk is irreducible data (one catalog
  line per dojo), so this is lower-priority than the line count alone
  suggests. Flagged during a 2026-10-05 project-wide cleanup scan, not
  yet fixed. (`src/training/run_config.py`, `src/training/dojo_catalog.py`)
- [ ] **Update `requirements.txt` and `scripts/check.sh`.** Document the
  ROCm torch install (command in `Notes.md`). `check.sh` runs `python3`,
  which on this Windows setup needs a shim pointing at the ROCm venv
  (the only one with the full test dependencies) while black, flake8 and
  mypy come from the project `venv/`. Decide whether the project `venv/`
  should become the ROCm one.
