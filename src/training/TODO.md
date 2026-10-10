# TODO (training): getting to a first real training run

Started 2026-09-21 from a review of the codebase, the hardware and the
data on disk (findings: `Notes.md`). The trainer has run end to end on CPU
against two real Gwent dojos and ModernBERT (section E); it has not yet
run on the GPU or with a training driver/config (section D). A is done;
B is done; C and D are the remaining path to a first real run.

## Findings this list is based on

**Hardware.** AMD Radeon RX 6800 (16 GB VRAM, RDNA2, `gfx1030`), NOT
NVIDIA. Ryzen 7 3700X (8 cores / 16 threads), 32 GB RAM, ~1.4 TB free on
`G:`. The venv has `torch 2.14.0+cpu`, so nothing can use the GPU today.
No HIP SDK installed, WSL not set up. AMD's official Windows PyTorch
wheels cover only RX 7000/9000; RX 6000 reportedly works via AMD's nightly
`gfx103X-dgpu` wheels plus the HIP SDK, which is unofficial. ROCm under WSL
reportedly does not work for the RX 6800.

**Serialization.** `serialize_card_to_json` dumps the whole `raw_content`
(image URLs, IDs, prices, legalities included). Measured token counts
(ModernBERT tokenizer, 300 sampled cards per game):

| Game | Cards | Median tokens | Over 512 tokens |
|---|---|---|---|
| MTG | 38,741 | ~2,350 | 100% |
| FaB | 5,188 | ~740 | 100% |
| Pokemon | 16,803 | ~360 | 0% |
| STS2 | 578 | ~345 | 0% |
| Gwent | 1,261 | ~135 | 0% |

With the current default (`distilbert-base-uncased`, `max_length=512`) the
MTG rules text starts after token 512 in ~90% of cards, so the encoder
mostly reads URLs. A rough denylist of URL/ID/artist/price keys cuts MTG to
a median of ~200 tokens and FaB to ~570.

**Data coverage.**

- Card binders: MTG, Pokemon, FaB, Gwent, STS2. None for Dominion or
  Hearthstone; Pitchstack has no ingestion stage.
- Deck boxes: FaB (4.1k), Gwent (60k), STS2 (7.8k) only.
- Metric parquets on disk: Gwent (all 8 masks), STS (25 files), 17lands
  (MSH PremierDraft: 2 draft files; KTK TradSealed: 11 game files). Nothing
  for the other ~34 17lands sets, 17lands replay data (139 GB raw),
  Play-Gwent, FaB decklists, Dominion or Isotropic.
- Dojos exist for Gwent, STS, 17lands (all 26 metrics), Dominion,
  Play-Gwent and contrastive. Only Gwent and STS were ever exercised (the
  old smoke test); no 17lands dojo has produced split files.

## A. Blockers (DONE 2026-09-21: the GPU works, no cloud/other box needed)

- [x] **Pick a GPU path.** Native Windows with AMD's nightly `gfx103X-dgpu`
  wheels works and needs no HIP SDK (the ROCm runtime ships inside the
  wheels). Installed `torch 2.10.0+rocm7.13.0a20260421` into a separate
  venv, `G:\Projects\venvs\tcg-rocm` (Python 3.12), plus numpy, pandas,
  pyarrow, transformers etc. The project's own `venv/` is still CPU-only.
  Install command:
  `pip install "torch==2.10.0+rocm7.13.0a20260421" --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/ --extra-index-url https://pypi.org/simple`
  The wheel index was last updated April 2026 (newest torch there is a
  2.13 alpha); it is a staging index, so it may change or disappear.
- [x] **Driver.** Works with the installed driver (`32.0.21045.5002`).
- [x] **GPU verification script:** `scripts/verify_gpu.py` (6/6 checks
  pass: matmul agrees with CPU, fp16 and bf16 autocast, SDPA, throughput,
  training steps).
- [x] **Add a fp16 `GradScaler` to the trainer** (2026-09-24).
  `HardwareLimits(precision="fp16")` runs training and evaluation forward
  passes under `torch.autocast` and steps through a per-phase `GradScaler`
  (unscale before clipping). An overflow skips the step without counting
  as a dojo fault until the scale reaches its floor of 1, where the step
  raises `NonFiniteGradientError` like the fp32 non-finite-gradient path.
  `"bf16"` is autocast only; `"fp32"` (default) is unchanged. Tested on CPU
  autocast; not yet run on the RX 6800. Scaler state is not checkpointed
  (only matters once resume exists).

### GPU measurements (RX 6800, 16 GiB)

- Raw matmul: fp32 15.1 TFLOPS, **fp16 25.2**, bf16 only 7.7. Use fp16
  autocast with a GradScaler, not bf16.
- No fast attention kernel: torch warns "not compiled with memory efficient
  attention", and `sdpa` and `eager` perform the same. Cost grows with
  sequence length (ModernBERT inference: ~40k tok/s at 256 tokens, ~28k at
  512, ~20k at 1024).
- **Padding dominates.** ModernBERT-base frozen, fp16, real cards: 28 cards/s
  with mixed-length batches vs 117 cards/s (cap 512) or 89 (cap 1024) when
  batches are length-sorted. Batch by similar length.
- **Fine-tuning ModernBERT-base** (fp16, GradScaler, synthetic full-length
  batches): 256 tokens: bs 16 = 51 cards/s, 5.6 GiB; bs 32 = 55 cards/s,
  9.2 GiB. 512 tokens: bs 16 = 22 cards/s, 11.5 GiB; bs 32 with gradient
  checkpointing = 17 cards/s, 4.6 GiB. 1024 tokens without checkpointing
  OOMs at bs 16 (bs 8 fits at 16 GiB).
- **Windows spills VRAM into system RAM instead of raising OOM.** 512 tokens,
  bs 32, no checkpointing reported a 21 GiB peak on a 16 GiB card and fell
  to 3.2 cards/s with no error. Watch for sudden slowdowns; the trainer's
  OOM handling will not catch this.
- Rough sizing: frozen LM at ~100 cards/s means a 32-card step costs ~0.3 s
  in the LM (before the head and dojo), so ~10k steps is roughly an hour
  or two. Fine-tuning at 256 tokens is ~5x slower than that per card.
  Feasible for a first model; not fast.

## B. Text encoder and serialization

- [x] **Choose the encoder: `answerdotai/ModernBERT-base`** (Apache-2.0,
  149M params, 8192-token context, cased, code-heavy pretraining). No open
  model is truly JSON-tuned; this is the closest. Runner-up:
  `jinaai/jina-embeddings-v2-base-code` (needs `trust_remote_code`). Skip
  `nomic-ai/modernbert-embed-base` (retrieval-tuned, needs prefixes). Now
  the default checkpoint in `reference_singlecard_models.py` and
  `reference_multicard_models.py` (was `distilbert-base-uncased`); confirmed
  it loads with `attn_implementation="sdpa"` by default, no override needed.
- [x] **Lean `raw_content` in each ingestion stage** (decision 2026-09-21:
  simplification lives in `src/data_refinement/card_binder/*/ingestion_stage.py`,
  not the serializer). Rules: `card_binder/README.md`, "What goes in
  `raw_content`"; evidence: `Notes.md` section 7 (MTG 2,352 to ~141 median
  tokens, FaB 732 to ~227). Work items:
  - [x] shared helper functions: `card_binder/lean_content.py`
    (`strip_noise`, `drop_repeated_strings`, `order_keys`), tested;
  - [x] Scryfall (MTG) stage converted, merge policy switched to
    `keep_incoming` (compared on content, so an unchanged re-ingest is a
    no-op). Verified on a copy of the real binder: all 38,741
    `nocab_uuid`s and 119,992 aliases preserved, second ingest changes 0,
    binder file 212.6 MB to 30.1 MB, MTG tokens 2,352 to a median of 183
    (153 with compact JSON), p99 351 (295);
  - [x] **re-ingested the binders** (2026-09-21, `--all`). New
    `nocab_uuid`s, so metric parquets/splits/deck boxes still need
    regenerating (untouched so far, see section C);
  - [x] **re-ingested `gwent_one` once more** (2026-09-23), after the
    `category`/`faction-duo` leak fix. In place, all uuids preserved; 42
    cards content-changed (exactly the 42 leader cards), confirming the
    fix and nothing else moved. `decks/gwent.jsonl` and Gwent
    metrics/splits stay valid;
  - [x] converted (details in `plans/archive/card_content_conversion.md`):
    FaB (median tokens 657 to 70), Pokemon (349 to 142), STS2 (342 to 69),
    Gwent (131 to 89), Scryfall/MTG (2,352 to ~150). All under 384 tokens
    at p99 or better;
  - [x] **dominiontabs** converted (median 48 tokens, max 403). Open
    decision (2026-09-21): the 2 German BB2DE duplicates are skipped (817
    cards); the 3 fan-made cards and 13 pile headers/"Start Deck" stay in.
- [x] **`serialize_card_to_json` emits compact JSON**
  (`ensure_ascii=False`, `separators=(",", ":")`), no per-game logic; tested.
  About 17% fewer tokens than `json.dumps` defaults.
- [x] **Serialization report script:** `scripts/report_card_content.py
  --source <name>` (token percentiles, costliest keys, identical
  serializations); run when a game is onboarded (README rule 11).
- [x] **`max_length=384`** (`text_encoder.py`'s `_DEFAULT_MAX_LENGTH`), and
  `PretrainedTextEncoder` now takes `model_kwargs: dict | None` forwarded to
  `AutoModel.from_pretrained` (e.g. `attn_implementation`). `reference_compile`
  no longer exists in transformers 5.x (passing it raises `TypeError`), so
  it is not needed and not passed by default.
- [x] **Batch by similar length**, inside `PretrainedTextEncoder.encode`
  itself (no dojo/trainer changes needed): a call over more than
  `length_bucket_size` texts (default 16) is sorted by character length and
  tokenized/forwarded in sorted sub-batches, then reassembled into one
  padded result in original order - tested, including that bucketed and
  unbucketed calls produce identical output and that gradients still reach
  the model. `length_bucket_size` is a real tuning knob (right value
  depends on hardware and typical batch size); revisit once real training
  numbers are in (see section E).
- [x] **Checked masking dojos against the new serialization** - found and
  fixed two real leaks in gwent.one (the only games with masking dojos
  today: gwent_one, 8 dojos; dominiontabs, 2 dojos - checked both, only
  gwent.one had a leak):
  - `category` was exactly `"Leader"` on every `color == "leader"` card
    and never otherwise (confirmed both directions against the live
    corpus) - pure duplication, not real category info for those 42 cards,
    so the gwent.one ingestion stage now drops it for them (same as an
    empty category), closing the leak for `ColorMaskDojo`;
  - `faction-duo`, when present (15 of 1260 cards), always contains the
    true `faction` as its own prefix - not pure duplication (it names a
    real second faction), so `FactionMaskDojo` now also masks it, rather
    than the stage dropping it.
  Dominion's `CostRegressionDojo` masks `cost`, but `potcost`/`debtcost`
  (different currencies, never literally equal to the coin cost) don't
  restate it - checked, no leak. `TypeMaskDojo` masks the whole `types`
  list at once, so list length doesn't matter.
- [x] **Measure frozen-LM throughput.** Done, see "GPU measurements" in
  section A: ~100 cards/s length-sorted. The trainer re-encodes every card
  every step; caching frozen-LM outputs is "not built" and is worth it only
  if this proves too slow (masking/key-shuffle mods change the input, so
  caching only fits unmodified cards).

## C. Data and dojo readiness

- [ ] **Build a real inventory.** The code half is done:
  `scripts/report_metric_dojo_inventory.py` regenerates
  `docs/metric_dojo_inventory.csv` (every metric class, its output path,
  its dojo and cell; a test keeps it current). Still to add, needs data:
  row count per parquet and whether its cards resolve in the binder.
- [ ] **Regenerate the corrupt MSH PremierDraft draft metrics.**
  `pool_conditioned_pick.parquet` (1.8 GB) and
  `pack_to_pick_choice_set.parquet` (949 MB) are truncated (no footer),
  most likely from an OOM kill mid-write. The cause, one parquet row group
  per CSV row, is fixed in code: both metrics now write through
  `ParquetBuilder` (bounded row groups). Re-run
  `scripts/run_metrics.py --source seventeenlands_draft_data --raw-path
  data/raw/17lands/draft_data/MSH.PremierDraft.csv` and watch memory;
  until then these two stay out of the first-run dojo set.
- [ ] **Report card-loading health per game.** Unresolved names, alias
  collisions, 17lands-name -> Scryfall-UUID resolution rate for MSH and
  KTK. Check STS2: 578 cards against 7.8k decks.
- [ ] **STS deck-level dojos need a metrics re-run.** They read
  `STS_GG_DECK_BOX_PATH` (`data/metrics/sts_gg/deck_box.db`), but only an
  older `deck_box.jsonl` is on disk; run `scripts/run_metrics.py --source
  sts_gg`.
- [ ] **Normalize loss across dojos.** Dojo losses sit on unrelated
  scales: cross-entropy near ln(classes), InfoNCE near ln(batch items),
  MSE in the target's raw units (STS card-average regressions up to ~7e4).
  Plan agreed 2026-09-30, in three steps:
  1. **Done.** Regression labels are z-scored from TRAIN-split statistics;
     every dojo reports a baseline (`Dojo.baseline_loss`) and its
     normalized loss (loss / baseline, 1.0 = learned nothing) beside the
     raw loss, in `RoundReport`, the rounds CSV, the log line and
     `validation.json`; best-checkpoint selection and saturation use
     normalized loss (`epsilon`/`reactivation_delta` are now fractions of
     the baseline; the existing configs' 0.001 / 0.05 still read sensibly).
  1b. **Done.** The gradient is rescaled too: each step's loss is divided
     by the dojo's baseline and multiplied by a per-run `loss_weights:`
     entry (`loss_weighting.py`). `baseline_floor` defaults to 0.05 from
     `scripts/survey_dojo_baselines.py` (111 buildable keys, baselines
     0.0077 to 4.9; only `play_gwent.faction_conditioned_inclusion` and
     `hearthstonejson.races_mask` sit below it). Open: a per-dojo grad-clip
     counter in the round log and a look at the clip rate on the first GPU
     run, and later a diet whose weights shift over time.
  2. Two-level diet sampling, below.
  3. Much later, as experiments: learned per-dojo weights (uncertainty
     weighting, Kendall et al. 2018) and learning-progress task selection
     (Graves et al. 2017).
- [x] **sts_gg.card_win_rate has a constant label.** Every row of
  `data/metrics/sts_gg/card_win_rate.parquet` is 1.0 (546 cards). Cause
  (2026-10-01): sts.gg's source is its leaderboard, which lists winning
  runs only (all 1,004 raw runs have `win: true`, `killedBy: null`). So
  `sts_gg.card_win_rate` and `sts_gg.win` left the catalog (code kept),
  and the win-based labels come from the `sts2_runs.*` keys instead
  (spire_codex runs, which include losses).
- [x] **Handle outlier regression labels (Huber built; experiment pending).** z-scoring rescales labels but
  does not tame heavy tails. isotropic.kingdom_game_length has mean 19.8
  turns and std 6.4, but a max of 323 (about 47 std out). Under MSE that
  one example adds about 2,200 to its batch loss and dominates the
  gradient. Options, per dojo, in increasing effort:
  - clip labels to a percentile range when the metric is built (e.g.
    p0.5-p99.5) and record the bounds;
  - Huber / smooth-L1 loss in the regression cells: squared error near
    zero, linear beyond a threshold (in std units after z-scoring);
  - transform heavy-tailed labels first, e.g. log(turns) or log(1 +
    count).

  Check every regression metric's label histogram before picking; the
  isotropic C1 report also flags mid_game_next_turn_action_count (max 112).
  Done for those two (2026-10-01): kingdom_game_length's metric now skips
  solo/resigned games (the sub-5-turn labels) and its dojo clips at 50;
  NextTurnActionCountDojo clips at 30 (IsotropicDeckRegressionDojo's
  LABEL_CAP). Other regression dojos are unchecked.
  Still unclipped (2026-10-01): the new `sts2_runs` deck labels, e.g.
  total_cards_skipped (max 209) and elites_killed (max 28).
  Built (2026-10-08): `HuberLoss` (`src/dojos/loss/huber_loss.py`) in the
  three regression cells, selected per run by `regression_loss: huber`
  (default `mse`, today's behavior). The MSE-vs-Huber experiment is still
  to run: both arms with loss weighting on and off
  (`weight_by_baseline: false`), compared on RMSE and MAE in label units
  on TEST plus wall-clock time (normalized loss is not comparable: the
  baselines differ). Then decide the default, whether to retire the two
  domain caps (cap-only, Huber-only and both arms on those two dojos), and
  the Huber baseline's role as the loss-weighting divisor (rerun
  `scripts/survey_dojo_baselines.py --regression-loss huber`). That survey
  ran 2026-10-08 on the 57 buildable regression keys: Huber baselines
  0.49 to 1.0 (median 0.87, none under the 0.05 floor), so loss weighting
  scales those dojos by at most about 2x; the card-average regressions
  and the sts2 deck counts are lowest (`sts2_runs.card_deck_size` 0.49).
  26 regression keys could not build (stale MTG/Pokemon/17lands metrics). Still open:
  label transforms (log) for the most extreme labels, and robust scaling
  (median/IQR) for the std inflation tails cause.
- [ ] **Save each dojo's `LabelStats` with the run.** The TRAIN mean/std
  that z-score a regression dojo's labels are recomputed at every dojo
  construction and only appear in the construction log line. That is
  reproducible while the split files stay put, but a rebuilt split would
  silently pair a checkpoint's head with different stats. Write each
  dojo's stats into the run directory (manifest or `validation.json`) and
  read them back for evaluation, so predictions convert to label units
  reliably.
- [x] **Shuffle split files uniformly; calibrate from a prefix**
  (2026-10-09). Split files used to keep their metric file's order
  (`make_splits` shuffled only within 10k-row batches): 17lands
  `pool_conditioned_pick` TRAIN ran AFR first to WOE last, so training
  and per-round TEST scoring read only AFR, and calibration had to read
  whole files for a fair sample (~61 minutes over the 142 dojos of a
  series-1 config). `make_splits` now shuffles the whole file in two
  passes (`src/dojos/README.md`, `file_managers/`), and calibration reads
  the first 20k TRAIN rows. Still to do: **re-split once.** Every
  existing split file lacks the new order stamp, so the next dojo build
  (a run or `--check`) rebuilds all of them; the two 141M-row 17lands
  splits dominate that time. Run one `--check` of a series-1 config to do
  it before the series. A baseline cache is no longer needed for speed.
- [ ] **Cache calibrated baselines** (only if startup is still slow after
  the re-split). Keyed by split-file identity, stored beside the splits,
  together with `LabelStats` (item above).
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
- [ ] **Decide the first-run dojo set.** Ready today: Gwent masks, STS
  metrics. Contrastive on FaB / Gwent / STS2 needs no metric. Add MSH draft
  and KTK game dojos after the split fixes below.
- [ ] **Generate more metrics only where they pay off.** A few more 17lands
  sets (game and draft) via `scripts/run_metrics.py`; Play-Gwent and FaB
  decklists still need metrics. Defer replay data and Dominion (no raw data
  or binder on disk).
- [x] **Fix split-file collisions** (2026-09-29). Every per-metric wrapper
  constructor (all `GenericDojo`-based ones - `paired_metric_dojos.py`'s 3
  Template Method bases plus the 22 standalone wrappers that build their
  own `DojoConfig`) now takes an optional `name: str | None = None` and
  threads it into `config=DojoConfig(name=name, ...)`, alongside the
  existing `rng_seed`/`strict_version_check`. `ContrastiveDojo` already had
  `name`. Callers building two dojos from same-stem files across
  expansion/format directories (e.g. `KTK/TradSealed/deck_win_prediction`
  and `MSH/PremierDraft/deck_win_prediction`) now pass distinct `name`s to
  avoid the collision; nothing constructs those dojos yet (no training
  driver, see section D), so no call site needed updating. Regression test:
  `tests/dojos/seventeenlands/game_data/test_game_deck_label_dojos.py`'s
  `TestNameAvoidsSplitFileCollision`.
- [x] **Make splits deterministic.** Existing split files are reused
  (`FileManagerParquet.splits_exist()`; `force_resplit` rebuilds), and
  `scripts/run_training.py` seeds every dojo's first split build from the
  config's `seed`.
- [x] **Scryfall ingestion fixes** (details in `Notes.md` section 8; the
  `keep_incoming` switch was needed by the lean-`raw_content` work in
  section B). Skip `token`/`double_faced_token`/`emblem`/`scheme`/
  `planar`/`vanguard`/`art_series`/`front_card` rows outright (~14% of
  the raw dump) - found while debugging the 2026-09-24 deck box
  regeneration: 95 real card names collide with a same-named row in one
  of these layouts (e.g. a "Tarmogoyf" token alongside the real
  creature), making `get_by_name` ambiguous and silently substituting
  the Unknown sentinel for a real, common card. Fixed in
  `scryfall/ingestion_stage.py`'s `ingest()`; docstring refreshed.
- [x] **Confirm every `all_cards()` consumer excludes the `Unknown`
  sentinel.** Checked: none of the on-disk binders currently have the
  sentinel persisted (`ensure_unknown_card` + `binder.save()` only
  happens inside `run_deck_box_ingestion.py`, not yet re-run since the
  last card re-ingest). But it's a real, imminent bug: every one of
  gwent_one's 8 masked-field metrics and dominiontabs's
  `CostRegressionMetric` indexes `card.raw_content[...]` directly with
  no empty check, so the *next* deck-box ingestion run (needed anyway,
  see below) would seed the sentinel into `gwent.jsonl`, and the next
  metrics re-scan would then crash with `KeyError`. Fixed at the base
  class: `MaskedFieldMetric.scan()` and
  `MaskedFieldRegressionMetric.scan()` now skip any card with empty
  `raw_content` before calling `_is_eligible()`, so no per-subclass
  fix is needed; tested (`test_masked_field_metric.py`, new
  `test_masked_field_regression_metric.py`).
- [x] **Exercise every chosen dojo once before training** (2026-09-27).
  `src/training/preflight.py`'s `preflight_dojo()` constructs one TRAIN
  batch, builds random embeddings nested exactly like the batch's own
  `GenericCard` structure (no encoder, no GPU needed), and runs
  `compute_loss` against them - the same shape mismatch or 0-example
  staleness (see the gwent_one incident in section E) surfaces in
  seconds instead of a quarantined round. `scripts/preflight_dojos.py`
  runs it over the gwent_one and sts_gg first-run candidates; not yet
  run against real data (no `data/` in this cloud session).
- [x] **Let label dojos take the per-game card-field augmentations (D1)**
  (2026-10-01, T5). Every catalog dojo now gets its game's defaults
  (`augmentation_defaults.py`); a metric dojo runs them after its own task
  mods (`GenericDojo.append_mods`), and `mods:` replaces them per dojo.
  Default-on: card-field mods only remove information and never move a
  card, so a masked target stays masked and option order and group
  membership are kept (tested per game in `test_mod_specs.py`). Preflight
  with defaults on: `scryfall.rarity_mask`, `sts2_runs.win` and
  `final_decks.held_out_card_gwent` 3/3 OK, every mod firing. Deck mods
  (D2) stay opt-in (`DECK_MOD_GROUPS`). Open: on a mask dojo whose target
  is also a row of its game's mask table (e.g. `gwent_one.faction_mask`,
  `spire_codex.color_mask`), that row only re-masks the target, so up to
  half the draws do nothing yet count as changed in the tally; per-dojo
  tables would fix it if it matters.
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

## D. Training driver

- [x] **Write `scripts/run_training.py`** (2026-09-29). YAML config
  (`configs/training/`, parsed by `run_config.py`) plus repeatable
  `--set key.path=value` overrides; dojos named by `dojo_catalog.py` keys
  (gwent_one masks, sts_gg card and deck dojos); preflight before
  training; `--check` builds and preflights only. Run directory gets the
  config copy, `rounds.csv`, `checkpoints.csv` and checkpoints. The config
  seed also seeds each dojo's split shuffle. `--check` on
  `gpu_smoke.yaml` passes on real data (8/8 gwent dojos). Split files are
  now keyed by catalog name (`gwent_one.color_mask`), so the first build
  writes new ones.
- [x] **Decide checkpoint retention** (2026-09-30): only the best per phase
  plus `latest/` (rewritten every round) are kept. Still open: checkpoints
  hold the frozen LM too (~1.2 GB each, twice: `state.pt` and `encoder.pt`).
  Previously: `DirectoryCheckpointer(run_dir)`
  writes `<phase>_roundNNNN/{state.pt, encoder.pt, manifest.json}` on every
  new best round and never deletes old ones; each includes the full LM
  (several hundred MB for ModernBERT). Runs go in `data/runs/<name>/`
  (gitignored) with the config beside them as `run_config.yaml`; still
  open: a retention policy, and whether the manifest itself should carry
  the config (it stores only `repr(plan)`).
- [ ] **Set `max_batch_cost` sensibly.** It counts cards, not tokens, and a
  40-card deck counts as 40. Start small (~32) and raise it while watching
  VRAM.
- [x] **Add run logging** (2026-09-28). `CsvRunListener`
  (`recording/run_listener.py`) appends one row per (round, dojo) to a
  rounds CSV and one row per checkpoint to a checkpoints CSV, both openable
  mid-run (pandas, `tail -f`, a spreadsheet). Long format, not one column
  per dojo, since a phase's dojo set can differ from the next phase's. No
  TensorBoard listener; not needed until a run is long enough to want live
  plots rather than a CSV.
- [ ] **Decide about resume.** Not supported (tracker and RNG state are not
  saved); accept that for the first run or scope it.
- [x] **Multi-card models in the run config** (2026-10-09). `model:` takes
  `embedding_head`, its sizes, and an optional `group_attention:` block
  (`src/training/README.md`); the driver assembles the model and prints
  its trainable parameter count. Size-matched `residual_mlp` arms for
  series 1 (card_embedding_size 256, mlp_hidden_dim 512, ModernBERT-base):
  single-card `mlp_num_blocks: 9` = 2,898,176 trainable parameters;
  multi-card `mlp_num_blocks: 3` plus 2 attention layers at `ffn_dim:
  1024` = 2,895,616.
- [ ] **Learn pre-norm vs post-norm** (`group_attention.norm_first`).
  Defaulted to pre-norm (`true`): LayerNorm before attention and the
  feed-forward block rather than after, the usual choice now because it
  trains stably without a learning-rate warmup, which the trainer does
  not have. torch's own default is post-norm. Read up on it, or ask for an
  explanation, and decide whether it deserves an ablation.

## E. Shakedown

- [x] **CPU tiny run** (2026-09-24): `scripts/smoke_test_training_loop.py`
  (new - distinct from the existing, pipeline-only `scripts/smoke_test.py`,
  which still needs the Section F fix/removal). A frozen
  `LinearProjectionCardModel` (ModernBERT-base + linear head, card_embedding_size
  32) against two real gwent_one dojos (`ColorMaskDojo`, `FactionMaskDojo`),
  `HoldoutSpec.no_holdout()`, 5 steps/round, 3 rounds. First time this
  project's `Trainer` has run against anything but fake dojos/a fake
  encoder. Ran clean: no quarantines, `stopped_early_reason: None`, loss
  fell every round (`color_mask` 0.99 -> 0.83, `faction_mask` 2.00 -> 1.94),
  all 3 checkpoints (`state.pt`, `encoder.pt`, `manifest.json`) written
  and readable.
  - Along the way, hit and fixed the exact staleness section C already
    flagged: gwent_one's 8 metric parquets
    (`data/metrics/gwent_one/*.parquet`) were built against the
    *pre-`--all`* card binder, so 0 of their `nocab_uuid`s resolved
    against the current one (checked directly: 0/20 sampled
    `color_mask.parquet` uuids found in the current `gwent.jsonl`) -
    every dojo silently yielded zero TRAIN/TEST examples and got
    quarantined, no exception raised. Fixed by regenerating:
    `PYTHONPATH=. python scripts/run_metrics.py --source gwent_one`
    (near-instant, in-memory scan over 1,260 cards - not the MSH-draft
    kind of risk). The other 5 games' metrics/deck boxes are presumably
    just as stale and will need the same regeneration before their dojos
    can be smoke-tested too.
- [x] **Fix the CPU/GPU device mismatch** (2026-09-29). Dojo heads live
  in the dojos, not the model, and the losses built their targets on the
  CPU, so any GPU step would have raised and quarantined every dojo.
  `Dojo.move_head_to(device)` is new; `Trainer.run()` moves every head to
  the model's device (`precision.parameter_device`) before training.
  Losses build targets on `nocab_loss.device_of(decoder_output)`; the
  contrastive negative mask moves to the similarity matrix's device.
  Tested on the `meta` device, and on the RX 6800: 2 gwent_one dojos, fp16,
  15 steps, heads on `cuda:0`, loss fell every round, no quarantines.
- [x] **GPU run** (2026-09-30): `gwent_contrastive.yaml` trained end to end
  and was evaluated; findings, fixes and open items in
  `plans/archive/first_train_shakeout.md`. Frozen ModernBERT, head-only phase; inspect loss curves
  and saturation behavior. First candidate (decided 2026-09-30):
  `configs/training/gwent_contrastive.yaml`, one single-card contrastive
  dojo with Gwent's default augmentations; `--check` passes on real data
  (48k train decks; all three default mods fire, none fail).
- [ ] **Trainable-LM phase later**, once VRAM is measured. Consider
  gradient checkpointing and a fp16 scaler.

## F. Housekeeping

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

- [x] **Fix or remove stale files** (2026-09-26): `scripts/smoke_test.py`
  and `src/training/demo_training_loop.py` deleted; `src/README.md`
  updated.
- [ ] **Update `requirements.txt` and `scripts/check.sh`.** Document the
  ROCm torch install (command in section A); `check.sh` uses `python3`,
  which is fragile on this Windows setup. Also decide whether the project
  `venv/` should become the ROCm one, or whether the ROCm venv is only for
  training runs.
- [ ] **Run the test suite in the ROCm venv** (dev tools like black, flake8
  and mypy are not installed there yet), to confirm nothing depends on
  the CPU-only torch build.
