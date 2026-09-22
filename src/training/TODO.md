# TODO (training): getting to a first real training run

Written 2026-09-21 from a review of the codebase, the hardware and the
data on disk. The trainer (`trainer.py`) is implemented and tested against
fake dojos and a fake encoder; nothing here has run on real dojos, a real
text encoder or a GPU yet. Work through the sections roughly in order:
A blocks everything, and B and C can proceed in parallel once A is done.

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
- [ ] **Add a fp16 `GradScaler` to the trainer.** fp16 is the fast path
  (see below) and the trainer has none (README "Not built"). A GradScaler
  worked in the benchmarks (finite losses).

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

- [ ] **Choose the encoder.** Suggested: `answerdotai/ModernBERT-base`
  (Apache-2.0, 149M params, 8192-token context, cased, code-heavy
  pretraining). No open model is truly JSON-tuned; this is the closest.
  Runner-up: `jinaai/jina-embeddings-v2-base-code` (needs
  `trust_remote_code`). Skip `nomic-ai/modernbert-embed-base`
  (retrieval-tuned, needs prefixes).
- [ ] **Lean `raw_content` in each ingestion stage** (decision 2026-09-21:
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
  - [ ] **re-ingest the binders.** Decision 2026-09-21: no legacy-data
    migrations; `data/final/cards` was deleted and is simply rebuilt with
    `PYTHONPATH=. python scripts/run_card_binder_ingestion.py` (all
    sources, or `--source <name>`). New binders mint new `nocab_uuid`s, so
    every metric parquet, split and deck box must be regenerated after;
  - [x] converted (details in `plans/card_content_conversion.md`):
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
- [ ] **Set `max_length` (384 after the serialization fix; not 1024)** and let
  `PretrainedTextEncoder` accept model kwargs. Verified: ModernBERT loads on
  the GPU with `attn_implementation="sdpa"`. `reference_compile` no longer
  exists in transformers 5.x (passing it raises `TypeError`), so it is not
  needed. Once the serialization is trimmed, most cards fit in 512 tokens
  (MTG median ~140 after the lean-`raw_content` work), and shorter caps are much
  cheaper on this GPU.
- [ ] **Batch by similar length.** `PretrainedTextEncoder.encode` pads to
  the longest text in the batch; the trainer's batches mix games and card
  sizes, which cost a ~4x slowdown in measurement. Options: sort or bucket
  within a batch's cards, or a token-aware `cost_of` (README "Not built").
- [ ] **Check masking dojos against the new serialization.** A masked field
  such as `set` or `rarity` must not leak through another key (e.g.
  `set_name`).
- [x] **Measure frozen-LM throughput.** Done, see "GPU measurements" in
  section A: ~100 cards/s length-sorted. The trainer re-encodes every card
  every step; caching frozen-LM outputs is "not built" and is worth it only
  if this proves too slow (masking/key-shuffle mods change the input, so
  caching only fits unmodified cards).

## C. Data and dojo readiness

- [ ] **Build a real inventory.** `docs/metric_dojo_inventory.csv` is stale
  (references paths and dojo names that don't exist). Script it: metric
  parquet, dojo class, row count, whether cards resolve in the binder.
- [ ] **Report card-loading health per game.** Unresolved names, alias
  collisions, 17lands-name -> Scryfall-UUID resolution rate for MSH and
  KTK. Check STS2: 578 cards against 7.8k decks.
- [ ] **Decide the first-run dojo set.** Ready today: Gwent masks, STS
  metrics. Contrastive on FaB / Gwent / STS2 needs no metric. Add MSH draft
  and KTK game dojos after the split fixes below.
- [ ] **Generate more metrics only where they pay off.** A few more 17lands
  sets (game and draft) via `scripts/run_metrics.py`; Play-Gwent and FaB
  decklists still need metrics. Defer replay data and Dominion (no raw data
  or binder on disk).
- [ ] **Fix split-file collisions.** Dojo `name` is the parquet stem and
  splits go to flat `data/splits/<stem>_*.parquet` (see the TODO at
  `src/dojos/generic/generic_dojo.py:73`). `KTK/TradSealed/deck_win_prediction`
  and `MSH/.../deck_win_prediction` collide, and duplicate names make
  `Trainer` raise. Namespace dojo names and split prefixes by expansion and
  format.
- [ ] **Make splits deterministic and cheap.** Every dojo constructor
  re-runs `make_splits`, streaming the whole parquet (1.8 GB for
  `pool_conditioned_pick`). With `rng_seed=None` a restart reshuffles the
  splits, so runs are not comparable. Pass seeds; skip rebuilding splits
  that already exist.
- [ ] **Scryfall ingestion fixes** (details in `Notes.md` section 8; the
  `keep_incoming` switch is needed by the lean-`raw_content` work in
  section B, the rest is deferred): skip `art_series` and `front_card`
  objects (~6.5% of the MTG binder are not cards); decide about
  tokens/emblems/schemes/planar; refresh the stage's stale docstring.
- [ ] **Confirm every `all_cards()` consumer excludes the `Unknown`
  sentinel** (empty `raw_content`, present in the MTG, FaB, Gwent and STS2
  binders): corpus-scan metrics, contrastive, masked-field dojos.
- [ ] **Exercise every chosen dojo once before training.** Construct it,
  pull one TRAIN batch, run `compute_loss` on random embeddings, confirm
  head dims match `card_embedding_size`.

## D. Training driver (does not exist yet)

- [ ] **Write `scripts/run_training.py`.** Load binders and deck boxes,
  build the model, dojos and `TrainingPlan`, call `Trainer.run()`. Today
  `README.md` only has a snippet.
- [ ] **Add a config file (YAML/JSON) plus CLI overrides.** Dojo list and
  diet, phases, learning rates, `HoldoutSpec`, `max_batch_cost`, run
  directory, seed. Dojo selection is currently `Phase.dojo_names` matched
  against whatever dojos were constructed.
- [ ] **Decide checkpoint output.** `DirectoryCheckpointer(run_dir)` writes
  `<phase>_roundNNNN/{state.pt, encoder.pt, manifest.json}` on every new
  best round and never deletes old ones; each includes the full LM (several
  hundred MB for ModernBERT). Pick a `runs/` location, a retention policy,
  and put the resolved config in the manifest (it stores only
  `repr(plan)` today).
- [ ] **Set `max_batch_cost` sensibly.** It counts cards, not tokens, and a
  40-card deck counts as 40. Start small (~32) and raise it while watching
  VRAM.
- [ ] **Add run logging.** Only `LoggingRunListener` exists; a CSV or
  TensorBoard listener would help for overnight runs.
- [ ] **Decide about resume.** Not supported (tracker and RNG state are not
  saved); accept that for the first run or scope it.

## E. Shakedown

- [ ] **CPU tiny run.** `FromScratchCardModel` (or frozen DistilBERT), two
  Gwent dojos, ~20 steps per round, a few rounds. Confirms the loop,
  checkpoints and manifest against real dojos.
- [ ] **GPU run.** Frozen ModernBERT, head-only phase; inspect loss curves
  and saturation behavior.
- [ ] **Trainable-LM phase later**, once VRAM is measured. Consider
  gradient checkpointing and a fp16 scaler.

## F. Housekeeping

- [ ] **Fix or remove stale files.** `scripts/smoke_test.py` calls the
  removed `training_data()` API; `src/training/demo_training_loop.py` and
  the "training" paragraph in `src/README.md` are stale.
- [ ] **Update `requirements.txt` and `scripts/check.sh`.** Document the
  ROCm torch install (command in section A); `check.sh` uses `python3`,
  which is fragile on this Windows setup. Also decide whether the project
  `venv/` should become the ROCm one, or whether the ROCm venv is only for
  training runs.
- [ ] **Run the test suite in the ROCm venv** (dev tools like black, flake8
  and mypy are not installed there yet), to confirm nothing depends on
  the CPU-only torch build.
