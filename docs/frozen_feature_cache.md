# Frozen feature cache: architecture (DRAFT, under discussion)

Status: draft, branch `cloud_env_oct16`. Decisions settled so far are in
section 4. Questions still open are in section 9. Nothing here is built
yet. Once section 9 is settled, this document is the brief for the
implementation agent.

## 1. Goal

Stop running the frozen language model (ModernBERT) on the same card text
every step. While the text encoder is frozen, every stage up to the first
trainable one is a pure function of the card's serialized text. Its output
can be computed once per distinct text and reused. The main target is
contrastive training: the 2026-10-11_model_by_diet_r2 run could not get
through the decks available.

Evidence that the LM is the bottleneck (r2, AMD RX 6800):

- Uncached training: about 1.4 s per step at 256 cards per step
  (`max_batch_cost: 256`), 380 min for `single_contrastive`, with GPU
  compute busy at a 57% median.
- Cached prototype (fresh heads over the same frozen features): 3,000
  steps plus TEST and VALIDATION in about 4-9 min per encoder, once 51.5k
  cards were cached. That is an order of magnitude or more per step.
  Never measured for full training (experiments/TODO.md, proposed
  experiment D).
- Deck I/O is not the bottleneck: about 1.7 ms per 16-deck group
  (benchmark from the batched `DeckBox.get_by_uuids` work on this branch).

## 2. What exists today

**In `src/`: nothing.** Listed as not built in
`src/encoder_model/TODO.md` ("Cache frozen token states for the trainable
poolings"), `src/training/README.md` ("cached card encodings while the LM
is frozen"), `src/evaluation/TODO.md` ("Shared frozen features") and
`experiments/TODO.md` (proposed experiment D).

**In the experiment scripts: a working prototype.**

- `experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_held_out.py`,
  `FrozenFeatureCache`:
  - an in-memory `dict[nocab_uuid -> pooled vector]`, filled lazily in
    chunks of 64 under fp16 autocast (text encoder, then token whitening,
    then mean pooling);
  - `use_cache()` monkey-patches `model.card_pipeline.embed` so only the
    embedding head runs;
  - reused by `extrinsic_gwent.py` and `zero_shot_contrastive.py`.
- `probe_cells.py` persists the same idea to disk (`pooled.pt`: a uuid
  list and a tensor).
- `probe_cells.py` and `intrinsic_suite.py` check the cached path against
  the real model on 64 cards: min cosine above 0.999, measured 1.00000.

**Prototype limitations the real version must fix:**

1. **It is keyed by `nocab_uuid`.** Card mods
   (`src/dojos/mods/card_field_mods.py`) return
   `dataclasses.replace(card, raw_content=...)`: the same uuid with
   different text. Confirmed for r2: its configs have no `mods:` section,
   so every catalog dojo ran its game's default train-only mods, and the
   fresh heads trained without augmentation. TEST and VALIDATION were
   unaffected, since mods are train-only (question Q6).
2. **The key does not cover the frozen stages.** Changing the checkpoint,
   `max_length` or the whitening stats leaves a stale cache that nothing
   detects.
3. Monkey-patching, no counters, unbounded growth, fp16 autocast
   hard-coded.
4. Pooled level only.

## 3. Facts that constrain the design

- **One choke point.** Every model (`SingleCardModel`, `MultiCardModel`)
  embeds cards only through `CardEmbeddingPipeline.embed()`
  (`src/encoder_model/card_embedding_pipeline.py`): serialize, then text
  encoder, then optional enrichment, then pooling, then head. Training,
  round evaluation and the evaluation suite all pass through it.
- **What is frozen today, by stage:**
  - `PretrainedTextEncoder(trainable=False)` (what
    `scripts/run_training.py` always builds) runs in eval mode under
    `no_grad`; `StaticEmbeddingTextEncoder` is trainable.
  - `TokenWhitening` has buffers only (mean, projection), saved in the
    state dict.
  - `MeanPooling` has no parameters.
  - `AttentionPooling` and `TransformerPooling` are trainable.
- **The LM never thaws mid-run today.** `Trainer` toggles only the
  parameters that had `requires_grad` at construction
  (`src/training/trainer.py:169`).
- **Mods, three kinds** (checked against the code):
  1. **Required task masks**: `MaskTargetKeyMod(train_only=False)` in the
     paired-metric dojos, `rarity_tier` and dominion cost regression.
     Deterministic: a card always masks to the same text, on every split.
  2. **Deck/group mods** (card dropout, subsample, shuffle deck,
     duplicate collapse) never change card text.
  3. **Random train-only augmentation.** Every game's default
     (`src/dojos/augmentation_defaults.py`) is `ShuffleKeysSpec`, then
     `WeightedFieldMaskSpec`, then `RandomKeyMaskSpec(0.1)`.
     - The field masks draw from 4-6 DropTable entries, so each card has
       a small, bounded set of variants.
     - A key shuffle gives up to n! orders, and the random key mask gives
       2^k subsets: effectively unique text on every draw.
- **Serialization is deterministic** for a given card value
  (`serialize_card_to_json`: compact JSON, insertion key order).
- **The r2 contrastive model is the target shape:** whitening, then mean
  pooling, then residual MLP head. Everything before the head is frozen.
- **Sizes:**
  - pooled: 768 x fp32 = 3 KB per card; 51.5k cards is about 158 MB, and
    fits on the GPU;
  - raw token states: about seq_len times more (MTG about 9 GB in fp16),
    which means host RAM or disk.
- **Bucketing:** `PretrainedTextEncoder` pads per length bucket, so a
  card's fp16 output can differ in the last bits with batch composition.
  The cache keeps the first value it sees; the prototype measured this as
  harmless (cosine 1.00000).

## 4. Decisions

### 4.1 Two cache levels: after BERT and after pooling

| Level | Caches | Valid when | Per step still runs | Stale when |
|---|---|---|---|---|
| `tokens` | raw text-encoder `TokenEncoding` rows (after BERT, before any enrichment) | text encoder frozen | enrichment, pooling, head | checkpoint or `max_length` change |
| `pooled` | one vector per card (after pooling) | text encoder frozen, and enrichment and pooling have no `nn.Parameter` (buffers allowed) | head (+ group attention) | the above, or any enrichment buffer change |

- **`tokens` caches before enrichment, on purpose.** It depends only on
  the checkpoint and tokenizer length. One persisted `tokens` cache
  therefore serves every ModernBERT-base experiment, whatever its
  whitening or pooling. That includes a future learned whitening, or a
  learned pooling with a raw unpooled pass-through.
- **`pooled`** is the fast path for the current fully-frozen shape. It is
  built first.
- **Validity is decided statically, when the model is built.**
  - Text encoder: a new `TextEncoder.is_frozen` property
    (`PretrainedTextEncoder` returns `not trainable`;
    `StaticEmbeddingTextEncoder` returns False).
  - Enrichment and pooling: frozen iff they have no `nn.Parameter` at
    all. A rule based on `requires_grad` would be fooled by a head-only
    phase that temporarily freezes a trainable pooling.
- **Requesting an invalid level fails at build time.** For example,
  `pooled` with attention pooling, or any level with a trainable text
  encoder. It never silently falls back.
- **Not built: a post-whitening, pre-pooling level.** It is worth adding
  only if a fixed enrichment ever becomes expensive. Until then, `tokens`
  plus re-running whitening (one matmul per token) covers it. Left as a
  TODO.

### 4.2 Key: the card's text, never its identity

`key = blake2b(serialized_text, digest_size=16)`, computed from the exact
string the text encoder would read. Mods then need no special handling:

- every mod is applied exactly as today;
- a masked card is a different key, an unmodded card hits;
- a required task mask adds one entry per card per masked field, and hits
  normally after that.

A **prefix fingerprint** names the frozen stages a level depends on:

- `tokens`: checkpoint and `max_length`;
- `pooled`: those, plus a hash of the enrichment buffers and the pooling
  class.

Every entry belongs to one fingerprint. The cache refuses to serve once
the fingerprint changes, for example after a `load_state_dict` that
replaces the whitening. The fingerprint is required for anything
persisted to disk.

### 4.3 Mod policy: cache plus light mods (option b)

The cache is correct under any mod by construction. Mods only change its
hit rate, which is an augmentation choice per experiment:

- **Recommended for cached contrastive runs:**
  - keep the field masks (`WeightedFieldMaskSpec`); hiding the
    deck-defining field is the augmentation that matters for contrastive;
  - drop `ShuffleKeysSpec` and `RandomKeyMaskSpec` through the run
    config's `mods:` section.
- **Documentation:** `src/training/README.md` gets a prominent note:
  "intensive text mods (key shuffles, random key masks) make almost every
  card text unique, so the cache cannot help and only costs memory: turn
  the cache off for such runs, or use light mods with the cache on."
- **Guard rails:**
  - Preflight warns when a dojo fed by a cached model runs a mod that
    makes texts unbounded (`ShuffleKeysSpec`, `RandomKeyMaskSpec`).
  - Hit rate is logged per dojo each round, so the effect is measured,
    not assumed.
- **Later, if key-order robustness turns out to matter:** a cache-friendly
  key shuffle that draws from K fixed permutations per card, seeded by
  card uuid. The variant count then stays bounded. Not built now.

## 5. When the cache helps (the deciding parameters)

Standard cache engineering applies. What decides the benefit:

- **Working set** = sum over games of (cards seen) x (distinct text
  variants per card, summed over the dojos that change text) x (bytes per
  entry).
- **Capacity:** VRAM for `pooled`, host RAM or disk for `tokens`.
- **Hit rate** ~= 1 - (first sightings of a text) / (texts embedded).
  - With no or light mods it approaches 1 after the first pass over the
    card pool.
  - With unbounded mods it stays near 0.

Rough sizes:

- **`pooled`, light mods:** 51.5k cards x ~5 mask variants x 3 KB is about
  0.8 GB in fp32. That fits on a 16 GB card next to the model, so no
  eviction is needed.
- **`tokens`:** the same working set is roughly 150x larger. It needs a
  capacity bound, an eviction policy, and probably chunked disk storage
  loaded on demand. That design is deferred to the `tokens` step (section
  7, step 4).

## 6. Components

- **`src/encoder_model/frozen_feature_cache.py`** (new):
  - `FrozenFeatureCache(level, fingerprint, capacity)`;
  - a preallocated, chunk-grown tensor plus a `dict[key -> row]` (the
    `pooled` store);
  - `lookup(keys) -> (rows, miss_positions)`, `insert(keys, values)`,
    `gather(rows)`;
  - counters for hits, misses, entries and bytes, with per-round deltas;
  - at `capacity` it stops inserting (later misses are computed live and
    not stored) and logs once;
  - not an `nn.Module` or buffer, so it never enters a state dict or
    checkpoint.
- **`CardEmbeddingPipeline.embed()`:**
  - takes an optional cache and splits each batch into hits and misses
    (duplicates computed once);
  - runs the cached prefix on misses only (`no_grad`, in the caller's
    autocast) and inserts them;
  - reassembles in input order, then runs the remaining stages on the full
    batch;
  - for `tokens`, re-pads ragged rows into one `TokenEncoding` (reusing
    `_merge_bucketed_encodings`' padding logic);
  - with no cache, behaviour is unchanged.
- **Config:** a `model.feature_cache:` section in `ModelSpec`, parsed in
  `config_parsing/model_parsing.py` and built in `scripts/run_training.py`.
  Shape is question Q1.
- **Observability:** the trainer logs hit rate, entries and size per round
  at INFO; preflight emits the mod warning (section 4.3).
- **Startup self-check** (as in the prototype): compare the cached and
  uncached paths on N cards; fail the run below a cosine threshold (Q5).

## 7. Work breakdown (one PR-sized step each)

1. **`pooled` in-memory cache:**
   - `FrozenFeatureCache`, `TextEncoder.is_frozen`, level validation,
     `embed()` integration;
   - config section, counters and round logging, the preflight mod
     warning, the README note;
   - unit tests (section 8).
2. **Startup self-check, plus an experiment-D comparison script**, so a GPU
   run can confirm the speedup and matching losses against an uncached
   run.
3. **Persistence and prefill.**
   - `scripts/precompute_frozen_features.py` writes the unmodded features
     of every card the run's games can see, under the fingerprint (e.g.
     `data/feature_cache/<level>/<fingerprint>/`).
   - Runs and evaluation scripts load it at startup.
4. **`tokens` level:** host-RAM or disk store with a capacity bound and
   eviction (likely chunked, loaded on demand), pinned-memory transfer per
   batch.
5. **Follow-ons, out of scope here:**
   - batch prefetch on a background thread: once the LM leaves the step,
     batch construction becomes the critical path;
   - re-tuning `max_batch_cost` (sized for LM VRAM; larger contrastive
     batches give more negatives);
   - the cache-friendly key shuffle;
   - a post-whitening level;
   - moving the experiment scripts onto the library cache.

## 8. Testing (CPU-only, no data)

- With a stub `TextEncoder` that counts calls, and tiny real poolings and
  heads:
  - a repeat `embed` makes zero encoder calls;
  - outputs equal the uncached pipeline's exactly (fp32 on CPU);
  - mixed hit/miss batches keep input order;
  - duplicates in a batch are computed once.
- Modded cards (masked, shuffled keys) get distinct keys; a deterministic
  task mask hits on its second sighting; an unmodded card hits.
- Level validation:
  - whitening + mean allows `pooled` and `tokens`;
  - attention pooling allows only `tokens`;
  - a trainable or static text encoder allows neither.
- Fingerprint: changing the whitening buffers (or `load_state_dict`)
  invalidates `pooled` but not `tokens`.
- The model's `state_dict()` keys are identical with and without a cache.
- Gradients reach the head, and at `tokens` the enrichment and pooling,
  through cached rows.
- At capacity the cache stops inserting and still returns correct
  outputs.
- Config parsing round-trip; preflight warns on `ShuffleKeysSpec`.

## 9. Open questions

- **Q1. Config shape and default.** For example `model.feature_cache:
  {level: pooled | tokens | none, capacity_bytes: null, prefill_path:
  null}`. Default `none` until experiment D confirms matching losses?
- **Q2. Storage dtype for `pooled`.** fp32 (exact; whitening already
  outputs fp32) or fp16 (half the memory, at the cost of rounding)?
- **Q3. Prefill vs lazy fill.** With lazy fill, the first pass is as slow
  as today. Prefill gives a stable step time from step 1, at a cost of
  roughly 51.5k cards at a few hundred cards/s, a few minutes (estimate,
  not measured). In step 1, or wait for step 3?
- **Q4. The store's implementation.**
  - For `pooled`: a hand-rolled tensor table (about 50 lines; batched GPU
    gathers, no per-entry Python objects or pickling).
  - For `tokens`: a library (`diskcache`, or numpy memmap or safetensors
    shards), decided in step 4.
  - Agree?
- **Q5. Exactness bar.** Is the first-seen value under bucketed fp16
  padding acceptable (prototype: cosine 1.00000), with a cosine > 0.999
  startup check as the guard?
- **Q6. The r2 extrinsic results.** Add a note to the r2 notebook's
  fresh-head method saying the prototype trained those heads without
  augmentation?
