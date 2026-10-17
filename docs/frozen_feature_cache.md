# Frozen feature cache: architecture (DRAFT, under discussion)

Status: draft 2026-10-17, branch `cloud_env_oct16`. Open questions are in
section 7; nothing here is built yet. When the questions are settled this
document is the brief for the implementation agent.

## 1. Goal

Stop running the frozen language model (ModernBERT) on the same cards every
step. While the text encoder is frozen, every stage up to the first trainable
one is a pure function of the card's serialized text, so its output can be
computed once per distinct text and reused. The main target is contrastive
training, where the 2026-10-11_model_by_diet_r2 run could not get through
the available decks.

Evidence that this is the bottleneck (r2, AMD RX 6800):

- Uncached training: about 1.4 s per step at 256 cards per step
  (`max_batch_cost: 256`); 380 min for `single_contrastive`. GPU compute
  busy at a 57% median.
- Cached prototype (fresh heads over the same frozen features): 3,000 steps
  plus TEST and VALIDATION in about 4-9 min per encoder once 51.5k cards
  were cached. That is an order of magnitude or more per step. It was never
  measured for full training (experiments/TODO.md, proposed experiment D).
- Deck I/O is not the bottleneck: about 1.7 ms per 16-deck group
  (benchmark from the batched `DeckBox.get_by_uuids` work on this branch).

## 2. What exists today

**In `src/`: nothing.** The cache is listed as not built in:

- `src/encoder_model/TODO.md`, "Cache frozen token states for the trainable
  poolings". It gives sizes for token-level fp16 states (Gwent ~175 MB,
  MTG ~9 GB) and notes that text-changing mods bypass any cache.
- `src/training/README.md`, the "Not built" list: "cached card encodings
  while the LM is frozen".
- `src/evaluation/TODO.md`, "Shared frozen features".
- `experiments/TODO.md`, proposed experiment D: "Train from cached frozen
  features... check the results match an uncached run."

**In the experiment scripts: a working prototype.**

- `experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_held_out.py`
  `FrozenFeatureCache`: an in-memory `dict[nocab_uuid -> pooled vector]`,
  filled lazily in chunks of 64 under fp16 autocast. It runs text encoder,
  then enrichment (token whitening), then pooling, and stores the pooled
  vector. `use_cache()` monkey-patches `model.card_pipeline.embed` so only
  the embedding head runs. It is shared across encoders, since every r2
  cell has the same frozen prefix. Reused by `extrinsic_gwent.py` and
  `zero_shot_contrastive.py`.
- `probe_cells.py`: the same idea persisted to disk (`pooled.pt` holding
  the uuid list and a tensor), refusing to load if the card set changed.
- Exactness checks: `probe_cells.py` and `intrinsic_suite.py` compare the
  cached path with the real model on 64 cards and assert a min cosine
  > 0.999. Measured: 1.00000.

**Prototype limitations the real version must fix:**

1. It is keyed by `nocab_uuid`. Card mods
   (`src/dojos/mods/card_field_mods.py`) return
   `dataclasses.replace(card, raw_content=...)`, the same uuid with
   different text. With mods on, the prototype silently serves the
   unmodded features and turns augmentation off. Confirmed for r2: its
   configs have no `mods:` section, so every catalog dojo ran its game's
   default train-only mods, and the fresh heads trained without
   augmentation. TEST and VALIDATION were unaffected, since mods are
   train-only (question Q8).
2. The key does not cover the frozen stages themselves (checkpoint,
   `max_length`, whitening stats). A cache that outlives a
   `load_state_dict` or a refit of the whitening is stale and goes
   unnoticed.
3. Monkey-patching, no counters, unbounded growth, fp16 autocast hard-coded.
4. Pooled level only, so no help for trainable poolings (attention,
   transformer).

## 3. Facts that constrain the design

- **One choke point.** Every model (`SingleCardModel`, `MultiCardModel`)
  embeds cards only through `CardEmbeddingPipeline.embed()`
  (`src/encoder_model/card_embedding_pipeline.py`): serialize, then text
  encoder, then optional enrichment, then pooling, then head. Training,
  round evaluation and the evaluation suite all go through it.
- **What is frozen, by stage:**
  - `PretrainedTextEncoder(trainable=False)` (what `scripts/run_training.py`
    always builds) runs in eval mode, under `no_grad`.
  - `StaticEmbeddingTextEncoder` is trainable: never cacheable.
  - `TokenWhitening` has buffers only (mean, projection) and is saved in
    the state dict.
  - `MeanPooling` has no parameters.
  - `AttentionPooling` and `TransformerPooling` are trainable.
- **The LM never thaws mid-run today.** `Trainer` collects
  `_encoder_params` as the parameters with `requires_grad` at
  construction and toggles only those per phase
  (`src/training/trainer.py:169`). A thawed-LM phase is a listed TODO,
  not built.
- **Default mods make nearly every text unique.** Every game's default
  augmentation (`src/dojos/augmentation_defaults.py`) is `ShuffleKeysSpec`,
  then `WeightedFieldMaskSpec`, then `RandomKeyMaskSpec(0.1)`. A key
  shuffle alone gives up to n! orders per card, so with default mods a
  text-keyed cache hits almost never. The masks alone give a small finite
  set of variants per card (4-6 DropTable entries), which a cache can
  absorb.
- **Serialization is deterministic for a given card value**
  (`serialize_card_to_json`: compact JSON, insertion key order).
- **The r2 contrastive model is the target shape:** whitening, then mean
  pooling, then residual MLP head. Its entire prefix before the head is
  frozen, so the cache can sit after pooling.
- **Sizes.** Pooled is 768 x fp32 = 3 KB per card: 51.5k cards is about
  158 MB (79 MB in fp16), small enough to keep on the GPU. Token level is
  about seq_len times larger (MTG ~9 GB fp16): host RAM or disk, not VRAM.
- **Bucketing.** `PretrainedTextEncoder` pads per length bucket, so a
  card's fp16 output can differ in the last bits with batch composition.
  The cache freezes the first-seen value; the prototype measured this as
  harmless (cosine 1.00000).

## 4. Proposed design

### 4.1 Cache boundary: the frozen prefix, decided at build time

The pipeline computes its frozen prefix once, at construction:

- **text encoder**: frozen iff it declares itself frozen (a new
  `TextEncoder.is_frozen` property; `PretrainedTextEncoder` returns
  `not trainable`, `StaticEmbeddingTextEncoder` returns False).
- **enrichment / pooling**: frozen iff the stage has **no `nn.Parameter`
  at all** (buffers allowed). This rule is static on purpose. A rule based
  on `requires_grad` would be fooled by a head-only phase that temporarily
  freezes a trainable pooling, which would then go stale in the next phase.

This yields a cache level:

| Level | Condition | Cached value | Per step still runs |
|---|---|---|---|
| `POOLED` | encoder, enrichment and pooling all frozen | one vector per card | head (+ group attention) |
| `TOKENS` | encoder and enrichment frozen, pooling trainable | `TokenEncoding` rows | pooling, head |
| `NONE` | text encoder trainable | none | everything |

### 4.2 Key: content, not identity

`key = blake2b(serialized_text, digest_size=16)`, computed from the same
string the text encoder would read. This is correct under any mod: a masked
card is a different key, an unmodded one hits. Serializing every card every
step is cheap next to an LM forward pass. It is already done today.

Separately, a **prefix fingerprint** names the frozen stages: checkpoint id,
tokenizer `max_length`, and a hash of the enrichment's buffers. Every entry
belongs to one fingerprint. It is required for anything persisted to disk,
and in memory it lets the cache refuse to serve after a `load_state_dict`
changes the whitening.

### 4.3 Components

- `src/encoder_model/frozen_feature_cache.py`, new: `FrozenFeatureCache`.
  - Stores rows in a preallocated, chunk-grown tensor plus a
    `dict[key -> row]`.
  - `lookup(keys) -> (hit_rows, miss_positions)`, then
    `insert(keys, values)`, then `gather(rows) -> Tensor`.
  - Counters: hits, misses, entries, bytes; per-round deltas.
  - Optional `max_entries` (refuse to insert past it rather than evict;
    see Q4).
  - Not an `nn.Module` and not a buffer: never in a state dict or
    checkpoint.
- `CardEmbeddingPipeline`:
  - takes an optional cache;
  - `embed()` splits the batch into hits and misses;
  - it runs the frozen prefix only on the misses (under `no_grad`, in the
    caller's autocast);
  - it inserts the misses and reassembles in input order;
  - it runs the remaining trainable stages on the full batch.
  - Behaviour with no cache is unchanged. For `TOKENS` the cached rows
    have ragged lengths and are re-padded into one `TokenEncoding` (reuse
    `_merge_bucketed_encodings`' padding logic).
- Invalidation: the pipeline clears its cache when the fingerprint changes
  (checked lazily on `embed`, or by a `load_state_dict` post-hook).
- `ModelSpec` and the run config get a `model.feature_cache:` section (shape
  in Q3), parsed in `config_parsing/model_parsing.py` and built in
  `scripts/run_training.py:build_model`.
- Observability:
  - The trainer logs hit rate, entries and size per round at INFO.
  - Preflight (`src/training/preflight.py`) warns when the cache is on
    and a dojo it feeds has a text-changing mod that makes texts unique
    (`ShuffleKeysSpec`, `RandomKeyMaskSpec`), naming the dojo and the
    expected effect.
- Startup self-check, as in the prototype: on N cards, compare the cached
  and uncached paths and fail the run below a cosine threshold.

### 4.4 What gets faster, and what becomes the next bottleneck

With a POOLED hit, a step is a gather plus a small MLP. So:

- Batch construction becomes the critical path: deck reads, pair
  construction, card lookup, mods, serialization and hashing. That is item
  #4 (a background prefetch thread per batch stream), now worth doing. A
  57% GPU-busy median is already a hint.
- `max_batch_cost: 256` was sized for LM VRAM. Without the LM in the step,
  far larger contrastive batches (more negatives) fit. Re-tune this
  separately.

## 5. Work breakdown (proposed order, one PR-sized step each)

1. **POOLED in-memory cache.**
   - `FrozenFeatureCache`, `TextEncoder.is_frozen`, frozen-prefix
     detection, `embed()` integration.
   - Config section, counters and round logging, preflight mod warning.
   - Unit tests (section 6).
   - This alone covers the r2 contrastive shape.
2. **Startup self-check and an experiment-D style comparison script**, so
   a GPU run can confirm the speedup and matching losses against an
   uncached run.
3. **Persistence / prefill.**
   - `scripts/precompute_frozen_features.py` writes the unmodded features
     of every card the run's games can see, under the prefix fingerprint
     (e.g. `data/feature_cache/<fingerprint>.pt`).
   - Runs and evaluation scripts load it at startup. This gives a warm
     start and shared reuse across runs and evals.
4. **TOKENS level** for attention/transformer pooling: host-RAM ragged
   fp16 store, pinned-memory transfer per batch, possibly disk-backed for
   MTG.
5. **Follow-ons, out of scope here:**
   - batch prefetch (#4);
   - re-tuning the batch budget;
   - deciding the augmentation policy for cached runs (Q1);
   - moving the evaluation scripts onto the library cache.

## 6. Testing (CPU-only, no data)

- A stub `TextEncoder` that counts calls, plus tiny real poolings and heads:
  - the second `embed` of the same cards makes zero encoder calls;
  - outputs equal the uncached pipeline's exactly (fp32 on CPU);
  - mixed hit/miss batches keep input order;
  - duplicates within a batch are computed once.
- Modded cards (masked, shuffled keys) get distinct keys; an unmodded card
  hits.
- Frozen-prefix detection: whitening plus mean gives POOLED, attention
  pooling gives TOKENS, a trainable or static encoder gives NONE.
- Changing the whitening buffers (or `load_state_dict`) invalidates.
- The model's `state_dict()` keys are identical with and without a cache.
- Gradients still reach the head (and, at TOKENS, the pooling) through
  cached rows.
- Config parsing round-trip; preflight warns on `ShuffleKeysSpec`.

## 7. Open questions (to settle in chat)

- **Q1. Augmentation under the cache.** The default mods shuffle keys, so
  the hit rate is about 0. Options:
  - (a) turn mods off for cached contrastive runs (config only, your
    plan);
  - (b) keep the field masks but drop `ShuffleKeys` / `RandomKeyMask`
    from the defaults or per run; this is a bounded variant set, so the
    cache converges;
  - (c) cache only unmodded cards and compute modded ones live (costs
    LM time for every modded card).
  - Does masking the deck-defining field still matter for contrastive
    quality? r2 ran with it on.
- **Q2. TOKENS priority.** Are attention or transformer pooling runs
  coming soon (encoder_model TODO), or can step 4 wait?
- **Q3. Config shape and default.** For example
  `model.feature_cache: {enabled: true, level: auto, max_entries: null,
  prefill_path: null}`. Default off at first, flipped on once experiment D
  confirms matching losses?
- **Q4. Bounds.** Unbounded (fine at 158 MB pooled), a hard
  `max_entries` that stops inserting, or LRU eviction (more code, matters
  mainly at TOKENS)?
- **Q5. Storage dtype and device for POOLED.** fp32 on the GPU (simplest,
  exact; whitening already outputs fp32) or fp16 (half the memory, at the
  cost of rounding)?
- **Q6. Prefill vs lazy fill.** Lazy fill means early steps are as slow as
  today and get faster as the cache fills. Prefill gives a stable step
  time from step 1 and a known size: roughly 51.5k cards at a few hundred
  cards/s, a few minutes (estimate; not measured). Should prefill be in
  step 1 or wait for step 3?
- **Q7. Exactness bar.** Is the first-seen value under bucketed fp16
  padding acceptable (prototype: cosine 1.00000), with a cosine > 0.999
  startup check as the guard?
- **Q8. The r2 extrinsic results.** The prototype's uuid-keyed cache
  served unmodded features to fresh-head TRAIN batches whose dojos had
  default mods on (section 2), so those heads trained without
  augmentation. Evaluation splits were unmodded either way. Add a note to
  the r2 notebook's fresh-head method?
