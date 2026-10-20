# Embedding cache for faster training (plan)

Status: architecture agreed at a high level; nothing is built. This is the
document to hand to an implementation agent, once the open questions at
the end are settled.

## Goal

Increase training speed by caching the output of the model's frozen
stages (ModernBERT, token whitening, mean pooling), so the same card text
is not re-encoded every step. Today a step takes about 1.4 s and is
dominated by the frozen BERT forward; a cached prototype trained fresh
heads 3,000 steps in 4-9 minutes.

The cache is a new, config-driven part of the model: the cut point, tier
sizes, precision and policies are all chosen in config, and each can be
swapped without touching the model code.

## MVP (v1): post-pooling cache of unmodified cards, VRAM tier

The first target is narrow on purpose, so development iterates quickly,
but every piece is built behind the interfaces the later versions need.
The MVP must not have to be rewritten to add them.

**Scope:**
- Cut point: after pooling (one fixed-size vector per card).
- Workload: contrastive dojo diets, plus TEST and evaluation passes for
  any dojo. Mods are supported from the start (see "Mods" below), but the
  first runs use mods off; modded diets come in around v1.2.
- One tier: VRAM, with a configurable size.
- Both models: the cache sits behind `CardEmbeddingPipeline.embed()`,
  which the single-card and multi-card models both call.

**Pieces:**
1. **Store interface.** A small protocol the model talks to, so the
   policy and the storage technology are swappable:
   - `get_many(keys) -> (features, miss_mask)`: one batched lookup.
   - `put_many(keys, features)`: one batched insert; the store's policy
     decides what is admitted or evicted.

   The value type is a typed feature batch, not a bare `(N, dim)`
   tensor, so ragged token-level features (v1.3) fit later.
2. **VRAM tensor slab.** One preallocated tensor `(capacity, dim)` on the
   GPU, plus a CPU-side dict from key to row. A batch read is one
   `index_select`; keys and bookkeeping never touch the GPU, so lookups
   never force a CPU-GPU sync.
3. **Cache key.** A hash of the serialized text the encoder reads, plus
   a fingerprint of the frozen stages before the cut (checkpoint,
   tokenizer max length, whitening statistics). Not the `nocab_uuid`:
   mods keep the uuid and change the text, which is the bug the prototype
   had. Hashing the text is correct by construction, whatever mods ran.
4. **Lookup path in `embed()`:**
   1. Serialize and key each card; deduplicate keys within the batch
      (staples repeat inside one contrastive batch).
   2. `get_many` the unique keys.
   3. Run the frozen stages only on the misses, in one batch.
   4. `put_many` the results.
   5. Scatter hits and misses back into the batch's original order,
      then run the trainable stages (embedding head, group attention).
5. **Precision.** The storage dtype is a config option (fp32 by default;
   fp16 or bf16 to fit more). A miss returns the stored-precision value,
   not the live one, so a card's features never depend on whether it
   happened to hit or miss.
6. **Warming.** `model.warm(cards)` encodes and inserts in large batches
   ahead of time, off the training step's critical path. Two callers:
   - a tool that warms the cache with every card of a game, or every card
     a card-binder SQL query returns;
   - the prefetch worker (v1.1).

   Dojos never know the cache exists: the trainer (or prefetcher) calls
   `warm`, and the model owns the cache.
7. **Config validation.** Only frozen stages may come before the cut
   point. A post-pooling cache with trainable pooling (attention or
   transformer pooling) is rejected when the config is parsed, so the
   illegal state cannot be built.
8. **Exactness check.** Cached features match the live path within the
   batch-padding noise the live path already has (cosine ~1.0, as the
   prototype showed). This is a test, and experiment D checks it on a
   real run.

## After the MVP

- **v1.1 CPU RAM and disk tiers, plus prefetch.** Critical; built right
  after the MVP.
  - A pinned CPU RAM tier and a disk tier (memmap or LMDB), each with a
    configurable size. Entries evicted from VRAM drop to RAM, then disk.
  - The disk tier also persists encoded cards across runs and
    experiments, keyed by the frozen-stage fingerprint.
  - A prefetch worker builds the next k batches while the GPU trains
    (producer-consumer; today `trainer.py` calls `next_batch`
    synchronously), and warms their cards. The lookahead depth is a
    config knob. Start with a thread and a bounded queue; move to
    processes only if the GIL limits it.
- **v1.2 Swappable eviction and admission policies, and modded diets**,
  for experiments: always admit, admit on second hit (TinyLFU-style),
  freeze when full, LRU. These matter most for contrastive diets, whose
  card use is heavily skewed (staples in most decks, a long tail of rare
  cards), and for modded diets, where the deterministic-mod flag keeps
  noise texts out of the cache.
- **v1.3 More cut points:** after BERT (token states) and after token
  whitening. Needs ragged storage (one flat buffer plus offsets per
  card) and makes the RAM and disk tiers essential (roughly 400 KB per
  card at the token level, against 1.5 KB pooled at fp16). It enables
  training attention pooling on cached token states, or adding token
  features to the pooled vector.
- **v1.4 Experiments:**
  - D: train from cached features and match an uncached run
    (`experiments/TODO.md`).
  - F: a finite pool of K augmentations per card, which makes noise mods
    cacheable.
  - Locality-aware sampling: draw N decks (or one game's cards) per
    episode and train many batches from that working set, sized to the
    cache. This trades diet mixing for throughput, perhaps with a lower
    learning rate; prior art is GNS and ClusterGCN from GNN training.

## Mods

- **Supported in v1 for free.** The key is a hash of the text, so a
  modded card is just another key: one entry per distinct (card, mod)
  text. No mod-specific cache code is needed for correctness.
- Deterministic mods (required task masks, `MaskTargetKeyMod`) give a
  fixed text per card, so each version hits after its first encode.
- Randomized noise mods (key shuffle, random key mask) give nearly
  unique text every draw. Those texts always miss, and under an
  always-admit policy they fill the cache with entries never read again:
  correct, but wasteful. The fix is an efficiency feature, not a
  correctness one (v1.2): mods declare whether they are deterministic,
  and a card touched by a random mod is encoded without being admitted.
- **The first runs need a config change.** Default mods apply to every
  dojo, contrastive ones included (`src/dojos/augmentation_defaults.py`),
  and r2 configs had no `mods:` section, so they were on. The first
  cached contrastive runs therefore set mods off (`mods:` per dojo).
  Those defaults mask each game's deck-defining field (faction, set,
  color) so same-deck positives cannot be solved from that field alone.
  Turning them off may make contrastive too easy; experiment F (K fixed
  augmentations per card) would keep them and stay cacheable.
- Heavy noise mods and the cache work against each other: the training
  README should say so.

## Relevant code today

**Model** (`src/encoder_model/`):
- `card_embedding_pipeline.py`: `CardEmbeddingPipeline.embed()` is the
  single path every model uses to embed cards. It chains four injected
  stages: text encoder, optional enrichment, pooling, embedding head.
  Today it runs all four on every card, every call.
- `text_encoder.py`: `PretrainedTextEncoder` (frozen by default, eval
  mode, `no_grad`, length-bucketed batches) and `StaticEmbeddingTextEncoder`
  (trainable). Output is a `TokenEncoding` (hidden states plus attention
  mask).
- `enrichment/token_whitening.py`: `TokenWhitening`, fitted
  mean/projection buffers with no trainable parameters, computed in fp32.
- `pooling/`: `MeanPooling` (no parameters), `AttentionPooling` and
  `TransformerPooling` (trainable).
- `embedding_head.py`: linear and residual-MLP heads (trainable).
- `single_card_model.py`, `multi_card_model.py`: both call
  `card_pipeline.embed()`; the multi-card model adds group attention on
  top.
- `card_serialization.py`: `serialize_card_to_json`, the text the encoder
  reads (deterministic for a given card value).

**Building the model from config:**
- `src/training/model_spec.py`: `ModelSpec` and the pooling/enrichment
  specs.
- `src/training/config_parsing/model_parsing.py`: parses `model:` into
  `ModelSpec`. The cache config and the cut-point validation go here.
- `scripts/run_training.py`: `build_model()` always builds a frozen
  `PretrainedTextEncoder`.

**Training:**
- `src/training/trainer.py`: calls `model(batch.inputs)`, and calls
  `next_batch` synchronously (about line 402). Phases toggle only the
  parameters that were trainable at construction, so the LM never thaws
  mid-run.

**Mods** (`src/dojos/mods/`, `src/dojos/augmentation_defaults.py`):
- Card mods return `dataclasses.replace(card, raw_content=...)`: same
  uuid, new text.
- Every game's default mods are a key shuffle, weighted field masks and a
  random key mask, applied to contrastive and metric dojos alike. The key
  shuffle and random mask make nearly every draw unique text.
- Required task masks (`MaskTargetKeyMod(train_only=False)`) are
  deterministic per card.

**Existing prototype** (experiment scripts only, not in `src/`):
- `experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_held_out.py`,
  `FrozenFeatureCache`: an in-memory dict from uuid to pooled vector,
  monkey-patched onto `embed()`.
- It made fresh-head training fast (3,000 steps in about 4-9 min, against
  about 1.4 s per step uncached) and matched the real model (cosine
  1.00000).
- Because it is keyed by uuid, it served unmodded features to modded
  training cards.

**Open TODOs this would close:**
- `src/encoder_model/TODO.md`: "Cache frozen token states".
- `src/training/README.md`: the "Not built" list.
- `src/evaluation/TODO.md`: "Shared frozen features".
- `experiments/TODO.md`: experiments D and F.

## Open questions for the implementation hand-off

- Config shape: where `cache:` sits in the run config, and its defaults.
- Where the store lives on the model object (inside the pipeline, or a
  wrapper around its frozen stages), so dozens of model variants reuse it.
- How a mod declares itself deterministic, and how that reaches the
  model with the card.
- Whether the warm tool and the prefetch worker share one code path.
