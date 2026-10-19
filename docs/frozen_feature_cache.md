# Embedding cache for faster training (plan, early draft)

Status: goals and known problems only. The design is to be discussed;
nothing is built.

## Goal

Increase training speed with an embedding cache, so the frozen text
encoder (ModernBERT) stops re-encoding the same card text every step.

- **Flexible, with two cache points for now:** after BERT (raw token
  states) and after pooling (one vector per card).
- **Two access patterns to serve:**
  1. Contrastive dojos: a few extremely common cards (staples in most
     decks) and a long tail of unusual ones. Recent hits stay relevant
     from batch to batch.
  2. Truly random, uniform access.
- **Card mods are supported**, on the understanding that heavy mod use
  makes the cache miss often.
- **Embeddings easy to move to the GPU**, or stored there as the cache
  itself.

## Known problems and likely direction

- **Where the cache lives:** likely in the model itself. The cut point
  depends on which stages are frozen, so the model is the component that
  knows it.
- **Model architecture:** the model module needs reworking so that text
  embedding, whitening, pooling and embedding head are each interchangeable
  strategies, with a configurable cache inserted at a cut point. Loosely
  understood; needs more thought and planning.
- **Cache key: the card's text, not its uuid.**
  - Mods keep a card's `nocab_uuid` but change its `raw_content`, so a
    uuid key would serve stale, unmodded features.
  - The key is a hash of the serialized text the encoder reads.
  - Entries are also tied to a fingerprint of the frozen stages before the
    cut (checkpoint, tokenizer length, whitening stats), so changing any of
    them invalidates the cache.
- **Cache technology:** undecided. Possibly experiment with a few options
  later.

## Plan

**Step 1: define the cache store interface.** A small protocol that the
model talks to, so the eviction/admission policy and the storage
technology can each be swapped without touching the model:

- `get_many(keys) -> (features, miss_mask)`: one batched lookup.
- `put_many(keys, features)`: one batched insert; the store's policy
  decides what is admitted or evicted.

Candidate policies (always admit, admit on second hit, freeze when
full) and backends (plain dict, tensor slab on GPU or pinned CPU,
memmap/LMDB on disk) plug in behind it. Later steps: to be discussed.

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
  `ModelSpec`.
- `scripts/run_training.py`: `build_model()` always builds a frozen
  `PretrainedTextEncoder`.

**Training:**
- `src/training/trainer.py`: calls `model(batch.inputs)`. Phases toggle
  only the parameters that were trainable at construction, so the LM
  never thaws mid-run.

**Mods** (`src/dojos/mods/`, `src/dojos/augmentation_defaults.py`):
- Card mods return `dataclasses.replace(card, raw_content=...)`: same
  uuid, new text.
- Every game's default mods are a key shuffle, weighted field masks and a
  random key mask. The key shuffle and random mask make nearly every draw
  unique text.
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
- `experiments/TODO.md`: experiment D.
