# feature_cache

This package caches what a card pipeline's frozen stages (text encoder,
optional token whitening, mean pooling) output for each card text. The
frozen stages then run once per distinct text instead of on every
training step. The embedding head and everything after it still run on
every call.

The cache is optional: `CardEmbeddingPipeline(..., feature_cache=None)`
runs every stage on every call. Rows can live in three tiers, VRAM,
pinned RAM and an SQLite file on disk, and the disk tier lets runs and
experiments share their encodes.

The key is the serialized text, never the card's `nocab_uuid`. A mod
keeps the uuid and changes the text, so a modded card is just another
key.

## Files

- `card_text_key.py`: `CardTextKey.of_text(text)`, a 16-byte BLAKE2b
  digest of the text `serialize_card_to_json` produces.
- `feature_storage_dtype.py`: `FeatureStorageDtype` (`fp32` / `fp16` /
  `bf16`), the dtype rows are stored in. It has `.torch_dtype` and
  `.rows_in(capacity_mib, width)`, which turns a MiB budget into rows
  for every tier. It sets storage rounding only, not the autocast
  compute precision.
- `frozen_stage_fingerprint.py`: `FrozenStageFingerprint`, everything
  that fixes a cached row's numbers for a given text:
  - the text encoder's `TextEncoderSource` (checkpoint, Hugging Face
    commit hash, `max_length`, `model_kwargs`);
  - the whitening stats file's sha256;
  - the pooling label;
  - the compute precision;
  - the width;
  - the storage dtype.

  `digest()` names a row set on disk.
- `feature_store.py`: `FeatureStore[F]` (Protocol): batched,
  thread-safe `lookup(keys, device) -> StoreLookup` and
  `put_many(keys, features, durable) -> admitted`, `pin(keys)` /
  `unpin(keys)`, plus `tier_stats()`, `storage_dtype` and `close()`.
  Also `TierStats` (hits, fill, `pinned_rows`, `refused_for_pins`), and
  `require_offer_shape`, the shape check every `put_many` makes.
- `slot_policy.py`: `SlotPolicy` (Strategy), which decides which offered
  keys a memory tier admits and which held keys it evicts, given the
  tier's pinned keys: `admit(offered, free_rows, pinned) ->
  AdmissionDecision`. None evicts a pinned key.
  - `FillUntilFull` admits while rows are free and never evicts.
  - `FirstInFirstOut`, `LeastRecentlyUsed` (the default) and
    `LeastFrequentlyUsed` (fewest hits, oldest first among ties) share
    a Template Method base that admits the first `capacity_rows` offered
    keys and evicts each one's victims. `LeastRecentlyUsed` is
    `FirstInFirstOut` plus "a hit moves the key to the newest end".
  - `SecondOfferAdmission(inner, remembered_keys)` is a Decorator over
    any of them: a key reaches `inner` only on its second offer. First
    offers are remembered in a bounded FIFO set; a repeat that `inner`
    refuses stays remembered. Under prefetch a card's reservation is its
    first offer, so its step encodes it again.
- `tensor_feature_slab.py`: `TensorFeatureSlab(name, capacity_rows,
  storage_dtype, policy, placement)`, a memory tier. Its rows sit in one
  preallocated tensor, with a CPU dict from key to row. `SlabPlacement`
  is `FOLLOW_FIRST_PUT` (the first put's device, VRAM in training) or
  `PINNED_HOST` (the RAM tier).
- `sqlite_feature_store.py`: `SqliteFeatureStore(path, fingerprint,
  capacity_rows)`, the disk tier. One file holds any number of row sets,
  one per fingerprint digest. It keeps only durable puts and admits
  until its row set is full.
- `tiered_feature_store.py`: `TieredFeatureStore(tiers)`, the chain,
  fastest tier first.
- `pooled_feature_cache.py`: `PooledFeatureCache(store, fingerprint,
  durable_misses)`, the component the pipeline calls, with
  `features`, `reserve` and `release`. Also `FeatureReservation` and
  `FeatureCacheStats`.

## How it works

`CardEmbeddingPipeline.embed(cards)` serializes the cards, then calls
`cache.features(texts, pool_live, device)`, where `pool_live` is the
pipeline's own encode -> enrich -> pool:

1. Check that the active autocast is the fingerprint's compute
   precision.
2. Fold duplicate texts, and key each distinct text once.
3. `store.lookup(keys, device)`: the chain asks each tier in turn about
   the keys the tiers above missed, and offers each lower-tier hit to
   the tiers above it (promotion).
4. Run `pool_live` once, under `no_grad`, on the misses only. Round the
   result to the storage dtype and offer it to every tier
   (write-through), so each tier's policy decides what to keep.
5. Return float32 rows on `device`, in the texts' order.

A miss is rounded exactly as a stored row is, so a card's features never
depend on whether it hit or which tier answered. The head always
receives float32, as on the live path.

```mermaid
flowchart LR
    E[embed] --> C[PooledFeatureCache]
    C -->|lookup / put_many| T[TieredFeatureStore]
    T --> V[VRAM slab]
    T --> R[pinned RAM slab]
    T --> D[(SQLite, per fingerprint)]
    C -->|misses only| P[text encoder -> whitening -> mean pooling]
```

### Reservations

A prefetch worker reserves a batch before its step runs:
`model.reserve(cards, precision) -> FeatureReservation` (through the
pipeline to `cache.reserve(texts, ...)`).

1. Check the precision, then pin the batch's distinct keys in every
   memory tier.
2. Look up and encode as `features` does, without returning rows.
3. If that raises, unpin and re-raise.

`release(reservation)` unpins once the step has looked its features up.

- **Pins are reference counts.** One key can sit in several queued
  batches. A key can be pinned before it is held.
- **No pinned row is evicted.** A slab passes its pinned keys to its
  policy. The slab raises `RuntimeError` if a decision evicts one.
  When pinned rows fill a slab, the rest of an offer passes through
  unstored and counts in `refused_for_pins`.
- **The disk tier ignores pins**, since it never evicts.
- **Reservations are compared by identity.** The cache keeps the ones
  not yet released. Releasing one twice, or another cache's, raises
  `ValueError`. An empty reservation (no texts, or a pipeline without a
  cache) releases freely.
- **Totals.** A reservation's encodes count in `encoded`, but its texts
  are not counted in `texts`. The step's own `features` call counts
  them, so `avoided_fraction` still describes training lookups.

Releasing right after the step's lookup is safe because of the slab's
stream rule below. The lookup gathers into its own tensor on the
default stream, and any later overwrite is queued after it.

Invariants:

- **Inclusive tiers.** Every put is offered to every tier, so an eviction
  from an upper tier needs no copy downward: the row is usually already
  below.
- **Durable means disk.** The disk tier stores only puts made with
  `durable=True`. `warm()` always puts durably. A training miss is
  durable only with `disk.write: all`, since the disk never evicts.
- **Stamped texts are never stored.** `features` and `reserve` take
  `cacheable` flags parallel to the texts (the pipeline passes
  `GenericCard.cacheable`; a random mod's cards are stamped False). A
  distinct text is cacheable if any of its positions is. An uncacheable
  text is still looked up, since a random mask can land on a stored
  text, and on a miss it is encoded but never offered to any tier, and
  counted in `FeatureCacheStats.uncacheable`. `reserve` skips it
  entirely, so its encode runs in the step.
- **Frozen stages only, always in eval mode.** The pipeline refuses a
  cache when a stage before the head has a parameter with
  `requires_grad`. With a cache, `train()` keeps those stages in eval.
  `ModelSpec` refuses a pooled cut with attention or transformer
  pooling.
- **One compute precision.** `features()` raises `ValueError` when the
  active autocast is not the fingerprint's precision.
- **A pinned source for the disk.** `SqliteFeatureStore` refuses a
  fingerprint whose text encoder has no source or no revision (a local
  path), because its weights could change under the same name.
- **Not in the checkpoint.** The cache is a plain attribute of the
  pipeline, so `state_dict()` is the same with or without it.
- **Thread-safe.** Each store takes a lock around `lookup` and
  `put_many`. On CUDA, every slab gather and write runs on the device's
  default stream, synced with the caller's stream in both directions.
  Hits handed to another stream get `record_stream`. Pinned rows are
  written with blocking copies. No lock is held while encoding. Pins
  rely on this ordering, so a change to the slab's streams must keep
  it.
- **Lifecycle.** The SQLite connection opens on first use (WAL,
  `synchronous=NORMAL`, a 30 s busy timeout), so building a model creates
  no file. `close()` closes it, and a closed store refuses further use.

The disk file runs about 3x its row payload at pooled widths. Each row
carries its 64-character digest and its key, and a 4 KiB page holds only
about two 1.5 KB rows.

## How to run

```python
from src.training.feature_cache_assembly import assemble_feature_cache, build_fingerprint

fingerprint = build_fingerprint(encoder.source(), None, "mean", "fp16", 768,
                                spec.feature_cache.storage_dtype)
cache = assemble_feature_cache(spec.feature_cache, fingerprint)
pipeline = CardEmbeddingPipeline(encoder, None, MeanPooling(768), head, cache)
model = SingleCardModel(pipeline).to("cuda")
model.warm(cards, "fp16")            # optional
reservation = model.reserve(batch_cards, "fp16")   # pinned until released
model.release(reservation)
model.feature_cache_stats()          # FeatureCacheStats(texts, encoded, admitted, tiers, uncacheable)
model.close_feature_cache()
```

From a run config, see `model.feature_cache` and `warm_feature_cache` in
`../../training/README.md`. `scripts/warm_feature_cache.py <config>`
fills the disk tier without training.
