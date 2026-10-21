# Frozen feature cache (plan)

Status: v1 (pooled cut), v1.1 (VRAM, RAM and disk tiers; prefetch) and
v1.2 (scheduled prefetch with pinning; more slot policies; the
do-not-cache stamp) are built and documented in the READMEs cited
below. v1.3 and v1.4 are direction.

## Goal

Make training faster by caching what the model's frozen stages output
(ModernBERT, token whitening, mean pooling), so the same card text is
not encoded again every step. Measured on the RX 6800 with 256 Gwent
cards: a forward pass takes 807 ms live and 3.2 ms cached, and the two
agree to cosine 1.000000. Warming runs at about 200 cards/s.

The corpus is expected to grow far past VRAM and RAM (more games,
languages, fan-made cards), so every design choice here assumes the
working set does not fit in the top tier, even though today's ~66k cards
do.

## Current state this builds on

Cited from `src/encoder_model/feature_cache/README.md`,
`src/training/README.md` and `src/training/diet/README.md`.

- `PooledFeatureCache(store, fingerprint, durable_misses)`:
  `features(texts, pool_live, device, durable=None) -> float32 (N,
  width)`, `reserve(...) -> FeatureReservation`, `release(reservation)`,
  `stats()`, `close()`, `fingerprint`.
- `FeatureStore[F]` (Protocol): `lookup`, `put_many`, `pin`, `unpin`,
  `tier_stats`, `storage_dtype`, `close`. Implementations:
  `TensorFeatureSlab` (VRAM or pinned RAM), `SqliteFeatureStore` (disk,
  durable puts only, never evicts, ignores pins), `TieredFeatureStore`
  (inclusive chain, write-through, promotion).
- `SlotPolicy` (Strategy): `record_hits(keys)`, `admit(offered,
  free_rows, pinned) -> AdmissionDecision(admitted, evicted,
  refused_for_pins)`; it never evicts a pinned key.
  `FillUntilFull`, `FirstInFirstOut`, `LeastRecentlyUsed` (default),
  `LeastFrequentlyUsed`, and the `SecondOfferAdmission` decorator.
- `GenericCard.cacheable` (False: the do-not-cache stamp a random mod
  sets); `features`/`reserve` take parallel `cacheable` flags and never
  store an uncacheable text.
- The Trainer draws each round's `RoundSchedule` up front; a `BatchFeed`
  builds it in order (`begin_round`, `next_batch`, `reschedule`,
  `end_round`, `close`), and at `prefetch_depth >= 1` a worker reserves
  each batch's cards through a `BatchWarmer` up to depth steps ahead.

## Measured before v1.1 (2026-10-20)

A 100-step Gwent contrastive smoke run on the RX 6800, fp16, mods off,
32 cards per batch:

| | live | cached (warmed) |
|---|---|---|
| whole step | 295 ms | 144 ms |
| `next_batch` | 68 ms | 32 ms |
| forward + loss | 209 ms | 92 ms |
| backward + optimizer | 18 ms | 20 ms |

With the cache, the forward pass is nearly all
`contrastive_loss.anchor_loss`: a per-anchor Python loop of 32 calls at
about 2.7 ms each. `next_batch` time is mostly
`pair_constructor.known_card_uuids`. Both are dojo-side costs that the
cache cannot remove. Prefetch overlaps `next_batch`. The loss loop is a
separate vectorization job for the contrastive dojo.

## Known limits

- **The RAM tier never sees VRAM's hits.** Under `lru` it ranks the
  hottest keys as the oldest and evicts them first; `fifo` or
  `fill_until_full` avoid that. A policy sharing recency across tiers
  is still open.
- **The fingerprint has no library versions.** A `torch` or
  `transformers` upgrade that changes the encoder's numbers would reuse
  stale disk rows. Adding the versions would invalidate the disk on
  every ROCm nightly. For now, delete the row set (or the file) after
  an upgrade that matters.
- **Prefetch overlap is bounded.** The slab's two-way stream sync makes
  the training thread's default stream wait for a side-stream encode at
  every `put_many`.
- The disk file's size (~3x payload) and the CPU -> GPU -> CPU round
  trip of a disk hit promoted into RAM are tracked in
  `src/encoder_model/TODO.md`.

## v1.3: more cut points

After BERT (token states) and after token whitening. These need ragged
storage (one flat buffer plus offsets per card) and make the RAM and
disk tiers essential: roughly 400 KB per card at the token level,
against 1.5 KB pooled at fp16. They allow training attention pooling on
cached token states, or adding token features to the pooled vector.
This changes the pipeline seam: a token-level cut sits between encoding
and pooling, so `embed()` and the type of its cache parameter change. A
cut after the embedding head, for a fully frozen encoder in extrinsic
evaluation, closes `src/evaluation/TODO.md` "Shared frozen features".
Pinning matters more here: far fewer token-level rows fit in
VRAM.

## v1.4: experiments

- D: train from cached features and match an uncached run
  (`experiments/TODO.md`).
- F: a fixed pool of K augmentations per card, which makes noise mods
  cacheable.
- Locality-aware sampling: draw N decks (or one game's cards) per
  episode, and train many batches from that working set, sized to the
  cache. This gives up some diet mixing for throughput, perhaps with a
  lower learning rate. The prior art is GNS and ClusterGCN from GNN
  training. The round schedule is the natural place to express
  it.
