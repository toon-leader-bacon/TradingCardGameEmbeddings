# Environment sizing

Fixed numbers for sizing training runs and the feature cache
(`../encoder_model/feature_cache/`) on this machine. Measured 2026-10-21
in the ROCm venv (`torch 2.10.0+rocm7.13.0a20260421`), commit 951302a,
unless marked otherwise. Everything here assumes the **pooled cut**: one
768-wide vector per card, so every cached card is the same size.

## Hardware

| | |
|---|---|
| GPU | AMD Radeon RX 6800 (`gfx1030`), 15.98 GiB VRAM; 15.85 GiB free at idle; a 14 GiB allocation succeeds |
| CPU | Ryzen 7 3700X, 8 cores / 16 threads |
| RAM | 32 GB; about 20 GB free with the desktop's usual apps open |
| Pinned host RAM | 12 GiB pinned without trouble (larger not tried) |
| Host -> GPU copy | 24.7 GiB/s from pinned memory |

Windows spills VRAM into system RAM instead of raising out-of-memory, so
a run over budget gets slow, not killed (`Notes.md`).

## Corpus

65,634 cards in `data/final/cards` over 7 games (MTG 34,666, Pokemon
16,928, Hearthstone 6,195, Flesh and Blood 5,188, Gwent 1,261, Dominion
818, Slay the Spire 2 578); 65,626 distinct serialized texts.

## What the models take in VRAM

| | VRAM |
|---|---|
| Frozen ModernBERT-base + head, weights (any model below) | ~590 MiB |
| Trainable parameters and AdamW state: linear head (0.2M), residual MLP (1.3M), multi-card with 2 attention layers (2.9M) | at most ~22 MiB |
| Training step, single card, every card a cache hit (32 to 512 cards) | 1.19 GiB peak allocated, flat across batch sizes |
| The same with the default mods of the time on (most cards encoded live) | 1.48 GiB peak |
| One cold forward + backward over 512 cards, above the weights | MTG 1.16 GiB, Pokemon 0.89, FaB 0.80, Gwent 0.49 |
| Real runs in 2026-10-11_model_by_diet_r2 (multi-card, whitening, many dojos) | 3 to 5 GiB |

The frozen encoder is nearly all of it; the trainable model is small
enough to ignore. What a step adds depends on how many of its cards miss
the cache, since a miss runs ModernBERT.

## Pooled row size

| Storage dtype | Bytes per row (768 wide) | Rows per GiB | Whole corpus |
|---|---|---|---|
| fp16 / bf16 | 1,536 B | 699,050 | 96 MiB |
| fp32 | 3,072 B | 349,525 | 192 MiB |

The SQLite disk tier takes about 3x the payload: 293.5 MiB on disk for
65,626 fp16 rows, about 4.6 KiB per row.

## Suggested cache budgets

Leaves about 5 GiB of VRAM for the model, cold-encode spikes, the
allocator's slack and the display, and leaves RAM for the dojos' deck
boxes (about 2 to 2.6 GB in past runs).

| Tier | Budget | Rows at fp16 | Rows at fp32 | Times today's corpus (fp16) |
|---|---|---|---|---|
| VRAM | 10 GiB | ~7.0M | ~3.5M | ~107x |
| Pinned RAM | 8 GiB | ~5.6M | ~2.8M | ~85x |
| Disk (SQLite) | 20 GiB of payload, about a 60 GB file | ~14M | ~7M | ~213x |
| Default VRAM tier, for comparison | 512 MiB | 349,525 | 174,762 | 5.3x |

At the pooled cut the cache never limits VRAM: today's whole corpus is
under 100 MiB at fp16. A token-level cut (plan v1.3, about 400 KB per
card) would fit only about 26k cards in the same 10 GiB, and is where
these budgets start to bind.

## Speeds

| | Rate |
|---|---|
| Encoding (cold, fp16, warm() chunks of 256) | MTG 200, Pokemon 214, Gwent 222, FaB 337, StS2 328, Dominion 389, Hearthstone 389 cards/s |
| Warming the whole corpus | about 5 min |
| VRAM tier hit | ~140-200k rows/s |
| Pinned RAM tier hit | ~200-215k rows/s |
| Disk tier hit | ~10-15k rows/s |
| Refill the VRAM and RAM tiers from disk | 10 s for the corpus |

A VRAM hit is no faster than a RAM hit: at 1.5 KB a row, lookups are
bound by per-text work on the CPU (hashing, folding duplicates, the slot
policy), not by bandwidth. A disk hit is still about 50x faster than
encoding.

## Batch size (single-card contrastive, warm cache)

Gwent, FaB, StS2 and Dominion contrastive dojos (plus Pokemon at 32 and
128 cards), residual-MLP head, fp16, prefetch depth 2, mods off, every
card a cache hit, vectorized loss (2026-10-21). "Forward + loss" is the
model's forward pass and the loss; "waiting for the batch" is the
step's wait in `next_batch`, i.e. batch building that prefetch did not
hide.

| Cards per batch | Step | Cards/s | Forward + loss | Waiting for the batch |
|---|---|---|---|---|
| 32 | 11.7 ms | ~2,700 | 3.6 ms | 0.0 ms |
| 128 | 49 ms | ~2,600 | 6.7 ms | 32 ms |
| 512 | 182 ms | ~2,800 | 10.4 ms | 162 ms |
| 2,048 | 617 ms | ~3,300 | 31.6 ms | 556 ms |

Before the loss was vectorized, the same runs took 118 ms (32 cards),
1.33 s (128) and 19.5 s (512) per step: the per-anchor Python loop was
nearly the whole step. Now the GPU side is a few milliseconds and the
step is bound by building batches on the CPU, about 0.27 ms per card
(dealing decks, sampling, the deep copy per card, the cache lookup).
Bigger batches no longer cost more per card, so batch size is free to
follow what training wants. Caveats:

- These runs lifted a 16-deck cap on contrastive batches in a scratch
  script. The cap is gone since (2026-10-21): `max_batch_cost` alone sets
  the decks per batch.
- In these runs, a small deck box could not fill a batch (Pokemon, 159
  TRAIN decks, failed at 512 cards and up), and every dojo's TEST pass
  yielded nothing once a batch needed more decks than
  `eval_examples_per_dojo` (64). Both are fixed: a split's last batch
  holds whatever decks are left, and a capped pass trims its last batch.
  A huge budget against a small deck box still means every batch is the
  whole box (Pokemon at 512 cards: one 159-deck batch per step).
- With the default mods of 2026-10-21 (field mask and key mask; key
  shuffle no longer a default), 40% of cards were encoded live at 512
  cards: 642 ms per step, nearly all of it the live ModernBERT forward
  pass (peak 1.6 GiB allocated). Before the shuffle was dropped, at 32
  cards, about 78% were live: 199 ms per step against 119 ms with mods
  off.
