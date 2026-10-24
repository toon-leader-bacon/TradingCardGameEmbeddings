# Training at scale: where a multi-card mixed-diet run spends its time, with and without bounded mods

**Status:** proposed
**Dates:** 2026-10-24 -
**Parent:** [2026-10-21_cache_shakeout](../2026-10-21_cache_shakeout/NOTEBOOK.md) · **Branch:** `experiment/2026-10-24_scale_profile`

> At scale, a multi-card model on a mixed contrastive and metric diet
> trains at some rate set by its slowest parts; we measure those parts,
> with augmentation off and with it bounded to 4 cacheable variants, and
> project how long one contrastive pass over every card of every MTG deck
> would take.

## Aim

**Problem (priority: the fix, i.e. industry R&D; measurement, no bar).**
The shake-out (parent) measured a single-card model for 2 × 1,000 steps.
This experiment measures the current code over hours, on the model and
diet we expect to train (multi-card, r2's mixed diet), and answers:

1. **Bottlenecks:** where a step's time goes (waiting for the batch,
   forward + loss, the rest; and within those, mods, serialization,
   cache lookups and live encoding), and which dojos are slowest to
   build and to train.
2. **Memory pressure:** peak and steady VRAM (torch allocated and
   reserved, and the process's dedicated VRAM), process and free system
   RAM, and the feature cache's fill per tier.
3. **Cache hit rate:** the share of training texts not encoded, over
   time, and per tier; and whether evictions happen.
4. **GPU utilization:** the Windows compute-engine counter over the run.
5. **The at-scale estimate:** how long one contrastive pass that uses
   every known card of every TRAIN MTG deck as an anchor would take,
   without mods and with mods at 4 variants.

The answer is a ranked list of costs with a size for each, plus the
projection, to decide the next throughput work.

**Hypotheses** (predictions, second priority):

- **H1 (bounded mods are cacheable).** With `variants: 4`, after the
  first pass over the modded texts, training is about as fast as with
  mods off. *Refuted if*, over the run's last hour, the share of
  training texts avoided by the cache is under 95%, or the arm's
  cards per second (training steps only, same dojos) is under 0.85× the
  mods-off arm's.
- **H2 (CPU-bound).** With mods off and a warm cache, the training
  thread spends most of each step waiting for its batch, and the GPU is
  mostly idle. *Refuted if* the median share of step time spent waiting
  is under 30%, or the median GPU compute utilization is over 60%.
- **H3 (a full MTG pass takes days).** At the measured MTG-contrastive
  rate, one pass of every known card of every TRAIN MTG deck as an
  anchor takes more than 24 h, with or without mods. *Refuted if*
  either projection is under 24 h.

## Background

- **Parent (2026-10-21_cache_shakeout):** single-card model, mixed diet,
  2 × 1,000 steps. With the then-default random mods, live encoding was
  82% of a cached step (1.22 s per step); mods off, 0.255 s per step,
  the GPU busy 21%, batch wait 13%. A bigger batch (512 vs 256 cards)
  gave only 9% more cards/s: the step is per-card bound. CardBinder's
  deep copies cost about 22 min of startup.
- **Since then:**
  - PR #12 merged the parent's speed work into main.
  - PR #13 removed CardBinder's deep copies, after a test showed no mod
    changes a binder card.
  - PR #14 (open, branched from here) added `variants:` to card-field
    mods: N > 0 gives each card at most N modified texts. They are
    picked from 100 fixed seeds per (variant, card, mod class), so they
    are cacheable.
- **`src/training/ENV_SIZING.md`** (2026-10-22): 512-card budget,
  mods off, warm cache: about 2,100 cards/s, 78% of the step waiting
  for the batch. Encoding runs at about 200 MTG cards/s. The whole
  corpus (65,626 texts) is 96 MiB at fp16.
- **MTG deck box:** 6,911,380 decks in `data/final/decks/mtg.db`
  (17lands limited decks); counts of cards per deck are in Method.

## Method

_(filled in below as the design is fixed; see the Log for decisions)_

## Recipe

## Log

### 2026-10-24 · Design brief from the human

What: the user handed over the workstation for a multi-hour,
industry-style profiling experiment and gave the PI experimental
authority while away. Asked for:
- two full runs: N=0 (augmentation mods off, task mods kept) and N=4
  variants, with mods firing on about 10% of cards;
- a smart eviction policy, good cache sizes and good batch sizes;
- logging and timing added "throughout";
- a new experiment branch.

Answers to the PI's questions before they left:
- branch off PR #14;
- timing code goes permanently into `src`;
- about 4-5 h per arm, about 10 h in all;
- add a `probability` to every augmentation mod.

## Results

## Conclusions

## Open questions
