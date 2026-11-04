# Prefetch: a worked example

One cold-started round of the prefetching feed (README.md, "Prefetch").
The trainer draws the round's schedule up front, the worker builds and
*reserves* each batch ahead of the GPU, and reserved rows stay pinned
until the step that uses them is over. The numbers are a toy, so every
card fits on screen.

## Setup

| | |
|---|---|
| Dojos and schedule | A and B; round schedule `A B A B` |
| `prefetch_depth` | 2 steps ahead of the trainer |
| VRAM tier | 8 rows, LRU, honours pins |
| RAM tier | 16 rows (pinned host memory) |
| Disk tier | holds c1-c6 from an earlier `warm_feature_cache.py`; `write: warm_only` |

Batches, 4 cards each:

| Batch | Cards |
|---|---|
| A1 | c1 c2 c3 c7 |
| B1 | c1 c4 c8 c9 |
| A2 | c2 c5 c10 c11 |
| B2 | c8 c6 c12 c13 |

## Timeline

Four lanes on one clock (arbitrary time units). The worker builds a
batch on the CPU, then reserves it: it pins the batch's keys, reads
disk hits in one SQLite query, and encodes misses on its own GPU
stream. The training stream runs steps back to back from t=4; only the
first step waits, because the round starts cold. Markers S0-S5 match
the snapshots below.

```text
                 S0                          S1                   S2                   S3                   S4                   S5
Training thread      [~~~~~waits for A1~~~~~][======step A1======][======step B1======][======step A2======][======step B2======][eval=]
GPU train strm                               [====A1 fwd+bwd=====][====B1 fwd+bwd=====][====A2 fwd+bwd=====][====B2 fwd+bwd=====][eval=]
GPU worker strm                       [=c7==]              [c8,9=]              [c10+=]              [c12+=]
Worker thread        [build A1][=reserve A1=][build B1=][=rsv B1=][build A2=][=rsv A2=][build B2=][=rsv B2=][========idle: done========]
                 0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17
```

- **GPU worker stream:** the misses each reservation encodes: c7 for
  A1, c8 and c9 for B1, c10 and c11 for A2, c12 and c13 for B2.
- **Overlap:** each encode runs while the training stream runs the
  previous step. It is the same GPU, so the two share its compute.

## What each tier holds

How to read the snapshots:

- **Bold** means the card arrived at this marker; *(disk)*, *(encoded)*
  or *(hit)* says how.
- `pin X` means pinned by batch X.
- ~~c3~~ means evicted (its row is reused).
- `·` is an empty row.
- `r0`-`r7` are VRAM rows.

The disk holds c1-c6 throughout. It gains nothing during the round,
because `warm_only` keeps training-time encodes off it.

### S0 · t = 0 · Round starts cold

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 0/8 | · | · | · | · | · | · | · | · |

RAM 0/16: empty.

- The trainer draws the whole round at once (`A B A B`), then calls
  `begin_round`.
- It immediately asks for A1 and blocks, because nothing is built yet.
  This is the only wait in the round.

### S1 · t = 4 · A1 reserved, its step starts

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 4/8 | **c1** `pin A1` *(disk)* | **c2** `pin A1` *(disk)* | **c3** `pin A1` *(disk)* | **c7** `pin A1` *(encoded)* | · | · | · | · |

RAM 4/16: **c1** **c2** **c3** **c7**

- The worker pins c1, c2, c3 and c7 *before* looking them up, so no row
  of the batch can be evicted between arriving and being pinned.
- c1-c3 miss VRAM and RAM and hit disk in one batched SQLite read. They
  are promoted into RAM and VRAM with one write per tier, not one per
  card.
- c7 misses every tier. It is encoded on the worker's GPU stream and
  written to VRAM and RAM.
- The trainer takes A1. Its lookup is 4 VRAM hits, gathered into the
  step's *own* tensor. From here on, the slab's rows can change without
  touching the step.

### S2 · t = 7 · B1 reserved; A1 released

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 7/8 | **c1** `pin B1` *(hit)* | c2 | c3 | c7 | **c4** `pin B1` *(disk)* | **c8** `pin B1` *(encoded)* | **c9** `pin B1` *(encoded)* | · |

RAM 7/16: c1 c2 c3 c7 **c4** **c8** **c9**

- While A1's step ran (t=4-7), the worker reserved B1. c1 was already
  in VRAM, so it was a plain hit, and its pin count went to 2 (A1 and
  B1).
- c8 and c9 were encoded on the worker stream while A1's step ran on
  the training stream.
- At t=7 the trainer asks for B1. That call first releases A1's pins,
  so c2, c3 and c7 become evictable. c1 stays pinned by B1.

### S3 · t = 10 · A2 reserved; c3 and c7 evicted

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 8/8 | c1 | **c2** `pin A2` *(hit)* | **c10** `pin A2` *(encoded)* | **c5** `pin A2` *(disk)* | c4 | c8 | c9 | **c11** `pin A2` *(encoded)* |

Evicted from VRAM: ~~c3~~ ~~c7~~

RAM 10/16: c1 c2 c3 c7 c4 c8 c9 **c5** **c10** **c11**

- A2 needed 3 new rows and VRAM had 1 free. During the reservation B1
  was still pinned, because its step was running. So LRU could choose
  only between c3 and c7, the unpinned rows.
- c3 and c7 are evicted and their rows reused: rows 2 and 3 now hold
  c10 and c5. Both cards are still in RAM. The tiers are inclusive, so
  an eviction never copies anything down.
- At t=10 the trainer asks for A2, which releases B1. A2's step is again
  all VRAM hits.

### S4 · t = 13 · B2 reserved; c1, c4 and c9 evicted

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 8/8 | **c13** `pin B2` *(encoded)* | c2 | c10 | c5 | **c12** `pin B2` *(encoded)* | **c8** `pin B2` *(hit)* | **c6** `pin B2` *(disk)* | c11 |

Evicted from VRAM: ~~c1~~ ~~c4~~ ~~c9~~

RAM 13/16: c1 c2 c3 c7 c4 c8 c9 c5 c10 c11 **c6** **c12** **c13**

- VRAM was full. A2's four rows were pinned (its step was running) and
  c8 was pinned by B2, so the three victims had to come from c1, c4 and
  c9.
- If B2 had needed c3 again, it would have been a RAM hit promoted in
  one copy, with no disk read and no encode.
- At t=13 the trainer asks for B2, which releases A2. The schedule has
  no steps left, so the worker goes idle instead of building ahead.

### S5 · t = 16 · Round ends; evaluation

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|---|
| VRAM 8/8 | c13 | c2 | c10 | c5 | c12 | c8 | c6 | c11 |

RAM 13/16: c1 c2 c3 c7 c4 c8 c9 c5 c10 c11 c6 c12 c13

- `end_round` releases B2's pins. The worker has been idle since t=13,
  so evaluation waits for nothing, and no dojo is touched by
  two threads.
- Each dojo's stream advanced exactly 2 batches, the same as with
  `prefetch_depth: 0`.
- The next round starts with VRAM and RAM already full. Only its first
  step waits for a build.

In this toy, the rows LRU chose to evict were never pinned anyway, so
the pins did not change the outcome. They start to matter when the
rows the next few batches need would otherwise be the oldest: when the
batches in flight need more rows than VRAM can spare. That is the
large-corpus case.
