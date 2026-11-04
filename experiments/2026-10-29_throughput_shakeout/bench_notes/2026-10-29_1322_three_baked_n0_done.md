# three_baked_n0 done

- **Run:** `three_baked_n0`, attempt 1, commit 666ce89; 13:05:06 - 13:22:27
  (17.4 min), exit 0.
- **Startup:** 3.6 min to the first step. Binder bakes loaded from the save
  directory (no re-bake); dojos built and preflighted in about 5 s. Cache
  warm: 40,670 texts cold-encoded in 3.2 min (MTG 34,666 in 2 min 55 s,
  about 198 cards/s; FaB 5,188 in 16 s; Dominion 818 in 2 s).
- **Training:** 4 rounds of 2,000 steps; round wall times (TEST
  included) 204, 206, 207, 206 s, about 103 ms per step. Feature cache:
  4,139,518 texts looked up, 40,670 encoded (all during warming), 0
  uncacheable, 0 evictions; 40,670 rows in every tier.
- **Throughput (rounds 1-3, TEST included):** about 1,035,000 texts per
  206 s round, about **5,000 cards/s** (about 2,500 decks/s; single-card
  style, 2 cards per deck).
- **Memory:** torch VRAM peak 4.46 GiB allocated, 6.42 GiB reserved;
  process dedicated VRAM 6.82 GB (adapter 7.76 GB). Python working set
  6.0-6.5 GB; free system RAM 11.5-12.3 GB during training.
- **Utilization (13:09-13:22, 46 samples):** python CPU median 1,503% (of
  1,600%), GPU compute counter median 5.9%. scale_profile's `mtg_n0`
  read 2.2% and 1,309% on the same counters: the compute-engine counter
  does not track ROCm kernels well, so it is not evidence of an idle GPU.
- **Losses (not the point):** TEST normalized loss MTG 0.81 → 0.79x, FaB
  0.61 → 0.58x, Dominion 0.99 → 1.06x baseline.
- **Issues:** none. Free RAM dipped to 11.5 GB, under the queue's 12 GB
  pre-flight threshold, during the run (the check is only before each run).
