# three_baked_n4 done

- **Run:** `three_baked_n4`, attempt 1, commit 666ce89; 13:22:28 - 13:41:34
  (19.1 min), exit 0.
- **Startup:** about 30 s to the first step. Bakes loaded from the save
  directory; the 40,670 baked texts came from the shared disk tier in 6 s
  (no encoding), written there by `three_baked_n0`.
- **Training:** round wall times (TEST included) 361, 260, 249, 242 s for
  2,000 steps each, against 206 s per round in `three_baked_n0`.
  Rounds 2-4 average 250 s, 21% slower; round 4 alone is 17% slower.
- **Cache:** 62,760 texts encoded live during training (the dojo mods'
  bounded variants), 0 uncacheable, 0 evictions; rows grew from 40,670 to
  103,430. Live encodes per 200 s sample: about 27,800, 13,100, 8,500,
  6,100, 4,900, then about 2,400 in the last 110 s. Still falling at the
  end: the steady state was not reached in 8,000 steps.
- **Memory:** torch VRAM peak 4.53 GiB allocated, 5.83 GiB reserved;
  process dedicated VRAM 6.2 GB. Python working set 5.8-6.6 GB; free RAM
  11.4-12.7 GB.
- **Utilization (13:23:30-13:41:20):** python CPU median 1,491%, GPU
  compute counter median 25% (against 5.9% in `three_baked_n0`). The extra
  counter reading lines up with the live ModernBERT passes.
- **Losses (not the point):** TEST normalized MTG 0.86 → 0.86x, FaB
  0.64 → 0.60x, Dominion 0.97 → 0.99x. Gradient clipping rose on MTG in
  rounds 3-4 (234 and 193 of about 660 MTG steps; `three_baked_n0`: 0 after
  round 1).
- **Issues:** none.
