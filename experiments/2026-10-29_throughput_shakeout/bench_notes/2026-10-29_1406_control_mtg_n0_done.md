# control_mtg_n0 done

- **Run:** `control_mtg_n0` (scale_profile's `mtg_n0` config, own disk
  tier), attempt 1, commit 666ce89; 13:53:26 - 14:06:41 (13.3 min), exit 0.
- **Startup:** 3.3 min to the first step, 2 min 56 s of it cold-encoding
  34,666 MTG cards (about 197 cards/s).
- **Training:** 1 round of 6,000 steps in 593 s (TEST included), about
  99 ms per step; steady 10.06 steps/s from t = 250 s to the end.
- **Throughput:** 3,072,000 TRAIN cards in 593 s, about **5,180 cards/s**
  (about 2,590 decks/s). scale_profile's `mtg_n0`: 0.238 s mean step, about
  2,150 cards/s on step time alone; this run is 2.4× faster per step.
- **Cache:** 34,666 encoded (all in warming), 0 uncacheable, 0 evictions.
- **Memory:** torch VRAM peak 4.46 GiB allocated, 6.44 GiB reserved;
  process dedicated VRAM 6.85 GB. Python working set 6.1 GB; free RAM
  12.3-12.4 GB.
- **Utilization:** python CPU median 1,504%, GPU compute counter 4.7%.
- **Losses (not the point):** TEST normalized MTG 0.87x baseline.
- **Issues:** none.
