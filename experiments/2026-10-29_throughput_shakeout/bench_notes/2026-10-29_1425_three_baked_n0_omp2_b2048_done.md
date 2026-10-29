# three_baked_n0_omp2_b2048 done

- **Run:** `three_baked_n0_omp2_b2048` (`OMP_NUM_THREADS=2`,
  `max_batch_cost` 2048, 2 rounds of 500), attempt 1, commit 666ce89;
  14:19:04 - 14:25:21 (6.3 min), exit 0.
- **Training:** rounds 172 and 172 s, 345 ms per step: about **5,940
  cards/s**, against 6,035 in `three_baked_n0_b2048` (default threads)
  and 6,470 in `three_baked_n0_omp2` (512).
- **Utilization:** python CPU median 375%; GPU compute counter 3.9%.
- **Memory:** torch VRAM peak 3.90 GiB; working set 6.4 GB.
- **Issues:** none.
