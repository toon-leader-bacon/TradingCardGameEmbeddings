# batch_build_bench done

- **What:** `analysis/batch_build_bench.py --batches 300` with
  `OMP_NUM_THREADS=1`, CPU only, after the queue; configs
  `three_baked_n0` and `three_baked_n4`. Outputs
  `results/batch_build_three_baked_n0.csv` and `..._n4.csv`; logs
  `data/runs/2026-10-29_throughput_shakeout/logs/bench_*.log`.
- **n0 (no dojo mods), ms per 512-card batch:** MTG 27.8, FaB 31.2,
  Dominion 22.7 (0.044-0.061 ms per card). Dojos built in 5.0 s.
- **n4 (default mods, 4 variants):** MTG 46.5, FaB 45.9, Dominion 33.1
  (0.065-0.091 ms per card).
- **Issues:** none. One FaB deck skipped (1 known card, needs 2), as in
  training.
