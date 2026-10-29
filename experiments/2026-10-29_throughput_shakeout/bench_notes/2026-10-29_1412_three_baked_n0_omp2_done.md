# three_baked_n0_omp2 done

- **Run:** `three_baked_n0_omp2` (`three_baked_n0` with `OMP_NUM_THREADS=2`
  from `analysis/env/three_baked_n0_omp2.env`, 2 rounds), attempt 1, commit
  666ce89; 14:06:42 - 14:12:32 (5.8 min), exit 0. The env file is recorded
  in `provenance/three_baked_n0_omp2/provenance.txt`.
- **Training:** round wall times 158.6 and 158.3 s for 2,000 steps (TEST
  included), about 79 ms per step, against 206 s in `three_baked_n0`.
- **Throughput:** about **6,470 cards/s**, 29% above `three_baked_n0`.
- **Utilization:** python CPU median 347% (against 1,503%); GPU compute
  counter 6.5%. Python working set 6.0-6.2 GB.
- **Issues:** none.
