# 2026-11-11 19:50 · incident: main_R kept only its best checkpoint

**What happened:** main_R (attempt 1) finished cleanly: exit 0, 17:43:51 to 19:41:04 (117 min; the estimate was 205). But only `train_round0007/` and `latest/` are on disk. `DirectoryCheckpointer` saves a checkpoint only when a round improves on the best, and deletes the previous best (`src/training/recording/checkpointer.py`). `checkpoints.csv` lists rounds 0, 1, 2, 4, 5 and 7; rounds 3 and 6 did not improve, so they were never saved. The Method's speed verdict compares the two arms at every round checkpoint, so attempt 1 cannot answer it.

**Classed as:** small issue (1 of 5 so far). The gap is in the infrastructure the Recipe relied on, not in the design. The fix only adds files and changes nothing in training.

**Action:** fixed, and main_R is retried once with the fix. main_T runs after it with the same fix.

**Fix:** a new top-level config key `keep_round_encoders: true`. With it, each round's `save_latest` also writes `round<NNNN>/` (`encoder.pt`, `manifest.json`, `holdout.json`; no `state.pt`), and these are never pruned.
- Files: `src/training/run_config.py`, `src/training/recording/checkpointer.py`, `scripts/run_training.py`, and a test in `tests/training/test_checkpointer.py`.
- `analysis/make_configs.py` now sets the key on the main configs only. `main_R.yaml` and `main_T.yaml` were regenerated; the only difference is that one line.
- check.sh passed (5565 passed, 4 skipped).
- Runs before the fix: the six pilots and main_R attempt 1. Runs after it: main_R attempt 2 and main_T. The training code path is unchanged, so attempt 1 and attempt 2 should train identically up to GPU nondeterminism.

**Kept, not discarded:** attempt 1 is moved to `data/runs/2026-11-11_teacher_vs_real_pairs/main_R_attempt1/`, and its console log to `logs/main_R.attempt1.log`. It has the same config apart from the new key, and the same seed. So its final encoder against attempt 2's round 7 measures run-to-run nondeterminism at a fixed seed. That is a floor for the one-seed confound, not a seed spread.

**Attempt 1, for the record** (from its `rounds.csv`):
- Final round 7 at 79,965 steps.
- Median normalized TEST over its own 15 dojos: 0.767. Over the 15 held-out teacher dojos: 0.819.
- Round times: 1,429 s for round 0 (feature-cache warm-up), then about 720-910 s per 10k-step round (about 75 ms/step, half the pilot's 151 ms).
- Peak VRAM 4.6 GiB allocated, 7.7 GiB reserved. Feature cache: 99.3% of text encodings avoided.

**Queue ETA:** the estimates are updated from attempt 1.
- main_R attempt 2: ~120 min.
- main_T: ~150 min, scaled by the pilot's T/R ms/step ratio of 1.22.
- Queue done around 00:30.
