# TODO (metrics)

- [ ] **A vectorized 17lands scanner.** Even after the 1.7x speedup,
  PremierDraft across the three families would take about 50 h.

- **`isotropic_summary` segfaulted once (2026-10-08 regen).** It exited
  with rc 139 and no Python traceback on day 2 of the 2013 summary
  archive, about 11 min in. An identical re-run under `python -X
  faulthandler` passed (56 min, memory flat at about 215 MB), so the cause
  is unknown and may be flaky. If it recurs, the faulthandler output will
  show the native frame. Logs: `logs/regen_2026-10-08/25_*` and `25b_*`.

- **MtG set labels**, the "hard" label set for intrinsic evaluation
  (the cross-game rarity tier, `cross_game/`, is the "medium" one; game
  labels, the "easy" one, need no metric). Not started.

- [ ] **FaB deck box keys cards by name.** A name printed in several pitches
  maps to one arbitrary printing, so FaB inclusion rates are per name.
- [ ] **StS2 rest-site choice metric** (BRAINSTORM #26). The other
  per-floor picks (card reward, shop purchase, removal, upgrade) are
  built; see `sts2_runs/README.md`, "Per-floor metrics".
