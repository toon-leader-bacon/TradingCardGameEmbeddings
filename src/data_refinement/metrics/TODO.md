# TODO (metrics)

- [ ] **A vectorized 17lands scanner.** Even after the 1.7x speedup,
  PremierDraft across the three families would take about 50 h.

- **`isotropic_summary` segfaulted once (2026-10-08 regen).** It exited
  with rc 139 and no Python traceback on day 2 of the 2013 summary
  archive, about 11 min in. An identical re-run under `python -X
  faulthandler` passed (56 min, memory flat at about 215 MB), so the cause
  is unknown and may be flaky. If it recurs, the faulthandler output will
  show the native frame. Logs: `logs/regen_2026-10-08/25_*` and `25b_*`.

- **Cross-game rarity metric family.** Build one metric per game,
  each mapping that game's own rarity into a single shared rarity
  enum. The enum was proposed as common / uncommon / rare / legendary /
  unique-promo / not-applicable / other, but is not final. Outputs are
  per-card parquets with one row per `nocab_uuid` and a `label` column
  holding the enum value. That shape lets dojos (masking) and
  `src/evaluation/` use it the same way: evaluation reads several
  per-game parquets as one label source (`MetricParquetLabels`). This
  is the "medium" label set for intrinsic evaluation. The "easy" set,
  game labels, needs no metric; the "hard" set, MtG set labels, comes
  later. The precedent is `gwent_one/rarity_mask_metric.py`, which is
  per-game and not normalized. Open decisions:
  - where the enum lives;
  - how each game's values map, e.g. whether Gwent's `epic` becomes
    `rare`/`legendary` or `other`;
  - what games with no rarity get: `not-applicable` rows, or no metric.
- **Deduplicate the running-tally Template Method shape.** `sts_gg`'s
  `CardAverageMetric` (`card_average_metric.py`), `draft_data`'s
  `PackCardTallyMetric` (`pack_card_tally_metric.py`) and `game_data`'s
  `GameCardAverageMetric` (`game_card_average_metric.py`) each implement
  the same per-key running (sum, count) accumulate-then-average shape,
  and several replay_data metrics repeat it again. All three
  seventeenlands families now exist, so the shared shape can be read
  off real code; pull it into one base only where key shapes truly
  match.

- [ ] **FaB deck box keys cards by name.** A name printed in several pitches
  maps to one arbitrary printing, so FaB inclusion rates are per name.
- [ ] **StS2 per-floor metrics (B2).** The spire_codex run
  records carry per-floor `card_choices`, shop purchases, removals,
  upgrades and rest-site choices, none used yet. Top candidates: card
  reward pick given the deck so far (`[offered, partial deck]`, the same
  shape as 17lands' pool_conditioned_pick), shop purchase, removal
  target, upgrade target, rest-site choice. Needs the deck at each floor
  rebuilt from `cards_gained`/`cards_removed`. Design note in
  `sts2_runs/README.md`.
