# TODO (metrics)

- [ ] **Make the 17lands outputs consumable by dojos** (2026-09-30).
  A partial run exists: game_data for 76 set/format directories; draft and
  replay only one shakeout file each. See
  `plans/seventeenlands_metrics_run.md`.
  - Outputs are written per raw file, at
    `data/metrics/seventeenlands/<family>/<SET>/<Format>/<stem>.parquet`.
    But each dojo reads its metric's single DEFAULT_OUTPUT_PATH, which
    does not exist.
  - Proposed: a merge step per (family, metric) that writes to
    DEFAULT_OUTPUT_PATH:
    - stack the streaming files (deck and pool rows); they share the
      family deck box;
    - recombine card averages weighted by `sample_count`, keeping a set
      column;
    - subsample tutor_target_pool (151.8M rows).
  - Alternative: one catalog key per set and format (hundreds of keys).
- [ ] **Cache `CardBinder.version_for`.** It takes about 3 s and every
  17lands metric constructor calls it, once per CSV.
- [ ] **Unwrap or re-download the 19 tar-wrapped 17lands "CSV" files**
  (AFR, KHM, MID, STX, VOW; listed in `plans/seventeenlands_metrics_run.md`).
- [ ] **A vectorized 17lands scanner.** Even after the 1.7x speedup,
  PremierDraft across the three families would take about 50 h.

- **The registered 17lands metric families have never been run.**
  `dominiontabs`, `isotropic_summary` and `isotropic_games` were
  registered in `scripts/run_metrics.py` and run to completion on
  2026-09-27: all 3 + 24 outputs written and stamped with
  `card_binder_version` (isotropic: summary 41 min, games 19 min; the
  private `data/metrics/isotropic/deck_box.db` holds 1.7M decks,
  4.4 GB). Run heavy jobs one at a time; two in parallel ran the
  machine out of memory.
- [x] **`WinningDeckMaskedCardMetric` no longer raises on a game with
  no usable winner** (2026-09-29). `../generic/deck_card_mask_metric.py`'s
  `DeckCardMaskMetric.accumulate()` and its `_deck_uuid_for_row()`
  contract now allow returning `None` to mean "this row has no deck at
  all, skip it entirely" (a real, expected outcome, not an error) -
  `accumulate()` returns immediately in that case, before touching
  `deck_box` or the target-lookup path. `WinningDeckMaskedCardMetric`
  (isotropic's own `_deck_uuid_for_row()`) now returns `None` instead
  of raising `ValueError` when `row_utils.winner_entry(row)` is `None`,
  matching how sibling metrics (e.g. `DeckPairWinnerMetric`) already
  skip an ineligible row quietly. `play_gwent`'s
  `LeaderMaskedFromDeckMetric` (the base's only other consumer) is
  unaffected - its own `_deck_uuid_for_row()` always resolves a deck,
  never returns `None`. A full run no longer logs a traceback per
  no-winner game.

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
- [x] **Standardize metric constructor argument shape within sts_gg**
  (2026-09-29). Decision: keep `sts_gg`'s existing "every metric in
  this container takes the same params" convention, rather than
  switching to seventeenlands' "omit what you don't use" one - a
  deliberate style preference, not forced by any real need. Applied it
  fully: `DeckLabelMetric.__init__` (`deck_label_metric.py`) now takes
  `deck_box: DeckBox | None = None`, same as `CardAverageMetric`
  already did, instead of a required `deck_box: DeckBox` - it still
  raises `ValueError` if `deck_box` is `None`, so it's Optional in type
  only, required in practice. Every sts_gg metric constructor now has
  the identical `(card_binder, deck_box, output_path)` shape.
  `scripts/run_metrics.py`'s `run_sts_gg()` was simplified accordingly:
  the old hand-picked-args-per-metric list is now one
  `_STS_GG_METRIC_CLASSES` tuple plus `[cls(binder, deck_box) for cls
  in _STS_GG_METRIC_CLASSES]` - every class accepts `deck_box` now, so
  one call shape covers all 24.
  NOT extended to `gwent_one`/`dominiontabs`: unlike sts_gg, neither
  container's driver function (`run_gwent_one()`/`run_dominiontabs()`)
  constructs any deck-consuming metric at all today - their shared
  `MaskedFieldMetric` base has no deck_box-accepting sibling to be
  consistent with, so adding one there would be a purely speculative,
  permanently-unused parameter with no current counterpart to match,
  unlike sts_gg's real CardAverageMetric/DeckLabelMetric pairing.
  Revisit if either container grows a deck-based metric.
- **Deduplicate the running-tally Template Method shape.** `sts_gg`'s
  `CardAverageMetric` (`card_average_metric.py`), `draft_data`'s
  `PackCardTallyMetric` (`pack_card_tally_metric.py`) and `game_data`'s
  `GameCardAverageMetric` (`game_card_average_metric.py`) each implement
  the same per-key running (sum, count) accumulate-then-average shape,
  and several replay_data metrics repeat it again. All three
  seventeenlands families now exist, so the shared shape can be read
  off real code; pull it into one base only where key shapes truly
  match.

- [ ] **`run_metrics.py --source play_gwent` (and `--all`) rewrites the
  published `data/final/decks/gwent.db`** (found 2026-10-01). `run_play_gwent`
  opens the published box, `LeaderMaskedFromDeckMetric` writes decks into it,
  and the run calls `save()` on it. A re-run makes every metric keyed to
  that box stale (e.g. `final_decks/held_out_card_gwent.parquet`). Fix: give
  the leader metric a metric-private box (as isotropic and sts_gg do), or
  open the published box read-only. Until then, don't run that family.
- [ ] **FaB deck box keys cards by name.** A name printed in several pitches
  maps to one arbitrary printing, so FaB inclusion rates are per name.
- [ ] **Move `RenamedColumnDataConstructor`** from `dojos/isotropic/` to
  `dojos/generic/data_constructors/`: it now has three users.
- [ ] **Deduplicate FaB class/typebox parsing** between
  `metrics/cardvault_fabtcg/` and `metrics/fabtcg_decklists/`.
- [ ] **StS2 per-floor metrics (B2).** The spire_codex + sts2runs run
  records carry per-floor `card_choices`, shop purchases, removals,
  upgrades and rest-site choices, none used yet. Top candidates: card
  reward pick given the deck so far (`[offered, partial deck]`, the same
  shape as 17lands' pool_conditioned_pick), shop purchase, removal
  target, upgrade target, rest-site choice. Needs the deck at each floor
  rebuilt from `cards_gained`/`cards_removed`. Design note in
  `sts2_runs/README.md`.
- [ ] **held_out_card rows can share a deck across splits.** Dojo splits
  are by row, and Pokemon, FaB and Gwent take several targets per deck,
  so one deck can appear in TRAIN and TEST with different held-out cards
  (the big boxes take one target per deck and are unaffected). Split by
  deck uuid instead, or take one target per deck everywhere.
