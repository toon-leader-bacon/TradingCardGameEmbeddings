# TODO (metrics)

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
- **Standardize metric constructor argument shape across containers.**
  `sts_gg`'s `CardAverageMetric`/`DeckLabelMetric` always take
  `card_binder` and `deck_box` for signature consistency within that
  container, even when a subclass never uses one; the seventeenlands
  metrics omit a parameter they have no use for. Decide whether one
  convention should win project-wide, or whether each container being
  internally consistent is enough.
- **Deduplicate the running-tally Template Method shape.** `sts_gg`'s
  `CardAverageMetric` (`card_average_metric.py`), `draft_data`'s
  `PackCardTallyMetric` (`pack_card_tally_metric.py`) and `game_data`'s
  `GameCardAverageMetric` (`game_card_average_metric.py`) each implement
  the same per-key running (sum, count) accumulate-then-average shape,
  and several replay_data metrics repeat it again. All three
  seventeenlands families now exist, so the shared shape can be read
  off real code; pull it into one base only where key shapes truly
  match.
