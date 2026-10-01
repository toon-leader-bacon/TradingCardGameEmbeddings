# TODO (isotropic summary metrics)

Found while writing the isotropic dojos (2026-09-30).

- [ ] **veto_rate is all zeros.** All 160 rows are 0.0, so the veto field
  parse is probably wrong. No dojo was built for it.
- [ ] **kingdom_game_length has a heavy tail.** Labels run 0-323 turns
  (mean 19.8, std 6.4), and about 2% are under 5 turns (early
  resignations?). Clip, filter, or log-transform before training; see
  src/training/TODO.md, "Handle outlier regression labels".
- [ ] **One row per (kingdom, card) leaks across splits.**
  winning_deck_membership, winning_deck_count and
  kingdom_ending_pile_prediction write one row per (kingdom, card), so a
  row-level split puts the same kingdom in both TRAIN and TEST. Emit one
  row per kingdom with a label vector instead.
- [ ] **WinningDeckMaskedCardMetric.LABEL_VALUES is too wide.** It lists
  807 names, but only 162 ever appear as labels. Restrict it to the names
  that occur.
- [ ] **Expose LABEL_COLUMN ClassVars.** The card-rate metrics have none,
  which is why src/dojos/isotropic has its own IsotropicCardRateDojo base
  instead of reusing CardAverageMetricDojo.
- [ ] **Name the deck reference column deck_uuid.** Several metrics use
  kingdom_uuid / partial_deck_uuid, which needed
  RenamedColumnDataConstructor in the dojos. The alternative is a
  deck_uuid_column argument on the generic DeckLabelDataConstructor.
