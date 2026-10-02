# TODO (isotropic summary metrics)

Found while writing the isotropic dojos (2026-09-30).

- [x] **veto_rate is all zeros.** Fixed 2026-10-01: vetoed cards are never
  in board.supply, so P(vetoed | in supply) was 0 by construction. Now
  P(vetoed | offered), offered = supply + vetoed; "*Name" vetoes resolve.
- [x] **kingdom_game_length has a heavy tail.** Fixed 2026-10-01: solo games
  and games with a resignation are skipped (the under-5-turn labels were
  resignation wins); KingdomGameLengthDojo also clips labels at 50.
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
