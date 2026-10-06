# TODO (isotropic summary metrics)

Found while writing the isotropic dojos (2026-09-30).

- [x] **veto_rate is all zeros.** Fixed 2026-10-01: vetoed cards are never
  in board.supply, so P(vetoed | in supply) was 0 by construction. Now
  P(vetoed | offered), offered = supply + vetoed; "*Name" vetoes resolve.
- [x] **kingdom_game_length has a heavy tail.** Fixed 2026-10-01: solo games
  and games with a resignation are skipped (the under-5-turn labels were
  resignation wins); KingdomGameLengthDojo also clips labels at 50.
- [x] **One row per (kingdom, card) leaks across splits.**
  winning_deck_membership, winning_deck_count and
  kingdom_ending_pile_prediction write one row per (kingdom, card), so a
  row-level split puts the same kingdom in both TRAIN and TEST. Fixed
  2026-10-02 at split time, not by reshaping the metrics: their dojos
  split by `kingdom_uuid` (`DojoConfig.split_group_column`), and
  deck_card_set_copy_count's (one row per (deck, card)) by
  `deck_set_uuid`. The old split files were deleted.
- [ ] **WinningDeckMaskedCardMetric.LABEL_VALUES is too wide.** It lists
  807 names, but only 162 ever appear as labels. Restrict it to the names
  that occur.
- [x] **Expose LABEL_COLUMN ClassVars.** Fixed 2026-10-06: the five
  card-rate metrics (AverageCopiesBoughtMetric, TurnCountAssociationMetric,
  VetoRateMetric, OpeningBuyRateMetric, PileExhaustionRateMetric) now carry
  a `LABEL_COLUMN: ClassVar[str]`, so `src/dojos/isotropic/card_rate_dojos.py`'s
  wrappers subclass `CardAverageMetricDojo` directly, naming only `METRIC`;
  the old `IsotropicCardRateDojo` base is gone.
- [ ] **Name the deck reference column deck_uuid.** Several metrics use
  kingdom_uuid / partial_deck_uuid, which needed
  RenamedColumnDataConstructor in the dojos. The alternative is a
  deck_uuid_column argument on the generic DeckLabelDataConstructor.
