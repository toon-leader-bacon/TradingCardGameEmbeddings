# TODO (isotropic summary metrics)

Found while writing the isotropic dojos (2026-09-30).

- [ ] **WinningDeckMaskedCardMetric.LABEL_VALUES is too wide.** It lists
  807 names, but only 162 ever appear as labels. Restrict it to the names
  that occur.
- [ ] **Name the deck reference column deck_uuid.** Several metrics use
  kingdom_uuid / partial_deck_uuid, which needed
  RenamedColumnDataConstructor in the dojos. The alternative is a
  deck_uuid_column argument on the generic DeckLabelDataConstructor.
