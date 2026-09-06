# TODO

- **Structured/grouped deck output.** `DecklistCardsMetric` currently
  flattens every card into one `card_nocab_uuids` list, discarding
  which group (Hero / Weapon / Equipment, or Pitch 1/2/3) each card
  came from — a deliberate first-pass simplification, not an
  oversight. A follow-up stage could instead emit one column (or field)
  per group, e.g. `hero_nocab_uuid`, `weapon_nocab_uuid`,
  `equipment_nocab_uuids`, `pitch_1_nocab_uuids`, `pitch_2_nocab_uuids`,
  `pitch_3_nocab_uuids`. Motivation: multi-group-aware training
  signal — e.g. masking out the hero slot and training a model to
  predict it from the rest of the deck, or similar per-slot metrics
  that a flat list can't express.
  The group label is already available for free at the point of
  extraction (the `<h3>` text on each `div.list-view-container` a
  card's `<li>` sits under) — a future stage doesn't need to re-derive
  the DOM walk in `_extract_card_uuids`, just thread that label through
  instead of discarding it.
