# TODO (metrics)

- **The registered 17lands metric families have never been run.**
  `dominiontabs`, `isotropic_summary` and `isotropic_games` were
  registered in `scripts/run_metrics.py` and run to completion on
  2026-09-27: all 3 + 24 outputs written and stamped with
  `card_binder_version` (isotropic: summary 41 min, games 19 min; the
  private `data/metrics/isotropic/deck_box.db` holds 1.7M decks,
  4.4 GB). Run heavy jobs one at a time; two in parallel ran the
  machine out of memory.
- **`WinningDeckMaskedCardMetric` raises on every game with no usable
  winner** (`isotropic/summary/deck_card_mask_metric.py`,
  `_deck_uuid_for_row()`) instead of skipping it quietly as its
  sibling metrics do. The scanner isolates it, so outputs are right,
  but a full run logs 9,411 full tracebacks, which buries real errors.
  Filter those rows out before `accumulate()` raises.

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
