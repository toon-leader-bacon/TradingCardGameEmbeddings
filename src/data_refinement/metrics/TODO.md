# TODO (metrics)

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
