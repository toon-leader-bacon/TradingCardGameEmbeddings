# isotropic/summary

Converts isotropic.org's Wayback-salvaged Flavor A data
(`data/raw/isotropic/*-summary.tar.bz2` - one `games-YYYYMMDD.json`
JSONL member per day, per archive) into Dominion training-data parquet
files, resolving every card name by NAME against the already-ingested
`dominiontabs` `CardBinder` (`GameId.DOMINION` - see
[`../../../card_binder/dominiontabs/README.md`](../../../card_binder/dominiontabs/README.md)).
Full row-shape documentation, confirmed field semantics, and every
known bias in this source lives in
[`../BRAINSTORM.md`](../BRAINSTORM.md) - read that first. This is the
first real-gameplay Dominion source in this project; the sibling
`../../dominiontabs/` container has card-intrinsic metrics only.

Every metric here satisfies the shared `Metric[dict]` Protocol
([`../../metric.py`](../../metric.py)): 10 streaming metrics that write
output rows immediately in `accumulate()`, and 3 accumulation metrics
whose real computation happens in `finalize()` (`veto_rate_metric.py`,
`average_copies_bought_metric.py`, `turn_count_association_metric.py`).
`deck_card_mask_metric.py`'s `WinningDeckMaskedCardMetric` additionally
satisfies `../../generic/deck_card_mask_metric.py`'s `DeckCardMaskMetric`
Template Method - this project's second consumer of that base, after
`play_gwent`'s `LeaderMaskedFromDeckMetric`.

Sibling to `../games/` (not yet built) - isotropic's *other* raw
flavor, one turn-by-turn HTML log per game, covering the mid-game/
sequential metric ideas `../BRAINSTORM.md` flags as Flavor-B-only.

## Files

- `row_utils.py` - shared, stateless row-reading helpers every metric
  in this container uses: `card_uuid_for_name()` (resolve one card
  name against `CardLookup`), `winner_entry()` (the rank==1 player
  with a real final deck, or `None`), `kingdom_card_names()`
  (`board.supply`), `is_natural_kingdom()` (excludes isotropic's own
  curated/challenge kingdom-generator flags -
  `constraints`/`required`/`prohibited`/`force_big_cards`/
  `force_equal_start`/`force_banes`), `eligible_player_entries()`
  (excludes resigned players), `deck_card_uuids()` (expand one
  `end.deck` dict into a flat card multiset), and `deck_for_player()`
  (build one player's full `GenericDeck` - the shared helper
  `full_deck_win_prediction_metric.py`, `deck_pair_winner_metric.py`,
  and `multiplayer_placement_metric.py` all delegate their own
  `_deck_for_player()` to, once three independent consumers reached
  the exact same logic).
- `scanner.py` - `scan_isotropic_summary_archives(archive_paths,
  metrics)` drives every metric in a list over every game row across
  every `games-*.json` member of every given archive (so multiple
  months, e.g. both surviving archives, pool into one training corpus
  in a single call), isolating one metric's `accumulate()`/
  `finalize()` failure (logged, not raised) from every other metric in
  the list - the isotropic analogue of
  [`../../sts_gg/scanner.py`](../../sts_gg/scanner.py)'s
  `scan_runs_jsonl()`, adapted for a tarball-of-JSONL-members archive
  shape instead of one flat file.

### Single-card metrics

- `veto_rate_metric.py` - `VetoRateMetric` (accumulation):
  `P(card in vetoed | card in board.supply)`, restricted to natural
  kingdoms.
- `copies_bought_distribution_metric.py` -
  `CopiesBoughtDistributionMetric` (streaming): one raw
  `(card, copies)` sample row per distinct card in every eligible
  player's final deck - a consumer buckets/histograms this itself,
  unlike the reduced-mean sibling below.
- `average_copies_bought_metric.py` - `AverageCopiesBoughtMetric`
  (accumulation): `E[end.deck[card] | card in end.deck]`.
- `turn_count_association_metric.py` - `TurnCountAssociationMetric`
  (accumulation): per-card average winner turns minus the whole
  natural-kingdom corpus's average winner turns (a signed float) -
  keeps a second, corpus-wide running tally alongside its per-card
  ones specifically for this baseline subtraction.

### Multi-card metrics

- `full_deck_win_prediction_metric.py` - `FullDeckWinPredictionMetric`
  (streaming): one eligible player's final deck -> `rank == 1`.
- `kingdom_veto_prediction_metric.py` - `KingdomVetoPredictionMetric`
  (streaming): the pre-veto candidate pool (`board.supply` union
  `vetoed`, stored via `DeckBox` as a generic card-group, not a real
  deck) -> which card(s) got vetoed. Skips rows with no real veto.
- `kingdom_game_length_metric.py` - `KingdomGameLengthMetric`
  (streaming): the kingdom -> winner turn count, restricted to natural
  kingdoms with a resolvable winner.
- `kingdom_member_label_metric.py` - `KingdomMemberLabelMetric`
  (streaming, abstract): Template Method base for "given the kingdom,
  label each individual kingdom card from the winner's final deck" -
  one output row per (kingdom, kingdom card) pair.
- `kingdom_member_label_metrics.py` - two concrete
  `KingdomMemberLabelMetric` subclasses: `WinningDeckMembershipMetric`
  (is the card in the winner's deck at all) and
  `WinningDeckCountMetric` (how many copies).
- `deck_card_mask_metric.py` - `WinningDeckMaskedCardMetric`
  (`DeckCardMaskMetric` subclass): winner's final deck, one card
  masked out, -> that card's name. Masks uniformly at random among the
  winner's *non-basic* cards (Copper/Silver/Gold/Platinum/Potion/
  Estate/Duchy/Province/Colony/Curse excluded - present in nearly
  every deck regardless of kingdom, so masking one carries far less
  signal than a kingdom card the winner specifically chose).
  `LABEL_VALUES` is populated once, at class-definition time, by
  reading the real `data/final/cards/dominion.jsonl` directly.
  **`_deck_uuid_for_row()` raises `ValueError` for any row with no
  resolvable winner** (e.g. a solo/practice game, ~7% of the corpus) -
  `DeckCardMaskMetric`'s own contract calls this unconditionally,
  before the row's masking target is even checked, so a per-row
  failure here is expected and handled by `scanner.py`'s isolation,
  not a bug - confirmed via a full-corpus scan, which completes
  cleanly end to end despite these per-row exceptions.
- `deck_card_set_copy_count_metric.py` - `DeckCardSetCopyCountMetric`
  (streaming): one eligible player's *distinct* card set (counts
  stripped, stored via `DeckBox` as its own group, never conflated
  with a real deck's uuid) -> each card's actual count in that deck.

### Multi-group metrics

- `deck_pair_winner_metric.py` - `DeckPairWinnerMetric` (streaming):
  both decks from a real 2-player game -> which one won, written as
  `(deck_uuid_lo, deck_uuid_hi, lo_won)` with the two decks in
  ascending-`deck_uuid` order (never isotropic's own `players[]`
  order) - a model trained on this must never be able to learn
  "whichever deck the source data lists first tends to win."
- `multiplayer_placement_metric.py` - `MultiplayerPlacementMetric`
  (streaming): every deck in a 3-4 player game -> the full finishing
  order, as parallel `deck_uuids`/`ranks` lists, canonically sorted by
  `deck_uuid` for the same ordering-leakage reason as the pair metric
  above. A single resignation anywhere in the game discards the whole
  row (not just that one player) - the simplest resolution to a
  question `../BRAINSTORM.md` originally left open.

## Known biases / known-unknowns

See [`../BRAINSTORM.md`](../BRAINSTORM.md)'s "Known biases" section
for the full list (Wayback survivorship, no skill/ELO signal,
resignation censoring of `end.deck`, non-uniform kingdom generation,
solo/practice games). Additionally:

- Every `GenericDeck` this container mints has `provenance=None` (the
  default) - these are private, metrics-only `DeckBox` entries, not
  canonical ingested decks, mirroring
  [`../../sts_gg/deck_label_metric.py`](../../sts_gg/deck_label_metric.py)'s
  own construction.
- `WinningDeckMaskedCardMetric`'s masking-target choice (uniform random
  among non-basic cards actually in the winning deck) is one
  reasonable rule among several `../BRAINSTORM.md` flagged as
  plausible - "the card with the most copies" and "a fixed card type"
  were considered and rejected (see that class's own module docstring
  for why), not exhaustively ruled out for every possible future use.
