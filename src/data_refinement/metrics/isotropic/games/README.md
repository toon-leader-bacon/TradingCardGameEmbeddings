# isotropic/games

Converts isotropic.org's Wayback-salvaged Flavor B data
(`data/raw/isotropic/2010_20101011.tar.bz2`,
`data/raw/isotropic/2013_20130315.tar.bz2` - one turn-by-turn game-log
HTML file per game) into Dominion training-data parquet files. Two
parsing layers: `header_parser.py` reads each file's HEADER block
(winner, pile-exhaustion sentence, kingdom, per-player score/turns/
opening buy) for the six header metrics; `game_log_parser.py` composes on
top of it and also reads the turn-by-turn `<hr/><b>Game log</b>` section
(buys, gains, trashes, plays per player-turn) for the five mid-game
metrics. Full row-shape documentation, confirmed field semantics, and
every known bias lives in [`../BRAINSTORM.md`](../BRAINSTORM.md) - read
that first. Card names resolve via
[`../card_names.py`](../card_names.py) against the
already-ingested `dominiontabs` `CardBinder` (`GameId.DOMINION`),
shared with [`../summary/`](../summary/README.md).

Sibling to `../summary/` (Flavor A, already implemented) - this
container is the first in the project able to answer questions Flavor
A's structured JSON cannot: which supply pile ran out, and what a
player's opening two buys actually were.

Every metric here satisfies the shared `Metric[...]` Protocol
([`../../metric.py`](../../metric.py)), specialized to a parsed row type
rather than a raw dict: the six header metrics are `Metric[GameHeader]`
(4 streaming, writing rows immediately in `accumulate()`, and 2
accumulation metrics whose real computation happens in `finalize()` -
`opening_buy_rate_metric.py`, `pile_exhaustion_rate_metric.py`); the five
mid-game metrics are all streaming `Metric[GameLog]`.

## Files

- `header_parser.py` - `parse_game_header(html_text) -> GameHeader |
  None`, using BeautifulSoup's `get_text()` (matching this project's
  own HTML-parsing convention - see `../../../fabtcg_decklists/
  fragment_parsing.py` and `../../../card_binder/gwent_one/
  ingestion_stage.py`) to get clean, entity-decoded, tag-free header
  text regardless of which of isotropic's two confirmed client-era
  markup shapes a given file uses. Returns `None` for a game outside
  this pass's scope: no `<pre>` element at all, no real `"{nick}
  wins!"` first line (a fully-resigned game), or a second line that's
  a resignation summary instead of a real pile-exhaustion sentence
  (confirmed live: a game WITH a genuine winner can still show "All
  but one player has resigned." there, when a different, non-winning
  player quit mid-game). A resigned player's own block is silently
  skipped (not the whole game) when the winner and ending are both
  otherwise valid.
- `game_log_parser.py` - `parse_game_log(html_text) -> GameLog | None`,
  a `GameHeader` plus a tuple of `Turn`s (`player_nick`, per-player
  `turn_number`, and `CardQuantity` tuples for `cards_bought`/
  `cards_gained`/`cards_trashed`/`cards_played`). A `CardQuantity` is the
  card text as isotropic renders it (still English-plural when count > 1)
  plus a count - it never touches a CardBinder. Handles both client eras
  (2013 `— nick's turn N —` headers; 2010 `--- nick's turn ---` with no
  number, so turns are numbered by counting). Only the turn owner's own
  events are captured; see Known limitations below and `TODO.md`.
- `partial_deck.py` - `partial_deck_card_uuids()` (a player's deck just
  before their turn T: 7 Copper + 3 Estate + gains - trashes),
  `expand_card_quantity()`, `distinct_card_uuids()`. Name lookups are
  memoized (a CardBinder lookup deep-copies the card).
- `row_utils.py` - `winner_player()` (the `GameHeaderPlayer` matching
  `header.winner_nick`), `is_single_pile_ending()`, and
  `pile_card_uuid_for_name()` - the last one is NOT a plain
  `card_uuid_for_name()` call: `GameHeader.exhausted_pile_names` is
  English-pluralized exactly as isotropic renders it ("Provinces",
  "Duchies", "Wharves" for Wharf, or an unchanged "Ironworks"), and no
  single depluralization rule inverts every real Dominion card name
  losslessly (confirmed live for all of the above, plus "Witches" -
  needs "-es" stripped, not "-s" - and "Duchesses"). This function
  tries several depluralization candidates against the real CardBinder
  and returns the first that actually resolves, rather than guessing
  once and risking silent mis-resolution.
- `scanner.py` - `scan_isotropic_game_log_archives(archive_paths,
  metrics)` (header metrics) and `scan_isotropic_game_logs_archives()`
  (mid-game `Metric[GameLog]` metrics) share one tar-member iterator; the
  former drives every metric in a list over every parseable
  `GameHeader` across every HTML member of every given archive,
  isolating both a per-metric `accumulate()`/`finalize()` failure and a
  per-member `parse_game_header()` exception (logged, not raised, in
  either case) - the games/ analogue of
  [`../summary/scanner.py`](../summary/scanner.py)'s
  `scan_isotropic_summary_archives()`, adapted for a tarball-of-HTML-
  files archive shape instead of one flat JSONL-per-day file.
  `201010_11_all.tar.bz2` is a byte-identical duplicate of
  `2010_20101011.tar.bz2` and should never be passed alongside it.

### Single-card metrics

- `opening_buy_rate_metric.py` - `OpeningBuyRateMetric`
  (accumulation): `P(card in {opening 1st buy, opening 2nd buy} | card
  in kingdom_card_names)`, tallied across EVERY player in every
  natural-kingdom game, not just the winner - a corpus-wide "is this a
  commonly-played opening" question, not a winning-opening one (that's
  `kingdom_opening_buy_prediction_metric.py`'s job).
- `pile_exhaustion_rate_metric.py` - `PileExhaustionRateMetric`
  (accumulation): `P(card's pile exhausted by game end | card in
  kingdom_card_names)` - Flavor-A-impossible, since Flavor A's JSON has
  no notion of which supply piles ran out.

### Multi-card metrics

- `kingdom_opening_buy_prediction_metric.py` -
  `KingdomOpeningBuyPredictionMetric` (streaming): the kingdom -> the
  eventual winner's opening buy (1-2 cards, variable-set label - the
  opening buy(s) range over dominiontabs' whole card pool, including
  Silver, a non-kingdom card that's still a legal opening buy).
- `opening_buy_outcome_metric.py` - `OpeningBuyOutcomeMetric`
  (streaming): the inverse question, one row PER PLAYER per game (not
  winner-only) - given only a player's own opening buy plus the
  kingdom, did that player win. Needs both winning and losing openings
  to answer "how much of the outcome is already determined by turn
  one," which a winner-only framing couldn't.
- `kingdom_ending_pile_prediction_metric.py` -
  `KingdomEndingPilePredictionMetric` (streaming): one row per
  (kingdom, kingdom card) pair - was this specific card's pile among
  the ones exhausted by game end. Membership is tested by UUID (via
  `row_utils.pile_card_uuid_for_name()`), never by comparing name
  strings, since one side is plural and the other singular.
- `kingdom_ending_type_metric.py` - `KingdomEndingTypeMetric`
  (streaming): a coarser, cheaper sibling of the above - just the
  ending TYPE (`province`/`colony`/`multi_pile`), not which specific
  piles. Resolves the real "Province"/"Colony" card uuids once in
  `__init__` and compares by uuid, never by string.

### Mid-game metrics (all `Metric[GameLog]`, streaming)

Every row is one player-turn checkpoint; the partial deck is written to
the metrics-private `DeckBox` and referenced by uuid.

- `mid_game_next_buy_metric.py` - `NextBuyPredictionMetric`:
  (partial deck, kingdom) -> the card(s) bought that turn (variable set).
- `mid_game_next_trashed_card_metric.py` - `NextTrashedCardMetric`:
  partial deck -> the card(s) trashed that turn, only turns with a trash.
- `mid_game_next_turn_action_count_metric.py` -
  `NextTurnActionCountMetric`: partial deck -> number of Action cards the
  same player plays on their next turn (Action-ness read from
  `raw_content["types"]`; the parser records every played card).
- `mid_game_win_probability_metric.py` - `EventualWinProbabilityMetric`:
  (partial deck, kingdom) -> did this player eventually win, every turn.
- `mid_game_deck_pair_winner_metric.py` - `MidGameDeckPairWinnerMetric`:
  both players' partial decks at the same turn -> did the lower-uuid deck
  win (2-player games only; uuid-sorted so `players[]` order can't leak;
  checkpoints where both decks are identical are skipped).

## Known limitations of the mid-game partial deck

Only the turn owner's own gains/trashes are attributed. A line naming a
different player (a Witch's Curse to the opponent) is dropped, as are
Ambassador returns and Trader replacements. Measured against each file's
own final-deck line, the rebuilt deck matches exactly for ~55% of players
(~65% ignoring Curses). This was a deliberate ship-as-is call - see
[`TODO.md`](TODO.md) for the problem, the numbers, and a fix sketch.

## Known biases / known-unknowns

See [`../BRAINSTORM.md`](../BRAINSTORM.md)'s "Known biases" section for
the shared isotropic-wide list (Wayback survivorship, no skill/ELO
signal, non-uniform kingdom generation). Additionally:

- **Only two days survived** (2010-10-11's 108 games, 2013-03-15's
  10,815 games) - a much smaller, less representative sample than
  Flavor A's ~300K pooled games, and drawn from isotropic's client at
  two very different points in its lifetime (confirmed two distinct
  markup/rendering eras).
- **Resignation-tainted games are entirely out of scope for this
  container** - not just fully-resigned games (no winner at all), but
  also winner-decided games where a DIFFERENT player resigned mid-game
  (confirmed live, common enough to matter: roughly 16% of a 10,000-
  game live sample). `header_parser.py` returns `None` for both cases;
  neither is silently miscounted as a natural pile-exhaustion ending.
- Every `GenericDeck` this container mints has `provenance=None` (the
  default) - private, metrics-only `DeckBox` entries, matching
  `../summary/`'s own convention.
- `row_utils.pile_card_uuid_for_name()`'s candidate list was built
  against every irregular plural actually observed in live sampling
  (Wharf/Ironworks/Witch/Duchess) - a Dominion card with a plural form
  outside all five tried candidates (unchanged, "-ies"→"-y",
  "-ves"→"-f", "-es"-stripped, "-s"-stripped) would fail to resolve,
  logged and excluded like any other unresolved name, not crash.
