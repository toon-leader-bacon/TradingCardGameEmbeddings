# BRAINSTORM (isotropic / Dominion real played games)

A candidate-metric list for `data/raw/isotropic/*.tar.bz2`, Wayback-
salvaged remnants of isotropic.org's Dominion game logs (see
`src/data_retrieval/isotropic/downloader.py` for how these 5 files were
found and why nothing else survives, and
`src/data_retrieval/dominion/todo.md`'s "Real played decks / game logs"
section for the wider search this closed out). Card-intrinsic joins
below assume `data/final/cards/dominion.jsonl`
(`DominionTabsCardIngestionStage`, see `../dominiontabs/README.md`) is
already ingested, which it is.

This is the first and only real-gameplay Dominion source in this
project — `../dominiontabs/BRAINSTORM.md` explicitly notes "no deck,
draft, or game-outcome data exists for Dominion in this project"; that
line is now stale as of this file existing.

## Implementation status

**Flavor A (`summary/`) is built** — all thirteen metrics named below
as implemented (single-card #1/#2/#3/#4-as-#11, multi-card #2/#3/#4,
the "New candidates" section's #12/#13/#19/#22, and both "Multi Group
Metrics" entries) are real code today; see
[`summary/README.md`](summary/README.md) for what each one actually
computes, its output schema, and design decisions made during
implementation (e.g. `WinningDeckMaskedCardMetric`'s masking-target
rule). This file is left otherwise intact — including the now-built
ideas' own ranked-candidate write-ups below — both as the design
rationale summary/README.md's terser file-by-file listing doesn't
repeat, and because every idea marked *(Flavor B only)* still describes
real, unbuilt work for a future `games/` subpackage; deleting or
renumbering entries here would break the many "single-card idea #N" /
"multi-card #N" cross-references those Flavor-B ideas make back into
this same list.

**Flavor B (`games/`) is partially built** — a header parser
(`games/header_parser.py`) plus a full turn-by-turn log parser
(`games/game_log_parser.py`, both via BeautifulSoup) and eleven metrics
are real code today: six header metrics (the "More Flavor B-only
candidates" section's single-card #1/#2 and multi-card/multi-group #1-#4:
opening-buy rate/prediction/outcome and pile-exhaustion rate/prediction/
ending-type) and five mid-game metrics (the "New candidates raised during
human review" section's multi-card/multi-group #3-#7: next buy, next
trashed card, next-turn action count, eventual win probability, and
mid-game deck pair winner). See [`games/README.md`](games/README.md); the
mid-game partial deck is known to be ~55% exact (attack-forced gains and
trashes onto the opponent are dropped) - see
[`games/TODO.md`](games/TODO.md). Still unbuilt: the resignation metrics
(single-card #3, multi-card #5/#6 in the "More Flavor B-only candidates"
section) and the trash-summary-line metrics (#9/#10 in "New candidates").

## What the raw data actually looks like — two genuinely different flavors

Confirmed by extracting and sampling all 5 archived files directly
(counts below, not estimates):

### Flavor A — monthly "summary" tarballs (`*-summary.tar.bz2`)

One JSON-lines file per day (`games-YYYYMMDD.json`), one compact JSON
object per **completed game**, all players in that game on one row.
This is the dense, structured, high-volume flavor:

- `2010_201012-summary.tar.bz2`: 20 days (Dec 11-30, 2010), **66,707**
  game rows total.
- `2013_201303-summary.tar.bz2`: 15 days (Mar 1-15, 2013), **232,333**
  game rows total — isotropic's playerbase grew roughly 10x in the 2.5
  years between these two snapshots, worth keeping in mind if anything
  below stratifies "era" as a feature.

Confirmed row shape (sampled 18,158 rows from `games-20130301.json`):

```json
{"board": {"supply": [...11 card names...], "constraints": [["set=prosperity", 3, 10]], ...},
 "automatch": true, "lobby": "casual", "vetoed": ["Menagerie", "Grand Market"],
 "serial": 14849, "start_time": 1362124405, "end_time": 1362124802,
 "text_log": "201303/01/game-20130301-000002-3b7999d7.html.gz",
 "players": [
   {"id": "D8YwA/qAB/Z5MGyf9EjevXeF6P8", "nick": "unlikely", "mode": "images",
    "rank": 1, "score": 139, "turns": 54,
    "end": {"deck": {"Bank": 1, "Duchy": 8, "Estate": 11, ...},
            "vps": {"Duchy": 24, "Estate": 11, "Province": 48, "Vineyard": 56}}}]}
```

- **`board.supply`**: always exactly the 10 (rarely 11, with a Potion
  or Colony/Platinum-enabling card) kingdom card names for that game —
  a direct multi-card "kingdom" group, joinable to `nocab_uuid`s via
  `CardBinder.get_by_name_single(GameId.DOMINION, name)`.
- **`board.constraints`** (179/18,158 sampled) is how isotropic
  recorded a curated/challenge kingdom, e.g. `["set=prosperity", 3,
  10]` (at least 3, at most 10 cards from Prosperity). Other
  `board.*` flags confirmed present at nontrivial rates: `point_tracker`
  (34%, VP-token-scoring cards active), `black_market` (5.7%),
  `bane` (5.5%, Young Witch's extra 11th card), `required`/
  `prohibited`/`force_big_cards`/`force_equal_start`/`force_banes`
  (each under 2.5%) — these are isotropic's own kingdom-generator knobs,
  not vanilla Dominion rules, and matter for any metric conditioning on
  "was this a normal random kingdom."
- **`lobby`**: `casual` (90.6%), `meetup` (9.3%), `order` (0.03%) —
  worth excluding `order`'s handful of rows as noise, not a real
  stratum.
- **`automatch`** (83% of rows): true when isotropic's own matchmaker
  paired the players, vs. a manually-arranged table — a rough proxy
  for "these two players didn't specifically choose each other,"
  relevant to any player-vs-player metric.
- **`vetoed`** (40% of rows): card names the matched players banned
  from that game's kingdom before it was dealt — a real preference
  signal ("what do good/average players avoid"), distinct from cards
  that just weren't in the kingdom to begin with.
- **`players`**: 1-4 entries (2 players: 88%, 1: 7% solo/practice
  games, 3: 4.4%, 4: 0.3%). Each has `id` (a stable per-account hash,
  `null` for guests — usable to link one human across many games),
  `nick`, `mode` (`images` vs `text` client), `rank` (this game's
  **finishing placement**, 1st/2nd/3rd/4th — NOT a skill rating; no
  skill/ELO field exists anywhere in this source, unlike the
  Glicko/mu-filtered stats `dominion/todo.md` documents from
  `markus`'s external pipeline), `turns`, and optionally `resigned`
  (16% of players; that player's `end` block is then absent — no
  final deck to read for a resignation) and `bias` (rare, an
  isotropic handicap setting, seen but not sampled at any real rate
  here).
- **`end.deck`**: final deck composition as `{card_name: count}` —
  this IS a full deck list, but notably **not filtered to only cards
  that were actually purchasable in that game's kingdom** — treasures/
  victory basics plus whatever kingdom cards were bought. A real
  `DeckBox`-eligible full deck per `deck_box_ingestion`'s "only full
  flat decks" rule, once resolved to `nocab_uuid`s.
- **`end.vps`**: the VP breakdown by source (`{"Province": 48,
  "Duchy": 24, "_vp_tokens_": 11, ...}`) — richer than the bare
  `score` total, since it shows *which* VP cards/effects the win came
  from.
- **`text_log`**: a path into isotropic's full per-game log archive
  (`YYYYMM/DD/game-....html.gz`) — confirmed these files were never
  separately fetched by Wayback at scale (see Flavor B below); this
  field is a dead pointer for the vast majority of summary rows, not a
  usable join key today.

### Flavor B — daily "full log" tarballs (bare `YYYYMMDD.tar.bz2`)

One HTML file per **individual game**, the human-readable turn-by-turn
log (same rendering the dead Woodcutter/`dombot` tooling in
`dominion/todo.md` used to scrape from the live site). Two days
survived: **2010-10-11** (108 games) and **2013-03-15** (10,815
games — the last day of the March-2013 summary month, so this one day
has *both* a summary row and a full turn-by-turn log per game,
joinable by matching each summary row's `serial`/timestamp/player
names against the HTML filename's embedded date-time).

Confirmed shape (sampled two files directly): opening header block
(winner, final kingdom, per-player final score/VP breakdown, opening
buys, final hand — same information as Flavor A's `end.deck`/`vps` but
rendered as HTML/text rather than structured JSON), then a `<hr/>Game
log` section: one line per action, indented by whose turn it is
(`Dave plays 5 Coppers.` / `... trashing an Estate.` / `undertoe
reveals a Silver and trashes it.`), including reaction/attack
resolution detail (Saboteur reveals, Secret Chamber discards, etc.)
that Flavor A cannot express at all — Flavor A only has the *final*
deck, never the sequence of buys/plays/reactions that produced it.

**This is a genuinely different metric surface from Flavor A**: Flavor
A supports large-N statistical metrics cheaply (kingdom → outcome,
card → win rate) at true scale (~300K games combined); Flavor B
supports sequence/strategy metrics (turn-by-turn buy order, reaction
timing, engine-assembly pacing) but only for two single days
(108 + 10,815 games) and needs real HTML parsing (strip `<span
class=card-*>` wrappers, parse the turn-indentation convention) that
doesn't exist in this project yet — no `IsotropicGameLogExtractionStage`
or equivalent has been written. Recommend treating these as two
separate metric-building efforts, not one combined pass: build
Flavor A's metrics first (cheap, high-N, JSON-native), treat Flavor B
as a smaller follow-on once someone wants sequential/strategy signal
specifically.

`201010_11_all.tar.bz2` is a byte-identical duplicate of
`2010_20101011.tar.bz2` under isotropic's older URL shape (confirmed
by `downloader.py`'s own docstring) — not a third day of data.

## Known biases / known-unknowns

- **Wayback survivorship, not isotropic's real corpus.** The original
  estimate in `dominion/todo.md` was ~13GB of daily logs and ~1GB of
  monthly summaries across isotropic's whole lifetime; these 5 files
  are everything that happened to get crawled, not a designed sample —
  two arbitrary months (Dec 2010, Mar 2013) and two arbitrary days
  within them. Any month-over-month or era comparison drawn from just
  these two windows is anecdote, not trend.
- **No skill signal.** Unlike `markus`'s external mu≥1.9-filtered
  stats (`dominion/todo.md`), this source's `rank` field is a
  per-game finishing placement, not an ELO/skill rating — "won" here
  says nothing about the players' overall strength, so win-rate
  metrics from this source measure "won this specific game," not
  "is a strong player."
- **Resignations censor `end.deck`.** 16% of player-rows have no
  `end` block at all (`resigned: true`) — any metric keyed on final
  deck composition silently drops these players' contribution to that
  game, not just their outcome.
- **`vetoed` reflects two players' joint preference, not one.**
  Isotropic's veto mechanic lets either player ban a card before the
  kingdom is dealt; a vetoed card's absence can't be attributed to a
  specific one of the two players from this data alone.
- **Solo (`nplayers==1`) games are practice/bot games**, not real
  competitive outcomes — any player-vs-player metric should filter to
  `len(players) >= 2`, and even 2-4p games mix `automatch` (paired by
  the matchmaker) with manually-arranged tables, a real confound for
  anything trying to measure "skill" via win rate.
- **`board.constraints`/`required`/`prohibited`/etc. mean the kingdom
  wasn't uniformly random.** A metric like "P(card X in kingdom)"
  computed over the raw corpus is biased by isotropic's own
  kingdom-generator settings, not just by vanilla Dominion's equal-
  probability card pool — worth filtering to unconstrained games (the
  ~majority with none of these `board` flags set) for any metric that
  wants a "natural" baseline.

---

## Single card (10)

1. **Win Rate When In Final Deck** — Input: single-card. Label:
   probability. `P(rank == 1 | card in end.deck)` — the isotropic
   analogue of 17lands' "Win Rate When In Deck," at real scale
   (~300K games). Accumulator.
2. **Kingdom Inclusion → Win Rate** — Input: single-card. Label:
   probability. `P(rank == 1 | card in board.supply, game has >= 2
   players)` — distinct from #1: this asks whether a card's mere
   *availability* correlates with who wins the game overall (a weak,
   confounded signal, since it doesn't require the winner to have
   bought the card), vs. #1's "did the winner actually buy it."
   Accumulator.
3. **Veto Rate** — Input: single-card. Label: probability.
   `P(card in vetoed | card in board.supply)` — which kingdom cards
   players actively ban given the chance, a real preference signal
   `dominiontabs` alone (no gameplay data) could never produce.
   Accumulator.
4. **Copies-Bought Distribution** — Input: single-card. Label: count
   (distribution). `end.deck[card]`'s observed value distribution
   across all games containing that card — e.g. Copper's is bimodal
   (7 untouched vs. heavily trashed), engine pieces cluster around
   small counts, terminal draw clusters differently — a per-card
   "how many copies does a good deck actually run" prior.
5. **VP-Source Contribution Rate** — Input: single-card
   (Victory-typed or VP-token-granting only, e.g. Gardens, Duke,
   Vineyard). Label: regression-continuous. Average
   `end.vps[card]` conditioned on `card in end.deck` — how much of a
   winning deck's score a given alt-VP card typically accounts for,
   vs. its face-value point count (e.g. Gardens' realized VP depends
   on deck size, not a fixed number).
6. **Resignation-Adjacent Rate** — Input: single-card. Label:
   probability. `P(opponent resigned | card in my end.deck or
   board.supply)` — kingdoms/decks associated with lopsided-enough
   games that the losing side gave up early; a rough "how oppressive
   is this card/kingdom" proxy distinct from a normal win-rate metric.
7. **Turn-Count Association** — Input: single-card. Label:
   signed float (regression-continuous). Average `turns` (winner's)
   over games where `card in board.supply`, minus the corpus-wide
   average — surfaces game-shortening cards (e.g. cursers, alt-VP
   rushes) vs. game-lengthening ones (e.g. trashers, engines), the
   Dominion analogue of the 17lands "Game Length Association" idea.
   Accumulator (needs a first-pass corpus-wide baseline).
8. **Mode Association** — Input: single-card. Label: probability.
   `P(mode == "text" | card in end.deck)` — a weak, mostly-curiosity
   signal (does a card's presence correlate with the isotropic
   text-client vs. images-client userbase, two populations that may
   differ in experience level) — flagged as speculative/low-priority
   but free to compute alongside the others.
9. **Automatch vs. Arranged-Table Rate** — Input: single-card. Label:
   probability. `P(automatch | card in board.supply)` — whether a
   card/kingdom shows up disproportionately in matchmaker-paired games
   vs. manually-arranged ones (e.g. league/tournament play might favor
   specific curated kingdons over isotropic's default random deal).
10. **Solo/Practice Rate** — Input: single-card. Label: probability.
    `P(len(players) == 1 | card in board.supply)` — whether certain
    kingdoms show up more in single-player practice games (e.g. newer
    or unusual cards a player wants to test alone before playing them
    against someone).

## Multi card (10)

1. **Kingdom → Winning Deck Prediction** — Input: multi-card (the
   10-11 card `board.supply` kingdom). Label: classification
   (variable-set, over the winner's `end.deck` card set intersected
   with the kingdom). Given only the kingdom, predict which kingdom
   cards the eventual winner actually bought — the deck-construction
   analogue of 17lands' "Deck Composition -> Win Prediction," but
   input and output are both drawn from the same small kingdom rather
   than a large card pool. Streaming (label already on the row, one
   winner per game).
2. **Full Deck -> Win Prediction** — Input: multi-card (one player's
   full `end.deck`, resolved to `nocab_uuid`s). Label: classification
   (fixed-set/binary: `rank == 1`). Streaming. The direct isotropic
   counterpart of the 17lands "Deck Composition -> Win Prediction"
   idea, at real scale, and one of the very few multi-card win-outcome
   metrics possible for Dominion anywhere in this project today.
3. **Kingdom -> Veto Prediction** — Input: multi-card (the full dealt
   kingdom before vetoes, i.e. `board.supply` union `vetoed`). Label:
   classification (variable-set, over the vetoed card(s)). Given a
   full undealt kingdom candidate, predict which card(s) get banned —
   a genuinely rich "what do real players avoid, in context of the
   rest of the options" signal `vetoed` alone (single-card idea #3)
   can't express, since veto choices are inherently comparative
   (players pick the *worst-relative-to-the-others* card, not an
   absolutely bad one). Streaming.
4. **Kingdom -> Game-Length Prediction** — Input: multi-card (the
   kingdom). Label: regression-continuous. Predict winner `turns`
   from the kingdom alone — the deck-level counterpart to single-card
   idea #7, letting a model learn from full kingdom composition
   (e.g. "this kingdom has no trashers and two curses" as a joint
   signal) rather than one card's marginal association. Streaming.
5. **Deck Pair -> Same-Game Winner Classification** — Input:
   multi-card (both decks from one 2-player game). Label:
   classification (fixed-set/binary: which deck won). Streaming. A
   genuine head-to-head framing (unlike idea #2, which scores one
   deck in isolation) — lets a model learn relative strength given
   both sides' actual purchases, the Dominion analogue of a
   draft-vs-draft matchup predictor.
6. **Kingdom -> VP-Breakdown Prediction** — Input: multi-card (the
   kingdom). Label: regression-continuous (multi-output: predicted
   share of winning score from Province/Duchy/alt-VP/`_vp_tokens_`).
   Given the kingdom, predict how the eventual winner's score will
   be composed — distinguishes "Province rush" kingdoms from
   "alt-VP engine" kingdoms at the composition level, not just total
   score. Streaming.
7. **Full Pool (Deck + Vetoed) -> Kingdom Reconstruction** — Input:
   multi-card (winner's `end.deck` plus that game's `vetoed` cards,
   as a combined candidate-kingdom pool). Label: classification
   (variable-set, over the remaining kingdom cards the winner never
   bought). Given what a player *did* buy plus what got banned before
   the game, predict the rest of the kingdom that existed but went
   unused — a "what did this player ignore, given what was on offer
   and what was pre-emptively removed" signal.
8. **Two-Kingdom Similarity -> Outcome-Distribution Comparison** —
   Input: multi-card (two different games' kingdoms, compared as two
   groups). Label: probability. `P(games have a similar winner
   turn-count / VP-breakdown profile | kingdom_1, kingdom_2 card-
   overlap)` — a coarse "do similar kingdons play out similarly"
   check, useful as a sanity baseline before trusting any
   kingdom-conditioned metric above to generalize across
   never-exactly-seen-before kingdoms (true of nearly every kingdom in
   a corpus this size, since 10-of-~300-card combinations rarely
   repeat exactly).
9. **Multiplayer (3-4p) Deck Set -> Placement-Order Prediction** —
   Input: multi-card (all 3-4 players' full decks from one game).
   Label: ranking. Given every deck in a multiplayer game, predict the
   full finishing order (`rank` 1 through N) — a genuine ranking-label
   metric (the only source of >2-player Dominion outcomes in this
   project), though flagged as a smaller slice of the corpus (~4.7%
   of games have 3+ players) and a real design question on how ties/
   resignations mid-ranking should be handled.
10. **Sequential Buy-Order Prediction (Flavor B only)** — Input:
    multi-card (a partial turn-by-turn buy sequence, parsed from a
    Flavor B HTML log, up to some turn N). Label: classification
    (variable-set, over the kingdom's remaining unbought cards).
    Predict what a player buys next given their buys so far — the one
    metric idea in this list that's structurally impossible from
    Flavor A alone (which only has final deck composition, never
    order-of-acquisition), and the main reason to eventually invest in
    parsing Flavor B's two surviving days rather than treating them as
    "more of the same data" as Flavor A.

---

## Open items for whichever pass implements these

- No `IsotropicSummaryExtractionStage`/`CardIngestionStage`-equivalent
  reader exists yet for either flavor — Flavor A needs a JSONL-per-
  tarball-per-day reader plus name -> `nocab_uuid` resolution via the
  existing `dominiontabs` `CardBinder`; confirm all `board.supply`/
  `end.deck`/`vetoed` card names actually resolve cleanly before
  building anything on top (spot-checked names above look like plain
  dominiontabs card names, but hasn't been checked at full-corpus
  scale, e.g. Colony/Platinum/Potion and Black Market's extra pile).
- Flavor B needs real HTML parsing (strip `card-*` span wrappers,
  interpret the turn-indentation convention) that doesn't exist
  anywhere in this project yet — a bigger lift than Flavor A and
  probably its own follow-on design pass, not bundled into whichever
  metric gets built first.
- Decide whether Flavor A's `end.deck` (a full, DeckBox-eligible deck)
  should also get routed through `deck_box_ingestion` as a proper
  clean `DeckBox` entry (per this skill's "Consider DeckBox Ingestion"
  step), separately from any metric-local `deck_box` a given metric
  builder maintains per this container's own conventions.
- The `board.constraints`/`required`/`prohibited`/etc. minority of
  games (curated/challenge kingdoms) should probably be excluded from
  any "natural kingdom" baseline metric (ideas single-card #2, #7,
  #9, #10 and multi-card #1, #4, #6) rather than silently pooled in —
  worth a concrete filter, not just a footnote, before those are
  built.
- 2013-03-15's Flavor A/Flavor B overlap (same day, both a summary row
  and a full log per game) is a real opportunity to build a combined
  metric (e.g. "given the full buy sequence, predict `end.vps`
  breakdown") once a join key between the two flavors is worked out —
  flagged as a nice-to-have, not required for either flavor's metrics
  to stand alone.

## Human decision on short list of metrics — BUILT (see summary/README.md)

### Single Card metrics

- **Veto Rate** → `summary/veto_rate_metric.py::VetoRateMetric`
- **Copies-Bought Distribution** → `summary/copies_bought_distribution_metric.py::CopiesBoughtDistributionMetric`
- **Turn-Count Association** → `summary/turn_count_association_metric.py::TurnCountAssociationMetric`
- New metric if possible: Given a card, guess the average number of times it was bought in the course of the game (popular cards vs unpopular cards as a regression effectively) → `summary/average_copies_bought_metric.py::AverageCopiesBoughtMetric`

### Multi Card Metrics

- **Full Deck -> Win Prediction** → `summary/full_deck_win_prediction_metric.py::FullDeckWinPredictionMetric`
- **Kingdom -> Veto Prediction** → `summary/kingdom_veto_prediction_metric.py::KingdomVetoPredictionMetric`
- **Kingdom -> Game-Length Prediction** → `summary/kingdom_game_length_metric.py::KingdomGameLengthMetric`

### Multi Group Metrics

- **Deck Pair -> Same-Game Winner Classification** → `summary/deck_pair_winner_metric.py::DeckPairWinnerMetric`
- **Multiplayer (3-4p) Deck Set -> Placement-Order Prediction** → `summary/multiplayer_placement_metric.py::MultiplayerPlacementMetric`

## New candidates raised during human review

Several of these need a **mid-game partial deck** — the set of cards a
player has bought up through some turn T, before the game ends. That
state only exists in Flavor B (turn-by-turn HTML logs, see above) and
would need real parsing of each `plays`/`buys`/`trashes` log line, not
just Flavor A's already-confirmed `card-*` span/turn-indentation
structure — no more of a lift than what Flavor B's other ideas already
require, but every metric below marked *(Flavor B only)* inherits that
same parser dependency, not a new one (that parser now exists -
`games/game_log_parser.py`).

Also newly confirmed while checking these: both sampled Flavor B files
have a `trash: ...` summary line in their header block (e.g. `trash: 8
<span class=card-victory>Provinces</span>` /
`trash: 5 Silvers, 4 Monuments, a Chapel, 2 Duchies, 2 Estates, 9
Coppers, a Saboteur, and a Province`) — a full end-of-game trash-pile
census that **Flavor A's JSON has no equivalent of at all** (`end.deck`
is what a player kept, never what left play). Any trash-related metric
below is Flavor-B-only for that reason, not just because of the
mid-game-state issue.

Single card:

 1. **Average Copies-Bought (regression)** — Input: single-card.
    Label: regression-continuous. `E[end.deck[card] | card in
    end.deck]` — the mean-count regression framing of the existing
    "Copies-Bought Distribution" idea (single-card #4 above), which
    reports the full distribution; this reports just its mean, a
    simpler "how popular is this card, in copies-per-game-it-appears-in
    terms" scalar per card.

Multi card / multi group:

 1. **Kingdom -> Winning-Deck Card Membership** — BUILT, see
    `summary/kingdom_member_label_metrics.py::WinningDeckMembershipMetric`.
    Input: multi-card
    (the kingdom). Label: classification (variable-set, one bool per
    kingdom card: is it in the winner's `end.deck` at all). Splits the
    existing "Kingdom -> Winning Deck Prediction" idea (multi-card #1
    above) into its simplest concrete framing — "in or out," no count.
 2. **Kingdom -> Winning-Deck Card Counts** — BUILT, see
    `summary/kingdom_member_label_metrics.py::WinningDeckCountMetric`.
    Input: multi-card (the
    kingdom). Label: regression-continuous, per kingdom card (one
    count prediction per card). The richer sibling of #12 above and of
    the existing multi-card #1 — instead of "is it in the deck," "how
    many copies," reusing the same kingdom input.
 3. **Mid-game Deck + Kingdom -> Next Buy Prediction** — BUILT, see
    `games/mid_game_next_buy_metric.py::NextBuyPredictionMetric`. *(Flavor B
    only)* — Input: multi-card (a partial deck reconstructed at turn
    T, plus that game's kingdom). Label: classification (variable-set,
    over cards still gainable — kingdom plus basics). `P(next card
    bought = X | partial deck at turn T, kingdom)`. The same idea as
    the existing "Sequential Buy-Order Prediction" (multi-card #10
    above), restated with the kingdom made an explicit second input
    rather than implied by the log alone — kept as its own numbered
    entry since the human review named it directly as a priority.
 4. **Mid-game Deck -> Next Trashed Card Prediction** — BUILT, see
    `games/mid_game_next_trashed_card_metric.py::NextTrashedCardMetric`.
    *(Flavor B only)* — Input: multi-card (partial deck at turn T). Label:
    classification (variable-set, over cards currently in that deck).
    `P(card_trashed = X | deck at turn T, a trash event occurs on this
    turn)` — conditioned on a trash happening at all (the log's own
    trash-action lines, e.g. "... trashing 3 Coppers," are the only
    source of *when* one occurs), not predicting whether one will.
 5. **Mid-game Deck -> Next-Turn Action Count Prediction** — BUILT, see
    `games/mid_game_next_turn_action_count_metric.py::NextTurnActionCountMetric`.
    *(Flavor B only)* — Input: multi-card (partial deck at turn T). Label: count
    (regression-continuous). `E[number of Action cards played next
    turn | partial deck at turn T]` — an engine-momentum signal (a
    deck full of untapped Villages/Labs should predict a bigger next
    turn than a thin money deck).
 6. **Mid-game Deck -> Eventual Win Probability** — BUILT, see
    `games/mid_game_win_probability_metric.py::EventualWinProbabilityMetric`.
    *(Flavor B only)* —
    Input: multi-card (one player's partial deck at turn T, plus the
    kingdom). Label: probability. `P(this player eventually wins |
    partial deck at turn T, kingdom)` — a real-time "win probability
    curve" metric, buildable at many turn-T checkpoints per game from
    the same log.
 7. **Mid-game Deck Pair -> Winner Prediction** — BUILT, see
    `games/mid_game_deck_pair_winner_metric.py::MidGameDeckPairWinnerMetric`.
    *(Flavor B only)* —
    Input: multi-group (both players' partial decks at turn T in one
    game). Label: classification (fixed-set/binary). The mid-game
    counterpart of "Deck Pair -> Same-Game Winner Classification"
    above (multi-card #5 / this file's "Multi Group Metrics" shortlist)
    — same question, asked before the game ends rather than after.
 8. **Deck Card-Type Masking** — BUILT, see
    `summary/deck_card_mask_metric.py::WinningDeckMaskedCardMetric`
    (masks uniformly at random among the winner's non-basic cards -
    see that file's own module docstring for why, over the "most
    copies" / "fixed card type" alternatives this entry once left
    open). Input: multi-card (a final `end.deck`
    with one card removed). Label: classification (variable-set, over
    the kingdom plus basics). Given the rest of a finished deck, guess
    the single card taken out — this project's existing
    `generic/deck_card_mask_metric.py::DeckCardMaskMetric` Template
    Method is built for exactly this shape.
 9. **Kingdom -> Trashed-Card Membership** — Input: multi-card (the
    kingdom). Label: classification (variable-set, one bool per
    kingdom card: did any copy of it end up in either player's
    `trash:` summary by game's end). Uses Flavor B's `trash:` header
    line (see above) — Flavor-A-impossible, since Flavor A never
    records what got trashed, only what a player kept.
10. **Kingdom -> Trashed-Card Counts** — Input: multi-card (the
    kingdom). Label: regression-continuous, per kingdom card. The
    count-valued sibling of #20, same Flavor-B-only `trash:` line
    dependency.
11. **Deck Card-Set -> Per-Card Copy-Count Prediction** — BUILT, see
    `summary/deck_card_set_copy_count_metric.py::DeckCardSetCopyCountMetric`.
    Input:
    multi-card (the *set* of distinct cards present in a deck, counts
    stripped) plus the kingdom. Label: regression-continuous, per
    card in that set. Given only which cards made the final deck (not
    how many of each), predict each one's actual count — a
    context-conditioned version of single-card idea #4/#11 above,
    where the rest of the deck's composition is available as a signal
    rather than marginalizing over the whole corpus.

## More Flavor B-only candidates (turn-log-mined, confirmed against a live 60-game sample)

Checked a random 60-game sample from `2013_20130315.tar.bz2` directly
to confirm these are real, not just plausible-sounding: every sampled
game's header has an `opening: X / Y` line (60/60); 18% (11/60) contain
a real resignation event, with an explicit resigning nick, their
placement, and turn count (`<b>#2 jwpepa</b>: resigned (1st); 29
turns` / `jwpepa resigns from the game.`); the pile-exhaustion sentence
names the *specific* 1-3 piles that ran out (e.g. "Silk Roads,
Estates, and Hamlets are all gone" vs. the simpler "All Provinces are
gone"), confirmed to vary game-to-game rather than always being
Provinces. None of this exists in Flavor A at all — it has no opening
buys, no resignation detail beyond a single `resigned: true` flag with
no turn number, and no pile-exhaustion attribution whatsoever.

Single card:

 1. **Opening-Buy Rate** — BUILT, see
    `games/opening_buy_rate_metric.py::OpeningBuyRateMetric` (tallies
    every player's opening, not just the winner's — see that file's
    own module docstring for why). Input: single-card. Label: probability.
    `P(card in {opening 1st buy, opening 2nd buy} | card in
    board.supply)` — which kingdom cards get bought turn one at all,
    the cheapest and most direct "how good is this card early" signal
    this source offers, and only extractable from Flavor B's `opening:
    X / Y` line.
 2. **Pile-Exhaustion Rate** — BUILT, see
    `games/pile_exhaustion_rate_metric.py::PileExhaustionRateMetric`.
    Input: single-card. Label: probability.
    `P(card's pile is named as exhausted at game end | card in
    board.supply)` — how often a given kingdom card actually runs out
    before the game ends (Province piles run out constantly; a
    12-cost Colony-set card almost never does) — a direct, per-card
    "how contested is this pile" signal, distinct from anything Flavor
    A offers (which has no notion of remaining pile counts at all).
 3. **Resignation-Trigger Rate** — Input: single-card. Label:
    probability. `P(a losing player resigns before game end | card in
    board.supply)` — which kingdoms produce games felt "over" early
    enough that a player quits rather than plays it out (e.g. an
    early, unanswered Witch/Curse-heavy kingdom vs. a slow
    money/engine grind) — a real "how oppressive/unfun does this
    kingdom feel" proxy no other metric in this file captures.

Multi card / multi group:

 1. **Kingdom -> Opening-Buy Pair Prediction** — BUILT, see
    `games/kingdom_opening_buy_prediction_metric.py::KingdomOpeningBuyPredictionMetric`.
    Input: multi-card (the
    kingdom). Label: classification (variable-set, over unordered
    pairs of {kingdom cards, Silver, nothing}). Given only the
    kingdom, predict the (eventual winner's) opening `X / Y` buy —
    this project's Dominion analogue of a "best opening" strategy
    guide, learnable directly instead of hand-authored.
 2. **Opening Buy -> Outcome Prediction** — BUILT, see
    `games/opening_buy_outcome_metric.py::OpeningBuyOutcomeMetric`
    (one row per PLAYER per game, win and loss both, not winner-only —
    see that file's own module docstring for why). Input: multi-card (just
    the 1-2 opening-buy cards, as their own small group) plus the
    kingdom. Label: regression-continuous (`turns` to win) or
    probability (`rank == 1`) — the inverse of #26: given only what a
    player opened with, how much of the eventual outcome is already
    determined. A natural pairing with #26 to see how much of a
    kingdom's optimal-opening signal actually predicts winning, vs.
    just being a played convention.
 3. **Kingdom -> Game-Ending Pile Prediction** — BUILT, see
    `games/kingdom_ending_pile_prediction_metric.py::KingdomEndingPilePredictionMetric`.
    Input: multi-card (the
    kingdom). Label: classification (variable-set, over kingdom cards
    plus Province — one row per game, potentially 1-3 true labels for
    a 3-pile ending). `P(card's pile is one of the piles named as
    exhausted | kingdom)` — the deck-level counterpart of single-card
    idea #24, letting a model learn "this kingdom's cheap-engine-piece
    combination tends to 3-pile before Provinces run out" as a joint
    pattern rather than one card's marginal rate.
 4. **Kingdom -> Province-vs-3-Pile Ending Classification** — BUILT,
    see `games/kingdom_ending_type_metric.py::KingdomEndingTypeMetric`
    (labels: `province`/`colony`/`multi_pile` — "multi_pile" rather
    than "3-pile" since Dominion's own rule is "3 or more," never
    hardcoded to exactly 3). Input:
    multi-card (the kingdom). Label: classification (fixed-set:
    Province-exhaustion ending vs. 3-pile ending — Colony-exhaustion
    as a third class when Colony is in the supply). A coarser, cheaper
    sibling of #28 — just the ending *type*, not which specific piles.
 5. **Kingdom -> Resignation Likelihood Prediction** — Input:
    multi-card (the kingdom). Label: probability. The deck-level
    counterpart of single-card idea #25 — `P(any player resigns before
    game end | kingdom)` — whether specific kingdom *combinations*
    (not just individual oppressive cards) correlate with players
    giving up early.
 6. **Turn-of-Resignation Prediction** — Input: multi-group (the
    resigning player's partial deck at the turn they quit, plus the
    kingdom). Label: count (regression-continuous: turn number).
    Among games that do end in a resignation, predict how long the
    losing player held on before quitting — a "how much punishment
    will a losing player tolerate against this kingdom" signal,
    distinct from #25/#30's "does it happen at all."
