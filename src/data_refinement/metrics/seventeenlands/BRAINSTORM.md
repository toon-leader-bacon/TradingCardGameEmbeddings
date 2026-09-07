# METRIC_BRAINSTORM

A todo/idea list of candidate metrics for each of the three 17lands raw
data sources, produced by reviewing `data/tmp/{draft_data,game_data,
replay_data}/*_header.csv`/`*_rows.csv` (small extracted samples — the
full raw CSVs are multi-GB and were never loaded directly) plus a
`pandas`-based scan of the sample rows for actual column semantics.
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec. Every metric names its:

- **Card shape** — single card / multi card / multi group, per
  `README.md`'s taxonomy.
- **Label** — what a dojo would eventually try to predict from this
  metric's output.
- **Builder shape** — accumulation or streaming, per `README.md`.

Metrics already implemented (`AveragePickNumberMetric`,
`PickedVHeldVPackMetric`) are not repeated here.

## What the raw data actually looks like (found via this review)

- **Draft data** (`draft_data/MSH.PremierDraft.csv`, ~4.3GB): one row
  per pick. Non-card columns: `expansion, event_type, draft_id,
  draft_time, rank, event_match_wins, event_match_losses, pack_number,
  pick_number, pick, pick_2, pick_maindeck_rate,
  pick_sideboard_in_rate, user_n_games_bucket,
  user_game_win_rate_bucket`. `pick_maindeck_rate`/
  `pick_sideboard_in_rate` are themselves 17lands-precomputed
  aggregates already sitting on every row (not something we'd need to
  accumulate ourselves if reusing them directly). `pack_card_<name>`/
  `pool_<name>` are per-row copy counts, confirmed in this session's
  cleanup of `PickedVHeldVPackMetric` (see git history — the column
  header carries the name, the cell carries the count).
- **Game data** (`game_data/game_data_public.MSH.PremierDraft(1).csv`,
  ~1.3GB): one row per game. Non-card columns: `expansion, event_type,
  draft_id, draft_time, game_time, build_index, match_number,
  game_number, rank, opp_rank, main_colors, splash_colors, on_play,
  num_mulligans, opp_num_mulligans, opp_colors, num_turns, won,
  user_n_games_bucket, user_game_win_rate_bucket`. Per-card columns:
  `opening_hand_<name>, drawn_<name>, tutored_<name>, deck_<name>,
  sideboard_<name>` — each a per-row count.
- **Replay data** (`replay_data/MSH.PremierDraft.csv`, ~2.7GB): one
  row per game, far richer. Shares game data's header fields (minus
  `main_colors`/`splash_colors`/`opp_rank` naming quirks, see actual
  header) plus:
  - `candidate_hand_1`..`candidate_hand_7` + `opening_hand`:
    pipe-delimited Arena ID strings, one per London-mulligan hand seen
    (`candidate_hand_1` = first 7 cards seen, `candidate_hand_2` = the
    hand after a mulligan, etc.; `opening_hand` = the hand actually
    kept). Confirmed via sample rows — e.g. `candidate_hand_1 ==
    opening_hand` when there's no mulligan.
  - `deck_<name>`: full deck list, same per-row-count shape as game
    data's `deck_<name>`.
  - `user_total_*`/`oppo_total_*`: whole-game aggregate counts
    (`cards_drawn, cards_tutored, cards_discarded, lands_played,
    cards_foretold, creatures_cast, non_creatures_cast,
    instants_sorceries_cast, cards_learned, mana_spent`).
  - `{user,oppo}_turn_{1..30}_*`: ~30 columns per (player, turn)
    combination. Confirmed by sampling: many of these are **Arena ID
    lists** (pipe-delimited when multiple), not counts —
    `user_turn_1_lands_played`, `..._cards_drawn`,
    `..._creatures_cast`, `..._user_abilities`, etc. all held Arena
    IDs in the sample. `eot_user_cards_in_hand` is a pipe-delimited
    Arena ID list (own hand, private-but-visible-to-self); the
    opponent-mirror `eot_oppo_cards_in_hand` is a bare count (public
    hand *size*, not identity) — this is the exact public/private
    split `README.md` already documents. `eot_*_life` and
    `*_combat_damage_taken` are plain numbers.
  - This session did **not** verify every one of the ~30 per-turn
    columns' exact semantics (e.g. whether `creatures_attacked` is an
    Arena ID list of the attacking creature(s) or something else) —
    only spot-checked a handful with all-`NaN`/single-value sample
    rows, so multi-value columns weren't directly observed. Treat
    column-by-column resolution mechanics as still-open per metric
    below, the same way `plans/replay_data_metrics.md` already flagged
    for its own six placeholder metrics.

**Open dependency flagged, not resolved:** several ideas below need a
card's **color identity** (e.g. "is this a splash card," "what colors
has this pool committed to"). `GenericCard.raw_content` is untyped
per-game-ingestion-stage data (`src/schema/card.py`) — whether color is
already a normalized, reliably-present key in MTG's `raw_content` shape
is unverified in this session. Any metric below that depends on color
identity is marked, and would need that checked before design.

---

## Draft data (10)

1. **Card Take Rate** — P(card is picked | card is present in the
   pack that row). Single card. Label: float in [0, 1]. Accumulator
   (needs a running `times_in_pack` and `times_picked` count per
   card, keyed off `pack_card_<name> > 0` and `pick == <name>`).

2. **Average Last Seen At (ALSA)** — average `pick_number` of the
   *last* row within a `(draft_id, pack_number)` group where a card
   still shows up in `pack_card_<name>` before it's gone (taken by
   anyone at the table, not necessarily this drafter). Single card.
   Label: float pick-number. Accumulator, but a harder one than
   `AveragePickNumberMetric`'s shape — needs per-`(draft_id,
   pack_number)` state grouped and ordered by `pick_number`, which may
   span chunk boundaries in `CsvScanner`'s streaming reads (rows
   aren't guaranteed to arrive in a chunk-local, draft-complete order
   unless the raw CSV is itself sorted by `draft_id`). Worth checking
   the raw CSV's actual row order before committing to this design.

3. **Wheel Rate** — P(a card seen in the pack at `pick_number < N` is
   still in the pack `table_size` picks later, i.e. it "wheeled" back
   around the table). Single card. Label: float in [0, 1].
   Accumulator, same cross-row `(draft_id, pack_number)` grouping
   complexity as ALSA, plus an assumption about table size (8-person
   pods?) that'd need confirming against the data, not guessed.
   This will require figuring out a good way to chunk the data by full draft
   story (or at least by full pack) so we can do cross row analysis more easily.

4. **First-Pick Rate (P1P1 Take Rate)** — P(card is the pick,
   specifically at `pack_number == 0 and pick_number == 0`), separate
   from the card's overall average pick number — isolates
   "bomb"/first-pick-worthy signal from general pick speed. Single
   card. Label: float in [0, 1]. Accumulator.

5. **Hate-Pick / Overdraft Signal** — for cards with a low
   `pick_maindeck_rate` on the picked row (i.e. historically rarely
   played), how early (`pick_number`) are they still taken? A high
   value suggests the card is picked to deny opponents rather than to
   play. Single card. Label: float (average pick_number, filtered to
   low-maindeck-rate picks). Accumulator, reuses the
   `pick_maindeck_rate` column already on each row rather than
   re-deriving it.

6. **Pool Color-Commitment Prediction** — given the current pool
   (`pool_<name>` columns at pick time), predict the color pair the
   drafter ultimately settles into. Multi card (pool group). Label:
   a color-pair class (needs card color identity — **depends on the
   open color-identity question above**, and on `main_colors` which
   only exists in game_data, not draft_data — this would be a
   cross-dataset join by `draft_id`, a new kind of dependency this
   pipeline doesn't have yet). Streaming in spirit, though the label
   source lives in a different raw file.

7. **Rank-Stratified Average Pick Number** — same accumulator shape as
   the existing `AveragePickNumberMetric`, but keeping separate
   running sums per `rank` bucket (or `user_game_win_rate_bucket`),
   surfacing whether strong drafters value a card earlier/later than
   average. Single card (with an extra stratifying dimension, not a
   second card-group). Label: float pick-number, per rank bucket.
   Accumulator.

8. **Color Openness / Pack Depletion Signal** — as `pick_number`
   increases within one `(draft_id, pack_number)`, track how many
   remaining `pack_card_<name>` cards belong to each color, to
   estimate which colors are "open" at the table. **Doesn't fit the
   single/multi/multi-group taxonomy cleanly** — the unit here is a
   color, not a card or a named card-group — flagging as a genuine
   outlier worth a design conversation before forcing it into the
   existing shape. Depends on color identity (open question above).
   Accumulator either way (needs per-pack running color tallies).

9. **Draft Color-Trajectory Consistency** — across a full draft, how
   much does the pool's inferred color distribution shift
   pick-to-pick (e.g. entropy of pool colors over the sequence of
   picks)? Multi card (pool group, evolving over the draft). Label:
   a scalar "consistency"/entropy-trend score per draft. Accumulator,
   same `(draft_id)`-grouped cross-row complexity as #2/#3, plus the
   color-identity dependency.

10. **Speculative Splash Signal** — rate at which a card that's later
    revealed as an off-color "splash" pick (per game_data's
    `splash_colors`, again a cross-dataset join by `draft_id`) was
    taken unusually late/early relative to its normal pick number.
    Single card. Label: float pick-number delta. Accumulator. Flagged
    as speculative/lowest-priority of the ten — leans on both the
    color-identity dependency and a cross-dataset join.

---

## Game data

1. **Win Rate When In Deck** — P(win | card present in `deck_<name>`
   for that game). Single card. Label: float in [0, 1]. Accumulator.

2. **Opening Hand Win Rate** — P(win | card present in
   `opening_hand_<name>`). Single card. Label: float in [0, 1].
   Accumulator.

3. **Drawn Win Rate** — P(win | card present in `drawn_<name>`,
   i.e. seen at some point in the game, opening hand or not). Single
   card. Label: float in [0, 1]. Accumulator.

4. **On-Play vs. On-Draw Win Rate Delta** — for each card, win rate
   in games where `on_play == True` minus win rate where
   `on_play == False` (both conditioned on the card being in the
   deck). Single card. Label: a signed float (win-rate delta) — a
   tempo/curve proxy: cards more play/draw-sensitive show a bigger
   gap. Accumulator (two running win-rate accumulators per card, one
   per `on_play` value).

5. **Game Length Association** — average `num_turns` for games where
   a card is in `deck_<name>`, vs. the format's overall average —
   surfaces cards correlated with faster (aggro) or slower (control)
   games. Single card. Label: float turn-count delta. Accumulator.

6. **Main colors** — Predict the main colors of a deck that contain this card (single card), or of the entire deck (multi-card)

7. **Splash Colors** - Predict the splash and main colors of an entire deck (multi-card)

8. **Tutor Target Rate (game-level)** — P(card appears in
   `tutored_<name>` | card is in `deck_<name>`) — a lighter-weight,
   game_data-only cousin of the replay-data tutor metric below (no
   per-turn resolution needed, just a per-game count). Single card.
   Label: float in [0, 1]. Accumulator.

9. **Predict tutor targets** - Given a deck (multi-card input) predict which cards are tutored.

10. **Deck Composition → Win Prediction** — given the full
    `deck_<name>` list for one game (as a set of nocab_uuids, one row
    of raw data), predict `won`. Multi card (the whole deck as one
    group, no sub-grouping — the ungrouped multi-card shape
    `plans/deck_outcome_dojo.md` already designed around, previously
    served by the now-deleted `DeckOutcomeScanner`). Label: bool.
    Streaming — each row already carries a complete
    `(deck_card_uuids, won)` example on its own, no cross-row state
    needed.

---

## Replay data (10)

1. **Average Turn Cast** — for a card resolved out of the
   `{user}_turn_{N}_creatures_cast`/`non_creatures_cast` Arena-ID
   columns, the average turn number `N` at which it's cast (across
   every game where it was cast at all). Single card. Label: float
   turn-number. Accumulator — needs scanning all ~30 turns' cast
   columns per row, resolving each Arena ID found to a card via
   `CardBinder.get_by_alias`, matching the resolution mechanism
   `plans/replay_data_metrics.md` already confirmed exists.

2. **Cast Rate** — P(card is cast at all in a game | card is in
   `deck_<name>`) — distinct from game_data's "drawn" rate, since
   replay data can distinguish "drawn but never cast" (e.g. dead in
   hand, mana-screwed, held back) from "cast." Single card. Label:
   float in [0, 1]. Accumulator.

3. **Average Turns Held In Hand Before Cast** — for a card seen in
   `eot_user_cards_in_hand` on turn *k* and then cast on turn *k+n*,
   average *n* across games. Single card. Label: float turn-count.
   Accumulator, and the hardest of the "cast timing" family — needs
   tracking, per card per game, the first turn it enters hand (drawn
   or opening hand) and the turn it's cast, i.e. real per-game
   temporal state, not just a running sum/count. Worth a
   `design-recipe-skeleton` pass on its own before committing to a
   specific state shape.

4. **Combat Kill Involvement Rate** — P(a creature card is involved
   in a combat kill — appears in a turn's
   `user_creatures_killed_combat`/`oppo_creatures_killed_combat`
   Arena-ID columns — in a game where it was cast). Single card.
   Label: float in [0, 1]. Accumulator. Mirrors the deleted
   `combat_kill_involvement_rate.py`'s intent, redesigned fresh rather
   than restored as-is.

5. **Combat damage involvemnt rate** P(a creature card is involved in combat &
   combat damage was dealt to the targeted player). Can't fully determine if
   this particular creature is the one that dealt damage, but if it was involved
   in a play that lead to player damage or not.

6. **Average number of turns the card is on the board at EoT** How long does this
   card stay in it's particular zone (before going to the discard). Simply a total
   number count of turns that creatures appears on the board is fine. Powerful cards
   tend to end the game quickly, or are countered right asap while weaker cards
   are allowed to linger.

7. **How long after cast until the game ends** What turn was this card first
   played in this game, subtracted from the total number of turns in the game.
   Weak early cards will have a higher number (it didn't have a huge impact on
   the game finishing by itself) while powerful cards are more expensive, cast
   later and tend to finish the game quickly.

8. **Discard Rate** — P(card appears in any turn's
   `cards_discarded` column | card is in `deck_<name>`). Single card.
   Label: float in [0, 1]. Accumulator.

9. **Tutor Target Rate (replay-level)** — same idea as game data's
   metrics, but resolved per-turn from `cards_tutored` columns instead of
   a single whole-game count — lets a future dojo also learn *when*
   in the game a card tends to get tutored for, not just whether.
   Single card. Label: float in [0, 1] (rate) — could extend to also
   emit the turn-number distribution, a design choice for later.
   Accumulator.

10. **Mulligan-Correlated Candidate Hand Composition** — compare card
   frequency across `candidate_hand_1` (always seen) vs.
   `candidate_hand_2`..`_7` (only reached after a mulligan) — cards
   over-represented in hands that got mulliganed away are a weak
   "hard to keep" signal, richer than game_data's mulligan-count-only
   version since replay data shows the actual rejected
   hand contents. Single card. Label: float (relative frequency
   delta between kept vs. rejected candidate hands). Accumulator.

11. **Likelyhood to attach solo** — for a creature card resolved out of
   `creatures_attacked` on a turn, are they attacking alone or in a group?
   Single card input, percentage label output; Accumulator

12. **Attack Blocked Rate** — P(a creature's attack is blocked |
   it attacked) — resolved from `creatures_attacked` vs.
   `creatures_blocked`/`creatures_unblocked` per turn. Single card.
   Label: float in [0, 1] — an evasion/threat-level proxy. Accumulator.

13. **Mid-Game Board State → Win Prediction** — freeze the game state
    at a fixed turn *N* (e.g. turn 4's `eot_*` columns: lands/
    creatures/non-creatures in play, life totals, hand size) for both
    players, and predict `won`. Multi group — two groups, "user's
    public board state" and "opponent's public board state," each
    itself a set of resolvable Arena IDs (own side) or bare counts
    (opponent's hidden-identity cards, per the public/private split
    documented in `README.md`) — the richest and most novel of the
    replay-data ideas, genuinely exercising the multi-group shape
    `plans/multi_card_dojos_discussion.md` designed for. Label: bool.
    Streaming in the sense that one row still yields one example, but
    picking *which* turn *N* to freeze at (fixed vs. variable per
    game) is an open design question, and asymmetric hidden/public
    information between the two groups is a new wrinkle none of the
    other 29 ideas here have to deal with.
