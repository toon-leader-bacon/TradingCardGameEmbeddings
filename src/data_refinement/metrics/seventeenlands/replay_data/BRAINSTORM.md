# METRIC_BRAINSTORM — 17lands replay_data

Second-pass brainstorm for `data/raw/17lands/replay_data/*.csv`, written
against the current `metric_writing` skill and superseding the
"Replay data" section of the old shared
`../BRAINSTORM.md` (kept there only as historical record — this file
is the live one for this source). Verified directly against
`MSH.PremierDraft.csv` (2,615 columns, 1,937 of them non-card-name)
via small streaming scans (`csv` module + targeted column checks, not
a full load — the file is ~2.7GB).

Every idea names: **Card shape** (single card / multi card / multi
group), **Label** (count / probability / classification-fixed /
classification-variable / regression-continuous / CLIP-style
contrastive / ranking), **Builder shape** (accumulator / streaming),
and a one-sentence description with formal `P(...)` notation wherever
probability-flavored, conditioning made explicit.

## Card resolution

- Arena-ID-list columns (`creatures_cast`, `creatures_attacked`,
  `eot_*_creatures_in_play`, `candidate_hand_*`, etc.): resolve each
  pipe-delimited id via `CardBinder.get_by_alias(GameId.MTG,
  DataSource.ARENA, arena_id)` — `arena_id` is registered as an alias
  at Scryfall ingestion time
  (`card_binder/scryfall/ingestion_stage.py`'s
  `_SINGLE_VALUE_ALIAS_FIELDS`).
- Name-keyed columns (`deck_<name>`, `sideboard_<name>`): resolve via
  `CardBinder.get_by_name_single(GameId.MTG, name)`.
- Color identity, where an idea needs it: `raw_content["color_identity"]`
  / `raw_content["colors"]` on the resolved card (Scryfall's raw JSON
  carries both) — the color-identity dependency the old brainstorm
  flagged as unverified is resolved; no further check needed before
  designing a color-dependent metric.

## Resolved per-turn column semantics (previously unverified)

The old brainstorm explicitly left these open. Verified this session
by finding populated (non-NaN) example rows and cross-referencing
multiple columns on the *same* row (not just independently-found
examples — see reasoning below each). All of this lives under the
`{actor}_turn_{N}_*` column family, where `actor` is `user` or `oppo`
and denotes *whose turn it was* (the active player that turn), turns
1-30.

- **`creatures_attacked`** — the *active player's* (`actor`'s) own
  creatures that declared an attack that turn. Single field, no
  user_/oppo_ split needed, since only the active player attacks.
- **`creatures_blocked`** — the subset of `creatures_attacked` that
  were blocked (still the *attacker's* creatures — the ones that got
  blocked, not the blockers).
- **`creatures_unblocked`** — the complementary subset of
  `creatures_attacked` that went through unblocked.
- **`creatures_blocking`** — the *defending* (non-active) player's
  creatures that declared as blockers. No split needed, since only
  the defender blocks.
- **`user_creatures_killed_combat`** / **`oppo_creatures_killed_combat`**
  — ABSOLUTE, not actor-relative: which side's creature(s) died to
  combat damage that turn, regardless of whose turn it is. Verified
  directly on one row (`user_turn_3`: `creatures_attacked=105091`,
  `creatures_blocked=105091`, `creatures_blocking=104894`,
  `user_creatures_killed_combat=105091`,
  `oppo_creatures_killed_combat=''`,
  `eot_user_creatures_in_play=''`, `eot_oppo_creatures_in_play=104894`)
  — the attacker (105091, user's) was blocked by 104894 (oppo's) and
  died; the board-state snapshot at end of turn confirms it (user has
  0 creatures left, oppo's blocker survived in play). This is the
  single clearest confirming row and resolves the whole family at
  once.
- **`user_creatures_killed_non_combat`** / **`oppo_creatures_killed_non_combat`**
  — same absolute-side convention, for non-combat removal
  (spells/abilities/sacrifice) rather than combat damage.
- **`user_instants_sorceries_cast`** / **`oppo_instants_sorceries_cast`**,
  **`user_abilities`** / **`oppo_abilities`**, **`user_mana_spent`** /
  **`oppo_mana_spent`** — all absolute-side pairs present under *every*
  turn prefix (both `user_turn_N_*` and `oppo_turn_N_*`), because
  either side can cast an instant, activate an ability, or spend mana
  in response during any turn, not just the active player.
- **`cards_discarded`** / **`cards_tutored`** — actor-relative, single
  field per turn prefix (best-effort reading, not triple-confirmed on
  one row the way combat was): the active player's own discards/tutors
  that turn. `cards_tutored` (and `cards_drawn`) only exist under
  `user_turn_N_*`, never `oppo_turn_N_*` — 17lands can't see the
  opponent's tutor targets (drawing from a hidden library), consistent
  with the hand-privacy split below. `cards_discarded` exists under
  both `user_turn_N_*` and `oppo_turn_N_*` since a discard is public
  once it happens.
- **`eot_{side}_lands_in_play` / `eot_{side}_creatures_in_play` /
  `eot_{side}_non_creatures_in_play`** — **both sides are Arena-ID
  lists, both fully resolvable.** This corrects an assumption in the
  old brainstorm (idea #13 there guessed the opponent's board would be
  a hidden bare count, mirroring hand privacy) — verified directly:
  `eot_oppo_lands_in_play` came back as pipe-delimited Arena ids
  (e.g. `'75022|75023'`) in every populated row sampled. This matches
  MTG's actual rules: the battlefield is public to both players: only
  the *hand* is a hidden zone.
- **`eot_user_cards_in_hand`** (own hand, Arena-ID list, private-to-self
  but visible to us since it's the drafter's own hand) vs.
  **`eot_oppo_cards_in_hand`** (bare count, e.g. `'6.0'` — opponent's
  hand size only, identity hidden) — this is the one genuinely
  asymmetric public/private pair, exactly matching the top-level
  `README.md`'s existing note.
- **`eot_{side}_life`** — plain floats, both sides, as expected.

## Important cross-cutting limitation (applies to most single-card ideas below)

Only the drafter's own deck is ever fully known (`deck_<name>` columns
are 17lands-drafter-perspective only — there is no `oppo_deck_<name>`).
So any metric conditioned on "card is in \<player\>'s deck" (the
denominator for a rate) can only be computed from the *user* side's
occurrences. Opponent cards *do* surface with resolvable identity in
combat/board-state columns (an opponent creature that attacks, blocks,
or sits on the battlefield reveals its Arena id), so ideas that key
off "card was observed doing X" rather than "card in known-deck" can
still include opponent-side samples — but any idea needing "sample
size out of full deck" as its denominator is implicitly user-deck-only
data, a real (if the same as game_data's existing) sampling bias, not
a full replay-data population.

---

## Single card (13)

1. **Average Turn Cast** — for a card resolved out of
   `{user,oppo}_turn_{1..30}_creatures_cast`/`non_creatures_cast`, the
   average turn number it's cast on, across every game it's cast in at
   all (both sides' occurrences valid here — no deck-membership
   conditioning needed). Label: float turn-number. Accumulator.

2. **Cast Rate** — `P(card cast in a game | card in deck_<name>)` —
   distinct from game_data's drawn rate since replay data can tell
   "drawn but never cast" from "cast." User-deck-only (see limitation
   above). Label: float in [0, 1]. Accumulator.

3. **Average Turns Held In Hand Before Cast** — for a card seen in
   `eot_user_cards_in_hand` on turn *k* (or `opening_hand`, turn 0) and
   then appearing in `user_turn_{k+n}_creatures_cast`/
   `non_creatures_cast`, average *n* across games. User-side only
   (hand visibility). Label: float turn-count. Accumulator — needs
   per-game, per-card temporal state (first-seen-in-hand turn, cast
   turn), not just a running sum — the hardest of this family, same
   as the old brainstorm flagged.

4. **Combat Kill Involvement Rate** — now precisely definable with the
   resolved semantics: `P(card appears in creatures_killed_combat (either
   side) | card appears in that turn's creatures_attacked or
   creatures_blocking)` — i.e. of the times this creature fought (as
   attacker or blocker), how often did *a* creature die as a result
   (not necessarily this one — combat kill involvement, not "this
   creature died"). Single card. Label: float in [0, 1]. Accumulator.

5. **Own-Death-In-Combat Rate** — the sharper sibling of #4: given a
   card resolved as an attacker or blocker, `P(this specific card's id
   appears in the matching-side creatures_killed_combat column that
   same turn | it fought)` — i.e. an actual survival/trade rate, not
   just "something died." Single card. Label: float in [0, 1].
   Accumulator.

6. **Combat Damage Push-Through Rate** — `P(card appears in
   creatures_unblocked | card appears in creatures_attacked)` — replaces
   the old brainstorm's vaguer "combat damage involvement" idea (#5
   there) with something the resolved columns directly answer: how
   often does this creature's attack actually connect versus get
   eaten by a blocker. Single card. Label: float in [0, 1]. Accumulator.

7. **Average Turns On Board** — for a card appearing in
   `eot_{side}_creatures_in_play` on turn *k*, count of consecutive
   turns it keeps appearing there before it stops (killed, bounced, or
   game ends) — resolvable now that both sides' board lists are known
   to be Arena-id lists, not just the user's. Single card. Label:
   float turn-count. Accumulator (needs per-game per-card presence
   tracking across turns, not just a sum/count).

8. **Turns-To-Game-End After Cast** — `num_turns - (turn card was first
   cast)`, for a card resolved out of `creatures_cast`/
   `non_creatures_cast` (either side). Label: float turn-count delta —
   weak/early cards score high (game drags on after they're played),
   strong/expensive cards score low (game ends soon after). Single
   card. Accumulator.

9. **Discard Rate** — `P(card in cards_discarded (its own side's
   discards) in some turn | card in deck_<name>)`. User-deck-only.
   Single card. Label: float in [0, 1]. Accumulator.

10. **Tutor Target Rate (replay-level)** — `P(card in
    user_turn_N_cards_tutored for some N | card in deck_<name>)` —
    per-game, plus optionally the turn-number distribution as a
    follow-on. User-side only (tutoring is drawn-from-library, hidden
    for the opponent). Single card. Label: float in [0, 1].
    Accumulator.

11. **Mulligan-Correlated Candidate Hand Composition** — relative
    frequency of a card across `candidate_hand_1` (always seen) vs.
    `candidate_hand_2..7` (only reached after a mulligan) — cards
    over-represented in mulliganed-away hands are a weak "hard to
    keep" signal. Single card. Label: float (frequency delta). 
    Accumulator.

12. **Attack-Solo Rate** — `P(count(creatures_attacked) == 1 that turn
    | card in creatures_attacked that turn)` — replaces the old
    "likelihood to attack solo" percentage-label framing with an
    explicit conditional probability. Single card. Label: float in
    [0, 1]. Accumulator.

13. **Removal-Signature Rate** — a card-function-inference idea new
    this pass, in the spirit of the skill's "predict value/utility"
    guidance: `P(an opposing creature appears in
    {opposite_side}_creatures_killed_non_combat that same turn | card
    was cast that turn)` — cards that are actually functioning as
    non-combat removal spells/abilities should show a much higher rate
    here than vanilla creatures/non-creatures, without needing any
    hand-authored "is this card removal" label. Single card. Label:
    float in [0, 1]. Accumulator. Caveat: correlational, not causal —
    a card cast the same turn a coincidental combat-adjacent kill
    happens will inflate the rate; still a useful weak-supervision
    signal at scale.

---

## Multi card / multi group (9)

1. **Fixed-Turn Board State → Win Prediction** — freeze both sides'
   public board state at a fixed turn *N* (`eot_user_lands_in_play`/
   `creatures_in_play`/`non_creatures_in_play` as one group,
   `eot_oppo_*` mirrors as a second group — **both fully resolvable**,
   correcting the old brainstorm's assumption that the opponent's side
   would need bare counts) plus each side's life total, predict `won`.
   Multi group (two groups, genuinely symmetric/public this time — the
   asymmetry is confined to hand contents, which this version can
   choose to include (`eot_user_cards_in_hand`, ID list) or exclude
   (`eot_oppo_cards_in_hand`, count-only) as a third, deliberately
   asymmetric group). Label: bool. Streaming (one row -> one example
   per chosen freeze-turn N) — picking N (fixed vs. variable per game,
   e.g. "turn 4" vs. "the turn before the game ended") is still an open
   design question, as the old brainstorm noted.

2. **Attacker Group vs. Blocker Group → Combat Outcome** — narrower
   than #1: group 1 = one turn's `creatures_attacked` (with the
   `creatures_blocked`/`unblocked` split as auxiliary per-member info),
   group 2 = that turn's `creatures_blocking`, predict net result
   (e.g. which side lost more total creatures that turn, via the
   `creatures_killed_combat` columns). Multi group. Label:
   classification-fixed (user-favorable / oppo-favorable / even trade)
   or a signed count delta. Streaming — one row's one combat turn is
   one self-contained example (though a game has many turns, so this
   is really "N examples per row," one per turn with any attackers).

3. **Full Deck → Game Length Regression** — `deck_<name>` as one group,
   predict `num_turns`. User-deck-only (mirrors game_data's per-card
   version but at the deck level). Multi card (whole deck, ungrouped).
   Label: regression-continuous. Streaming.

4. **Full Deck → Combat-Aggression-Profile Prediction** — given the
   full `deck_<name>` list, predict an aggregate combat-tempo scalar
   derived from the same game's per-turn columns (e.g. average
   `count(creatures_attacked)` per turn the deck's controller took, or
   the turn number of that side's first attack) — an archetype-style
   "how aggressive does this 40 fill out to be" signal, distinct from
   any one card's own stats. Multi card (whole deck). Label:
   regression-continuous. Streaming.

5. **Opening Hand vs. Rest-of-Deck → Win Prediction** — group 1 =
   `opening_hand` (post-mulligan, kept hand), group 2 = `deck_<name>`
   minus group 1, predict `won`. Multi group (a natural "hand quality
   relative to what's left" comparison, distinct from #1's board-state
   framing). Label: bool. Streaming.

6. **Mulligan Sequence → Final Decision** — treat
   `candidate_hand_1, candidate_hand_2, ..., candidate_hand_k` (up to
   `opening_hand`) as an ordered sequence of groups, predict
   `num_mulligans` (how many were rejected before keeping) or, per
   step, a keep/mulligan binary. Multi group, sequence-shaped — the
   richest use of the mulligan data, beyond #11's single-card frequency
   framing above. Label: classification-fixed (mulligan count) or
   ranking over the sequence. Streaming.

7. **Revealed-Opponent-Cards vs. Own Deck → Win Prediction** —
   group 1 = user's full `deck_<name>`, group 2 = every opposing card
   actually resolved this game (union of everything seen in
   `oppo_turn_*_creatures_cast`/`non_creatures_cast`/
   `eot_oppo_*_in_play`, deduped) — NOT the opponent's true deck, only
   what got revealed, so group 2 is a strict undercount/partial view by
   construction. Multi group. Label: bool (`won`). Flagged
   lower-confidence than the others here specifically because group 2's
   membership is incomplete by definition (cards never cast/played
   leave no trace) — worth a design discussion on whether a
   systematically-incomplete group is acceptable input before building
   this one, rather than assumed away.

8. **Full-Game Turn-State Trajectory → Win Prediction** — the
   sequence-over-time generalization of #1: instead of one frozen
   turn, the ordered list of every turn's `(eot_user_*, eot_oppo_*)`
   group-pair across the whole game, predict `won`. Multi group,
   sequence-shaped. Label: bool. Streaming in the sense that one row
   still yields one example, but this is explicitly a future
   sequence-model-shaped dojo, not a simple scalar-freeze one — noting
   it as the natural end-state of #1 rather than building it first.

9. **Candidate-Hand Group → Keep/Mulligan Classification** — group =
   one `candidate_hand_i`'s 7 cards, label = whether *that specific
   hand* was kept (only the last one, `== opening_hand`) or mulliganed
   away (every earlier one). Multi card (one hand as a group, not a
   sequence — contrast with #6, which models the whole sequence at
   once; this is the single-hand-at-a-time version). Label: bool.
   Streaming — one row yields up to 7 examples (one per candidate hand
   seen).

---

## Open design questions carried forward

- Picking freeze-turn *N* for ideas #1/#2 above (fixed vs. per-game
  variable) is unresolved — needs a design pass before implementation,
  not a brainstorm-time decision.
- `cards_discarded`/`cards_tutored`'s actor-relative reading (single
  field per turn, no user_/oppo_ split) is a best-effort inference from
  column-presence patterns, not confirmed on one fully-populated row
  the way the combat family was — worth a quick double-check before
  metric #9/#10 above are implemented, specifically whether a
  turn-prefix's `cards_discarded` can ever reflect the *non-active*
  player being forced to discard by an effect (e.g. a Liliana-style
  discard spell cast on someone else's turn) rather than only
  cleanup-step self-discards.
- Idea #7 (multi-group) has a structurally incomplete group by
  construction — flagged, not resolved; may be better framed as a
  weaker "known-partial-opponent-signal" auxiliary input to another
  dojo rather than its own primary prediction target.


## Human Review Short List of Top Metrics

Single Card Metrics:
- **Average Turn Cast**
- **Cast Rate**
- **Average Turns Held In Hand Before Cast**
- **Combat Kill Involvement Rate**
- **Combat Damage Push-Through Rate** — `P(card appears in creatures_unblocked | card appears in creatures_attacked)`
- **Average Turns On Board**
- **Turns-To-Game-End After Cast**
- **Discard Rate** — `P(card in cards_discarded (its own side's discards) in some turn | card in deck_<name>)`
- **Tutor Target Rate (replay-level)** — `P(card in user_turn_N_cards_tutored for some N | card in deck_<name>)`
- 

Multi Card Metrics:
- **Full Deck → Game Length Regression**
- **Full Deck → Combat-Aggression-Profile Prediction**
- **Candidate-Hand Group → Keep/Mulligan Classification**

Multi Group Metrics:
- **Fixed-Turn Board State → Win Prediction**
-  **Attacker Group vs. Blocker Group → Combat Outcome** 
  - Predict which creatures killed
  - Predict if damage made it through to defending player
- **Opening Hand vs. Rest-of-Deck → Win Prediction**
- **Revealed-Opponent-Cards vs. Own Deck → Win Prediction**