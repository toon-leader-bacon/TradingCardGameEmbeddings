# BRAINSTORM (sts2runs / Slay the Spire 2 run data)

Candidate metrics/dojos derivable from a single sts2runs run record
(`data/tmp/single_run.json`, one gzip-NDJSON line from
`src/data_retrieval/sts2runs/downloader.py`'s output). Not a plan, not
scoped or prioritized — a raw idea list to narrow down later.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Every run record gives us, per act, a `map_point_history` list of map
points (`monster` / `elite` / `boss` / `shop` / `rest_site` /
`treasure` / `unknown` (event)), each with a `rooms` entry (encounter
id, monster ids, turns taken) and a `player_stats` entry (gold/HP
deltas, `card_choices`, `cards_gained`, `cards_removed`,
`cards_transformed`, `relic_choices`, `potion_choices`,
`bought_relics`, `event_choices`, `rest_site_choices`,
`upgraded_cards`, `current_upgrade_level`). Top-level fields give
`character`, `ascension`, `win`, `killed_by_encounter`,
`killed_by_event`, final `deck`/`relics`/`potions`, and `_isCheated`.

## Known biases / known-unknowns

- **Survivorship/data-source bias** — this data is only obtainable
  from players motivated enough to install/run tracking tools; it is
  not a random sample of all Slay the Spire 2 play. Aggregate stats
  (win rates, card popularity) skew toward the engaged/skilled-player
  population, not the general player base.
- **Naive "deck at end of run" conditioning is contaminated** — a run
  that ends early (death, or a deliberate reset) has a final deck very
  close to the starting deck. A model conditioned on `deck` alone can
  learn "distance from starting deck" as a shortcut for "how far the
  run got," rather than anything about deck quality. Prefer explicit,
  narrower conditioning (`deck_at_start_of_act_2`,
  `deck & made_it_to_final_boss`) over an unqualified `deck`.
- **`_isCheated` flag exists but its coverage/reliability across the
  full corpus is unverified** — any metric that filters on it should
  treat that filter as a data-quality heuristic, not ground truth.
- **Per-turn/per-room resolution mechanics are confirmed only at the
  shape level** (e.g. `card_choices` is a list, Arena-ID-style
  resolution works) — exact edge cases (ties, empty choice lists,
  skipped rooms) haven't been exhaustively checked against real data.

## Candidate Metrics

### Single-card (11)

1. **Card Win Rate** — Input: single-card. Label: probability.
   `P(win | card in final_deck)` — the naive baseline; flagged above
   as contaminated by early-ending runs, useful mainly as a comparison
   point for the narrower conditioned versions below.
2. **Card Average Deck-Depth Survival** — Input: single-card. Label:
   count. Average floor reached across all runs containing the card,
   independent of win/loss — a popularity/power proxy less biased
   than raw win rate.
3. **Removal Likelihood** — Input: single-card. Label: probability.
   `P(card later appears in a cards_removed event | card in deck)` —
   an attrition-tendency stat per card.
4. **Upgrade Likelihood** — Input: single-card. Label: probability.
   `P(card is upgraded by run end | card in final_deck)`.
5. **Card Ascension-Weighted Win Rate** — Input: single-card. Label:
   probability. `P(win | card in deck, ascension >= k)` for a fixed
   high ascension `k` — isolates whether a card holds up at harder
   difficulty rather than just appearing in easy-mode wins.
6. **Potion Usage Rate** — Input: single-card (potion, same shape as a
   card for this purpose). Label: probability. `P(potion_used |
   potion in final_potions)` — a utilization/hoarding signal.
7. **Relic Pick Rate When Offered** — Input: single-card (relic).
   Label: probability. `P(relic taken | relic in relic_choices)`.
8. **Card Cast/Play-to-Removal Ratio** — Input: single-card. Label:
   count (ratio). Across all runs, fraction of appearances ending in
   a `cards_removed` event vs. an `upgraded_cards` event — a rough
   "kept vs. cut" signal per card.
9. **Card Average Turn Introduced** — Input: single-card. Label: count.
   Average floor/act at which a card is first gained (`cards_gained`)
   across all runs — distinguishes early-game staples from late-game
   luxury picks.
10. **Card Encounter-Kill Association** — Input: single-card. Label:
    probability. `P(encounter cleared in below-average turns | card in
    deck)` — a rough power/efficiency proxy, needs a per-encounter
    turn-count baseline to condition against.
11. **Card Gold Cost Sensitivity** — Input: single-card. Label:
    regression-continuous. Average gold spent in the shop visit
    immediately preceding a card's purchase — a price-elasticity-style
    stat for shop-bought cards specifically.

### Multi-card (12)

12. **P(beat act 2 | deck at start of act 2)** — Input: multi-card
    (deck snapshot). Label: probability. `P(win | deck_at_start_of_act_2)`.
13. **P(beat act 3 | deck at start of act 3)** — Input: multi-card.
    Label: probability. `P(win | deck_at_start_of_act_3)`.
14. **P(beat act 3 | deck at start of act 2)** — Input: multi-card.
    Label: probability. `P(win | deck_at_start_of_act_2)` conditioned
    further on reaching act 3 at all — isolates deck quality from
    "did the run even continue."
15. **Draft Pick Prediction** — Input: multi-card (current deck +
    offered `card_choices`, a variable-size option set). Label:
    classification (variable-set, includes a "took nothing" option).
    `P(card chosen | deck, card_choices)`.
16. **Shop Purchase Prediction** — Input: multi-card (current deck +
    shop's card offerings). Label: classification (variable-set),
    conditioned on at least one card being bought. `P(card bought |
    deck, shop_card_choices, a_purchase_happened)`.
17. **Masked-Card-in-Deck Prediction** — Input: multi-card (deck with
    one card masked). Label: classification (variable-set, over the
    game's full card pool). Predict the masked card from the rest of
    the deck.
18. **Ascension Prediction from Final Deck** — Input: multi-card
    (final deck). Label: classification (fixed-set, ascension levels
    0-20). Predict the ascension level a run was played at, for both
    wins and losses.
19. **Upgraded-Card-Set Prediction** — Input: multi-card (final deck).
    Label: classification (variable-set). Predict which cards in the
    final deck were upgraded (`upgraded_cards`/
    `current_upgrade_level`).
20. **Next-Combat HP Loss Prediction** — Input: multi-card (current
    deck). Label: regression-continuous. Predict HP lost in the
    player's next combat, given the current deck snapshot.
21. **Character Prediction from Deck** — Input: multi-card (deck).
    Label: classification (fixed-set, one per playable character).
    Predict which character a run is for, from deck composition alone
    (character-specific card pools make this largely solvable, useful
    as a sanity-check baseline).
22. **Encounter/Shop/Event Count Prediction** — Input: multi-card
    (final deck). Label: count (multiple values). Predict how many
    elite fights / monster fights / shops occurred over the run.
23. **Upgrade-Target Prediction Given an Upgrade Happened** — Input:
    multi-card (deck at the point of an upgrade choice). Label:
    classification (variable-set), conditioned on an upgrade
    occurring. `P(card upgraded | deck, an_upgrade_did_happen)`.
24. **Card-Removal Target Prediction** — Input: multi-card (deck +
    current gold, at a shop visit). Label: classification
    (variable-set, over cards currently in deck). `P(card removed |
    deck, current_gold)`.

### Multi-group (7)

25. **Event Choice Prediction** — Input: multi-group (deck group +
    event's offered `event_choices`, a distinct option group). Label:
    classification (variable-set). `P(choice taken | deck, event_id,
    event_choices)`.
26. **Rest Site Choice Prediction** — Input: multi-group (deck group +
    HP-state group + fixed rest-site option set). Label:
    classification (fixed-set: rest/smith/dig/recall/etc.). `P(choice
    | deck, current_hp, max_hp, rest_site_choices)`.
27. **Encounter Difficulty Matchup** — Input: multi-group (deck group +
    a specific encounter's monster-id group). Label: regression-
    continuous. Predict `turns_taken` for a specific encounter given
    the deck and that encounter's `monster_ids` — a matchup-specific
    difficulty metric, distinct from the single-card kill-involvement
    metrics above.
28. **Encounter Damage Matchup** — Input: multi-group (deck group +
    encounter's monster-id group). Label: regression-continuous.
    Predict `damage_taken` in a specific encounter given the deck and
    that encounter's `monster_ids` — same matchup framing as #27,
    health-loss instead of turn-count.
29. **Card Co-Occurrence / Synergy Statistic** — Input: multi-group (all
    decks containing card A, all decks containing card B, as two
    groups being compared). Label: probability. `P(card_B in deck |
    card_A in deck)` — a synergy statistic across all runs, useful as
    a pretraining signal for card-embedding proximity independent of
    any single-deck task.
30. **Deck-Power Trajectory Forecast** — Input: multi-group (deck
    snapshot at floor N group + relic-state group at floor N+k).
    Label: count (distribution: relic count and/or rarity mix).
    Given a deck snapshot at floor N, predict relic count/rarity
    distribution at floor N+k — a continuous-progress generalization
    of the ascension/win predictions above.
31. **Cause-of-Death Classification** — Input: multi-group (deck group
    + floor-reached group, losing runs only). Label: classification
    (variable-set, over encounter/event ids). Predict
    `killed_by_encounter`/`killed_by_event` for runs that lost — which
    specific fight or event ended the run.
32. **Map-Route Prediction** — Input: multi-group (sequence-of-visited-
    map-point-types group, deck-independent). Label: classification
    (fixed-set: monster/elite/shop/rest/event/treasure). Predict the
    next map point's type from the sequence already visited this act —
    a deck-independent baseline to condition the deck-dependent
    metrics above against.
