# sts2runs metric brainstorm

Candidate metrics/dojos derivable from a single sts2runs run record
(`data/tmp/single_run.json`, one gzip-NDJSON line from
`src/data_retrieval/sts2runs/downloader.py`'s output). Not a plan, not
scoped or prioritized — a raw idea list to narrow down later.

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

## Already specified

1. `P(beat_act_2 | deck_at_start_of_act_2)`
2. `P(beat_act_3 | deck_at_start_of_act_3)`
3. `P(beat_act_3 | deck_at_start_of_act_2)`
4. `P(draft_card | deck, draft_options_or_pass)` — PDF over a
   post-monster/elite/boss `card_choices` list plus a "took nothing"
   option.
5. `P(buy_card | deck, buy_options, card_is_bought)` — PDF over a
   shop's `card_choices`, conditioned on at least one card having been
   bought.
6. Given a deck with one card masked, guess the masked card.
7. Predict a card's average winrate across runs it appears in.
8. Predict the ascension of the deck a run finished with (win or
   loss).
9. Given a final deck, predict which cards were upgraded
   (`upgraded_cards` / `current_upgrade_level`) by run's end.
10. Given a deck, predict HP lost in the player's next combat.
11. Given a deck, predict which character the run is for.
12. Given a final deck, predict how many elite fights / monster
    fights / shops / etc. occurred over the run.

## Additional ideas

13. `P(relic_pick | deck, relic_choices)` — PDF over a `relic_choices`
    list (treasure rooms, elites, bosses, some events all offer these),
    the relic analogue of the draft-card PDF.
14. `P(potion_pick | deck, potion_choices)` — PDF over a
    `potion_choices` list (elites/shops), the potion analogue of the
    same task.
15. `P(card_removed | deck, current_gold)` — PDF over which card in
    the current deck a shop's `cards_removed` entry targets, given the
    deck composition and gold on hand at that point.
16. `P(event_choice | deck, event_id, event_choices)` — predict which
    option a player takes in an `unknown`/event room, given the deck
    and the event's id and offered `event_choices`.
17. `P(rest_site_choice | deck, current_hp, max_hp)` — predict rest
    vs. smith vs. dig vs. recall etc. from `rest_site_choices`, given
    deck and HP state.
18. Predict `turns_taken` for a specific encounter, given the deck and
    that encounter's `monster_ids`/`model_id` — an encounter
    difficulty/matchup metric, keyed by monster rather than "next
    fight" generically.
19. Predict `damage_taken` in a specific encounter, given the deck and
    that encounter's `monster_ids`/`model_id` — same matchup framing
    as #18, health-loss instead of turn-count.
20. Predict `gold_spent` at a shop visit, given the deck, `current_gold`,
    and that shop's offered cards/relics/potions.
21. Given a final deck and potion list, predict which potions were
    used (`potion_used`) vs. carried to the end unused — a
    potion-utilization/hoarding classifier.
22. Predict a card's removal likelihood — across all runs containing
    that card, what fraction later show it in a `cards_removed` event
    — an attrition-tendency stat per card, independent of any one
    deck.
23. Predict the outcome of a `cards_transformed` event (`final_card`),
    given the `original_card` and the surrounding deck — a
    "what does this become" task for smith/transform-type effects.
24. For lost runs, predict the floor of death (regression), given the
    deck snapshot at some earlier floor.
25. Given two cards, predict `P(both in deck | either in deck)` —
    a co-occurrence/synergy statistic across all runs, useful as a
    pretraining signal for card-embedding proximity independent of any
    single-deck task.
26. Given a deck snapshot at floor N, predict relic count and/or relic
    rarity distribution at floor N+k — a "deck power trajectory"
    forecast, generalizing the ascension/win predictions into a
    continuous-progress signal.
27. Given a final deck/relic/potion signature, predict `_isCheated` —
    an anomaly-detection framing, useful for data-quality filtering
    rather than gameplay modeling.
28. Given a deck, predict `killed_by_encounter`/`killed_by_event` for
    runs that lost — which specific fight or event ended the run,
    as a multi-class target rather than #24's floor regression.
29. Given a deck at the start of an act, predict how much gold will be
    accumulated by that act's boss (an economy-trajectory analogue of
    the HP-loss metric in #10).
30. Given the sequence of `map_point_type`s already visited this act
    (monster/elite/shop/rest/event/treasure so far), predict the next
    map point's type — a route/pathing model independent of deck
    contents, useful as a baseline to condition the deck-dependent
    metrics above against.
