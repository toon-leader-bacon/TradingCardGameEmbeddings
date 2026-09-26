# METRIC_BRAINSTORM — game_data

A candidate-metric list for 17lands `game_data/*.csv` specifically,
produced per the updated `metric_writing` skill: sampled the raw CSV
directly (`game_data/MSH.PremierDraft.csv`, first ~3000 rows via
pandas), re-derived the top-level `BRAINSTORM.md`'s old "Game data"
section with fresh eyes, and gave the two previously-undeveloped ideas
(main/splash color prediction, tutor-target prediction) real design
attention. Raw material for picking what to build next, not a spec;
`README.md` lists which ideas are built.

## Confirmed by this pass (supersedes/extends the shared BRAINSTORM.md)

- **Header** (MSH sample): `expansion, event_type, draft_id, draft_time,
  game_time, build_index, match_number, game_number, rank, opp_rank,
  main_colors, splash_colors, on_play, num_mulligans, opp_num_mulligans,
  opp_colors, num_turns, won`, then five columns per card name:
  `opening_hand_<name>, drawn_<name>, tutored_<name>, deck_<name>,
  sideboard_<name>` (per-row copy counts). 339 distinct card names in
  the MSH sample.
- **`main_colors`/`splash_colors`/`opp_colors`** are short WUBRG-letter
  strings in canonical WUBRG order (e.g. `"WUB"`, `"UBRG"`), never
  delimited — confirmed via `.value_counts()`. `splash_colors` is
  empty/NaN when there's no splash. `opp_rank` was NaN in every sampled
  row (opponent rank apparently unavailable/deprecated in this format
  — verify before depending on it for anything).
- **`rank`** is a tier string (`bronze, silver, gold, platinum, diamond,
  mythic` — no numeric pip), same vocabulary the draft data's
  `rank`/`user_game_win_rate_bucket` family already uses.
- **Deck shape**: `deck_<name>` sums to exactly 40 per game (constructed
  Limited deck size); `sideboard_<name>` sums to the remainder of the
  draft pool (~15-16 in this sample); `opening_hand_<name>` always sums
  to 7 (Arena's London mulligan always redraws to 7, then the player
  bottoms `num_mulligans` cards — game_data's `opening_hand_*` is the
  post-bottom kept hand, NOT the raw pre-mulligan draw, so game_data
  alone cannot see what was in a mulliganed-away hand; that needs
  replay_data's `candidate_hand_N`, out of scope here).
- **`tutored_<name>`** is genuinely rare (254/3000 = ~8.5% of games had
  any tutor effect resolve at all) — any tutor metric here has a small
  positive class and should be built with that in mind (e.g. don't
  silently drop to a degenerate all-zero-recall classifier).
- **Color-identity dependency, resolved**: `GenericCard.raw_content`
  for MTG (from `ScryfallCardIngestionStage`) is the raw Scryfall JSON
  object, which does carry `color_identity` (list of WUBRG letters,
  e.g. `['G']`) and `colors`, plus `cmc`, `mana_cost`, `type_line`,
  `rarity`, `power`/`toughness`, `oracle_text` — confirmed directly
  against `src/data_retrieval/scryfall/example.jsonl`. Every idea below
  that needs "is this card on-color/splash for this deck" can resolve
  it via `card.raw_content["color_identity"]` vs. that game's
  `main_colors`/`splash_colors` — no cross-dataset join needed, since
  `main_colors`/`splash_colors` already live on every game_data row.
- **Card resolution**: resolve every `<name>`-suffixed column via
  `CardBinder.get_by_name_single(GameId.MTG, name)` against a binder
  loaded from `data/final/cards/mtg.jsonl` — `strict=True` (the
  default) will raise on a genuine ambiguous-name collision, which is
  worth letting surface rather than silently picking one, at least
  until a real collision is observed in this format.
- **Shape inspiration**: `metrics/sts_gg/`'s `CardAverageMetric` (per-
  card running-average accumulation) and `DeckLabelMetric` (per-run
  streaming: a full "final deck" plus a scalar already on that same
  row, deduped through a private `DeckBox` via
  `deck_ids.deck_uuid_from_cards()`) are close analogues in shape —
  game_data's `deck_<name>` is exactly sts_gg's "final deck" concept,
  one row per game rather than one row per run.

---

## Single card (10)

1. **Win Rate When In Deck** — `P(won | card in deck_<name>)`. Label:
   probability. Accumulator.

2. **Opening Hand Win Rate** — `P(won | card in opening_hand_<name>)`.
   Label: probability. Accumulator.

3. **Drawn Win Rate** — `P(won | card in drawn_<name>)` — drawn at any
   point in the game, opening hand or not; distinct from #2 since a
   card seen turn 8 still counts here. Label: probability. Accumulator.

4. **On-Play vs. On-Draw Win Rate Delta** — `P(won | card in deck,
   on_play=True) - P(won | card in deck, on_play=False)`. Label:
   signed float (regression-continuous) — a tempo/curve-sensitivity
   proxy. Accumulator (two running win-rate tallies per card, split on
   `on_play`).

5. **Game Length Association** — average `num_turns` over games where
   `card in deck_<name>`, minus the format's overall average
   `num_turns` — surfaces aggro (negative) vs. control (positive)
   cards. Label: signed float (regression-continuous). Accumulator
   (needs one extra pass, or a first pass, for the format-wide average
   baseline before per-card deltas are meaningful).

6. **Tutor Target Rate** — `P(card in tutored_<name> | card in
   deck_<name>)`. Label: probability. Accumulator. Flagged: rare
   positive class (~8.5% of games have any tutor hit at all, see
   above) — most cards' true rate will be nearly 0, and cards actually
   worth tutoring for will be a long tail; consider also emitting
   `sample_count`/raw hit-count alongside the rate so a consumer can
   filter out noisy near-zero estimates from cards seen in few games.

7. **Splash Reliance Rate** — among games where `card in deck_<name>`,
   `P(card.color_identity is a non-empty subset of that game's
   splash_colors, i.e. entirely off of main_colors | card in deck)`.
   Label: probability. Accumulator. Distinct from a simple "is this
   card off-color" static fact (already derivable from the card alone)
   — this measures how often a card that *can* be splashed actually
   gets played as the splash (vs. as a main-color card in a deck whose
   main colors happen to include it, vs. not played at all) — a
   drafted-splashability signal existing metrics don't cover. Resolves
   the old shared BRAINSTORM.md's under-specified "Splash Colors"
   single-card idea (#7) into something concrete.

8. **Rank-Stratified Win Rate** — same shape as #1, but keeping
   separate running `(win_count, total_count)` tallies per `rank`
   bucket — surfaces whether a card overperforms/underperforms at
   higher skill levels specifically. Label: probability, per rank
   bucket (five-ish parallel columns or a `rank: str` row-per-stratum
   shape, a build-time choice). Accumulator.

9. **Sideboard Rate (game-level)** — `P(card in sideboard_<name> |
   card in deck_<name> or sideboard_<name>)`, i.e. given the card was
   in that game's full draft pool, how often was it left in the
   sideboard rather than maindecked. Label: probability. Accumulator.
   Deliberately game_data-local rather than reusing draft_data's
   precomputed `pick_maindeck_rate`/`pick_sideboard_in_rate` columns —
   this version is conditioned on actually having reached a completed
   deck for a real match (not just a draft-time aggregate), so it can
   be cross-checked against those columns or stratified by `rank`/
   archetype the way they can't be out of the box.

10. **Mulligan-Adjacent Association** — average `num_mulligans` over
    games where `card in deck_<name>`, vs. the format's overall
    average. Label: signed float (regression-continuous). Accumulator.
    Explicitly a weak/indirect signal (game_data can't see *which*
    hand was mulliganed away, only how many times it happened for the
    whole game — see "Confirmed" above) — flagged as lower-confidence
    than replay_data's own candidate-hand-based mulligan metric, but
    cheap to compute here as a first pass and worth having even as a
    noisy prior.

---

## Multi card (9)

1. **Deck Composition → Win Prediction** — given the full
   `deck_<name>` list for one game (resolved to `nocab_uuid`s), predict
   `won`. Multi card (whole deck, one ungrouped group — the same shape
   the now-deleted `DeckOutcomeScanner` served, and a close match to
   `DeckLabelMetric`'s pattern from sts_gg). Label: bool
   (classification, fixed-set/binary). Streaming — one row already
   carries a complete `(deck_card_uuids, won)` example.

2. **Main-Colors Prediction** — given the full deck, predict
   `main_colors`. Label: classification (fixed-set) over the small set
   of `main_colors` strings actually observed in this format (mostly
   2-color pairs, occasional mono or 3+-color) — NOT a 5-way
   multi-label-per-color setup, since `main_colors` already comes as
   one discrete string per game rather than 5 independent booleans;
   treating each observed string as its own class keeps the label
   space matched to what the data actually contains. Multi card
   (whole deck). Streaming (label already on the row).

3. **Splash-Colors Prediction** — given the full deck (and, as a
   richer variant, `main_colors` as an additional input feature),
   predict `splash_colors` (including the "no splash" / empty case as
   its own class). Label: classification (fixed-set, including a
   dedicated empty/no-splash class). Multi card. Streaming.

4. **Predict Tutor Targets Given Deck** — given the full draft pool for
   one game (`deck_<name>` ∪ `sideboard_<name>`, resolved to
   `nocab_uuid`s) as the input group, label each pool card with whether
   it appears in `tutored_<name>` that game. Multi group in spirit
   (one shared "pool" input, a per-card boolean output over that same
   pool) rather than a single scalar over the whole group — closer to
   `DeckCardMaskMetric`'s per-card-labeled-from-a-deck shape than
   `DeckLabelMetric`'s one-scalar-per-deck shape, except potentially
   *many* cards get a `True` label per row rather than exactly one
   masked target, so it doesn't fit `DeckCardMaskMetric` verbatim
   either — flagged as needing its own small design pass (most likely:
   one output row per `(deck_uuid, pool_card_uuid, tutored: bool)`,
   mirroring the shared BRAINSTORM.md's draft-data
   `PickedVHeldVPackMetric` precedent for "one row per (group,
   member)"). Label: classification (variable-set, one bool per pool
   member). Accumulator only in the sense of iterating the pool per
   row — no cross-row state needed, so effectively streaming.

5. **Deck → Game-Length Prediction** — given the full deck, predict
   `num_turns`. Label: regression-continuous. Multi card (whole deck).
   Streaming. The deck-level counterpart to single-card idea #5 above
   — lets a dojo learn curve/aggro-vs-control signal from full
   archetype composition rather than one card's marginal association.

6. **Deck → Rank-Tier Prediction** — given the full deck, predict
   `rank`. Label: classification (fixed-set, the six tier strings).
   Multi card. Streaming. A rough "deck power level / build quality"
   proxy — weak (confounded by the human pilot's own skill, not just
   the deck), flagged as speculative/lower-priority, but cheap given
   `rank` is already on every row.

7. **Deck → Mulligan-Count Prediction** — given the full deck, predict
   `num_mulligans`. Label: count (small-integer regression, or
   classification over `{0, 1, 2+}` if the near-all-zero distribution
   makes plain regression a poor fit — worth checking the actual
   distribution before committing to one). Multi card. Streaming.
   Same weak/indirect-signal caveat as single-card idea #10.

8. **On-Play Win-Rate Sensitivity by Deck** — given the full deck,
   predict the same `P(won | on_play) - P(won | on_draw)` delta as
   single-card idea #4, but as one label for the whole deck rather than
   per card — an aggregate "how play/draw-sensitive is this whole
   strategy" signal (e.g. a low-curve aggro deck should show a smaller
   gap than a top-heavy control deck). Label: regression-continuous.
   Multi card. Accumulator (needs the same per-deck on_play/on_draw
   split as the single-card version, keyed by `deck_uuid` instead of
   per-card, so requires a private `DeckBox` to dedupe identical
   decklists the way sts_gg's `DeckLabelMetric` family does).

9. **Deck Color-Identity Consistency (mainboard vs. splash split)** —
   given the full pool (`deck_<name>` ∪ `sideboard_<name>`), what
   fraction of on-`main_colors` cards got maindecked vs. what fraction
   of off-`main_colors`/on-`splash_colors` cards did — a per-game
   "how disciplined was this build" scalar. Label: regression-
   continuous (a ratio/fraction in [0, 1]). Multi group (pool as input,
   with each member's card-vs-deck-colors relationship as the derived
   feature, distinct from a plain flat card list) — flagged, like the
   shared BRAINSTORM.md's draft-data "Color Openness" idea, as not
   fitting the single/multi/multi-group taxonomy perfectly cleanly;
   worth a design conversation before committing to a concrete builder
   shape.

---

## Open items for whichever pass implements these

- Confirm `opp_rank`'s NaN-everywhere pattern holds beyond the MSH
  sample before relying on (or discarding) it anywhere.
- Idea #4 in the multi-card list (tutor targets) needs a real shape
  decision before implementation — flagged above, not resolved here.
- Ideas 6/7/8/9 in the multi-card list all lean on the assumption that
  `rank`/`num_mulligans`/on-play sensitivity carry real *deck* signal
  rather than being dominated by pilot skill/variance — worth a quick
  sanity check (e.g. actual variance explained) before investing in a
  full implementation, not just a brainstorm-time guess.


## Human Review Short List of Top Metrics

Single Card Metrics:
- **Win Rate When In Deck** — `P(won | card in deck_<name>)`
- **Opening Hand Win Rate** — `P(won | card in opening_hand_<name>)`
- **Drawn Win Rate** — `P(won | card in drawn_<name>)`
- **On-Play vs. On-Draw Win Rate Delta** — `P(won | card in deck, on_play=True) - P(won | card in deck, on_play=False)`
- **Game Length Association**
- **Tutor Target Rate** — `P(card in tutored_<name> | card in deck_<name>)`
- 

Multi Card Metrics:
- **Deck Composition → Win Prediction**
- **Predict Tutor Targets Given Deck**
- **Deck → Game-Length Prediction**
- **Deck → Rank-Tier Prediction**
- **On-Play Win-Rate Sensitivity by Deck**
