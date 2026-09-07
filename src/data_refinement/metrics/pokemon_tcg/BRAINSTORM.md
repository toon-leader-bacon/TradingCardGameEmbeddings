# BRAINSTORM (pokemon_tcg / Pokémon TCG card and theme-deck data)

A todo/idea list of candidate metrics for the Pokémon TCG, produced by
scanning `data/raw/pokemon_tcg/cards/*.json` (per-set card dumps) and
`data/raw/pokemon_tcg/decks/*.json` (per-set official theme decks) —
see `src/data_retrieval/pokemon_tcg/downloader.py`. None of this is
designed or built yet — this is raw material for picking what to
build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed by sampling `cards/bw1.json` (115 cards) and
`decks/bw1.json` (3 decks):

- `supertype`: `Pokémon` (94/115 in this set), `Trainer` (13),
  `Energy` (8) — the three fundamental card kinds.
- Pokémon cards: `hp` (string numeral), `types` (list, usually
  single-element), `subtypes` (`Basic`/`Stage 1`/`Stage 2`/etc.,
  evolution stage), `evolvesFrom`/`evolvesTo` (name-string links
  forming an evolution chain), `attacks` (list of `{name, cost
  [energy-type list], convertedEnergyCost, damage, text}`),
  `abilities` (list of `{name, text, type}` — a smaller, optional
  field only some Pokémon have, e.g. "Royal Heal"), `weaknesses`/
  `resistances` (`{type, value}`, e.g. `×2`/`-20`), `retreatCost`/
  `convertedRetreatCost`, `rarity`, `legalities` (per-format legal/
  banned map, e.g. `unlimited`/`standard`/`expanded`),
  `nationalPokedexNumbers`.
- Trainer cards: `subtypes` (`Item`/`Supporter`/`Stadium`/etc.),
  `rules` (list of full effect-text strings) instead of `attacks`.
- Energy cards: presumably `subtypes` (`Basic`/`Special`) — not
  directly sampled from a full record in this review, worth
  confirming field shape before building an Energy-specific metric.
- `decks/<set>.json` is a **small, fixed list of official pre-built
  theme decks per set** (3 in this sample: `Green Tornado`, etc.),
  each `{id, name, types, cards: [{id, name, rarity, count}, ...]}` —
  not player-submitted competitive decks. This is structurally closest
  to `hearthstonejson`'s card-only shape than to a real deck corpus:
  useful for a handful of curated-deck metrics, but far too small (a
  handful of decks per set) to support real deck-composition
  statistics at scale.
- `legalities` gives real per-format banlist/rotation data — a
  structured field this project's other sources mostly lack (only
  `pitchstack`/`fabtcg_decklists` gesture at format at all, and only
  at the deck level, not per-card).

## Known biases / known-unknowns

- **No real player decks or match/tournament outcome data exists in
  this container** — `decks/*.json` is official pre-built starter
  decks only, not a metagame sample; every multi-card/multi-group
  metric below that needs deck composition should be understood as
  either card-intrinsic (joined from `cards/*.json` alone) or scoped
  to the small theme-deck set, not a "real deck" statistic.
- **`hp`/`damage`/cost fields are strings, not numbers, in the raw
  JSON** — need explicit parsing/casting before use as regression
  targets; a card with variable/conditional damage (e.g. "20+") would
  need a policy decision (strip suffix? treat as non-numeric?) before
  being usable at all.
- **Cross-set card reprints exist under different `id`s** — the same
  named Pokémon (e.g. "Snivy") likely appears with different stats/
  art across sets; whether/how this project's own card identity
  resolution collapses these (the way `card_binder` does for other
  sources) needs checking before any reprint-drift metric is built.
- **`abilities` is present on only a minority of Pokémon cards** —
  any ability-presence metric needs to note the resulting class
  imbalance up front.

## Candidate Metrics

### Single-card (14)

1. **Supertype Prediction** — Input: single-card. Label:
   classification (fixed-set: Pokémon/Trainer/Energy). Predict
   `supertype` from name + whatever text/stat fields are present —
   near-trivial given how structurally different the three supertypes
   are; a sanity-check baseline more than a research target.
2. **HP Prediction** — Input: single-card. Label: count. For Pokémon
   cards, predict `hp` from `types` + `subtypes` + attacks/abilities
   text.
3. **Attack Damage Prediction** — Input: single-card. Label: count.
   Predict an attack's `damage` from its `cost`/`convertedEnergyCost`
   + text — the Pokémon analogue of cardvault_fabtcg's power/defense
   prediction idea.
4. **Energy Cost Prediction** — Input: single-card. Label: count.
   Predict `convertedEnergyCost` from `damage` + attack text — the
   inverse framing of #3, a "power level" proxy like several other
   sources' cost-prediction metrics.
5. **Type Prediction from Attack Text** — Input: single-card. Label:
   classification (fixed-set, over Pokémon energy types). Predict
   `types` from attack names/text/damage alone.
6. **Weakness Type Prediction** — Input: single-card. Label:
   classification (fixed-set, over energy types). Predict a Pokémon's
   `weaknesses` type from its own `types` + attack text — tests
   whether the game's known type-matchup structure (e.g. Grass weak
   to Fire) is learnable from card content alone.
7. **Retreat Cost Prediction** — Input: single-card. Label: count.
   Predict `convertedRetreatCost` from `hp` + `types` + `subtypes`.
8. **Evolution Stage Prediction** — Input: single-card. Label:
   classification (fixed-set: Basic/Stage 1/Stage 2/etc.). Predict
   `subtypes`' evolution stage from `hp` + attack power + `evolvesFrom`
   presence/absence.
9. **Ability Presence Prediction** — Input: single-card. Label:
   probability. `P(has an ability | hp, types, subtypes, rarity)` —
   flagged above as imbalanced (a minority-class target).
10. **Trainer Subtype Prediction** — Input: single-card. Label:
    classification (fixed-set: Item/Supporter/Stadium/etc.). For
    Trainer cards, predict `subtypes` from `rules` text alone.
11. **Rarity Prediction** — Input: single-card. Label: classification
    (fixed-set, over observed rarity strings). Predict `rarity` from
    a card's other stat/text fields — a collector/power-level proxy,
    same role as several other sources' rarity-prediction ideas.
12. **Format Legality Prediction** — Input: single-card. Label:
    classification (variable-set, over `legalities`' format keys).
    Predict which formats (`unlimited`/`standard`/`expanded`) a card
    is legal in from its own set/rarity — largely a release-date/
    rotation-window inference task once enough sets are ingested.
13. **Resistance Presence Prediction** — Input: single-card. Label:
    probability. `P(has a resistances entry | types, hp, subtypes)`.
14. **Evolution-Chain Name Prediction** — Input: single-card. Label:
    classification (variable-set, over Pokémon names in the same
    set). Given a Pokémon's own stats, predict `evolvesTo`'s name —
    narrower and more specific than a general masking task, since
    evolution names are drawn from a fairly constrained real-world
    Pokédex vocabulary.

### Multi-card (9)

15. **Evolution-Chain Pairing** — Input: multi-card (pair: a Pokémon
    and its `evolvesFrom`/`evolvesTo` target). Label: probability.
    `P(card_2 is card_1's evolution | card_1, card_2)` — a
    structurally clean pairwise relationship this source uniquely
    supports well, unlike any single-face/no-evolution-chain source
    in this project.
16. **Type-Legality Pairing** — Input: multi-card (pair). Label:
    probability (deterministic sanity-check). `P(pair_can_share_a_deck)`
    — Pokémon TCG has essentially no color-identity deckbuilding
    restriction (unlike MTG/FaB), so this is close to always-true;
    included as a baseline check rather than a real research target.
17. **Weakness/Resistance Counter Pairing** — Input: multi-card (pair).
    Label: probability. `P(card_2 is a strong counter to card_1 |
    card_1's types, card_2's attacks and types)` — whether a model can
    learn the matchup implications of the weakness/resistance system
    from card text/stats alone.
18. **Theme-Deck Type Pair Co-Occurrence** — Input: multi-card (two
    Pokémon in the same theme deck). Label: probability. `P(card_2 in
    deck | card_1 in deck)` — real but very small-sample deck co-
    occurrence, scoped to the handful of official theme decks per set;
    flagged as low-data given the tiny deck corpus.
19. **Masked Card-in-Theme-Deck Prediction** — Input: multi-card
    (theme deck with one card masked). Label: classification
    (variable-set, over cards in the deck's stated `types`). The
    standard masking task, but scoped to this source's small
    official-deck corpus rather than a real metagame sample.
20. **Attack-Cost Curve of a Hypothetical Pool** — Input: multi-card
    (small synthetic group, since no real large-scale deck data exists
    here). Label: count (distribution). Predict the distribution of
    `convertedEnergyCost` across a synthetically-sampled small group.
21. **Cross-Set Reprint Pairing** — Input: multi-card (pair: same-
    named Pokémon across two sets). Label: probability. `P(same
    underlying Pokémon, different print | card_1, card_2)` — depends
    on confirming this project's card-identity handling for this
    source (flagged above as unverified).
22. **Trainer-Pokémon Synergy Pairing** — Input: multi-card (a Trainer
    card + a Pokémon card). Label: probability. `P(Trainer's rules
    text specifically benefits this Pokémon's types/subtype | Trainer
    rules text, Pokémon types/subtype)` — a targeted synergy
    detection task, e.g. type-specific search/draw Trainers.
23. **Deck Type-Pair Composition Prediction** — Input: multi-card
    (theme deck's `types` field, treated as a 2-element multiset).
    Label: classification (fixed-set pair, over observed dual-type
    combinations). Predict a theme deck's stated `types` from its
    card list — small-sample but a clean structured target.

### Multi-group (7)

24. **Type Representation Balance** — Input: multi-group (whole
    corpus, one set or across all ingested sets). Label: count
    (distribution). Count of Pokémon cards per `types` value — a
    dataset-balance stat before training #5/#6.
25. **Rarity-by-Supertype Distribution** — Input: multi-group (whole
    corpus, cross-tabulated by supertype × rarity). Label: count
    (distribution). Whether Trainer/Energy cards skew toward
    different rarity tiers than Pokémon cards.
26. **Evolution-Chain Length Distribution** — Input: multi-group
    (whole corpus, chains reconstructed via `evolvesFrom`/`evolvesTo`
    links). Label: count (distribution). How many Pokémon have 1-stage
    vs. 2-stage vs. 3-stage evolution chains — a structural balance
    check before training #8/#14/#15.
27. **Set-over-Set Power Creep** — Input: multi-group (all Pokémon
    cards, grouped by set in release order). Label: regression-
    continuous (per-set averages). Whether average `hp`/attack
    `damage` per energy cost trends upward across sets — a power-creep
    sanity check, same framing as the `gwent_one` version.
28. **Format Legality Coverage Over Time** — Input: multi-group (all
    cards, grouped by set/rotation era). Label: count (distribution).
    How many cards from each set remain legal in `standard` over
    time — a rotation-pattern study, useful context before training
    #12.
29. **Theme-Deck Type-Pair Frequency** — Input: multi-group (all
    theme decks across all ingested sets). Label: count
    (distribution). Which dual-type combinations Pokémon's own
    official theme decks favor — small-sample but free, a check for
    #23.
30. **Ability Vocabulary Coverage** — Input: multi-group (all Pokémon
    cards with a non-empty `abilities` list). Label: classification
    (variable-set, over ability-name clusters). Whether named
    abilities cluster into a small reusable vocabulary (many Pokémon
    share near-identical ability text) vs. being mostly unique per
    card — informs whether ability-name prediction (a variant of #9)
    is a fixed-set or open-vocabulary task.
