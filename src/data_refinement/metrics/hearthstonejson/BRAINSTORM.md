# BRAINSTORM (hearthstonejson / Hearthstone card data, per-build)

A todo/idea list of candidate metrics for Hearthstone's card pool,
produced by scanning `data/raw/hearthstonejson/<build_id>.json` — 81
per-build snapshots present as of this review (each a full ~30,306-row
card list for that patch's enUS data — see
`src/data_retrieval/hearthstonejson/downloader.py`). None of this is
designed or built yet — this is raw material for picking what to
build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed by sampling one build (`205031.json`, 30,306 rows):

- Core fields: `id` (a stable per-card string id, e.g. `AT_003`,
  reused across builds for the same card), `dbfId` (a stable numeric
  id), `name`, `cardClass` (hero class or `NEUTRAL`), `type`
  (`MINION`/`SPELL`/`HERO`/`WEAPON`/etc.), `cost`, `rarity`, `set`
  (expansion/core-set code), `collectible` (bool — many rows are
  non-collectible dev/tavern-brawl/AI-only cards, e.g. the sampled
  `Golden Legendary` TB row with `attack`/`health` both 0).
- Minion-specific: `attack`, `health`, `race`/`races` (tribe, e.g.
  `UNDEAD`), `mechanics` (a structured array of named mechanic tags,
  e.g. `HEROPOWER_DAMAGE` — Hearthstone's own equivalent of
  cardvault_fabtcg's unsourced keyword-ability problem, except
  already structured and enumerable here).
- Spell-specific: `spellSchool` (e.g. `FIRE`), `text` (effect text,
  with `$N`/`#N` placeholders for values that scale with spell
  damage/etc. — needs stripping/resolving before use as clean model
  input, unlike Scryfall's already-resolved `oracle_text`).
- Hero-specific (per the earlier single-card sample already reviewed
  in this project): `health`, `heroPowerDbfId` (links a Hero card to
  its Hero Power card via `dbfId`).
- `flavor`/`artist` are print-flavor metadata, same category as other
  sources' equivalent fields.
- **81 build files exist, each a complete point-in-time card-pool
  snapshot.** Build ids are opaque integers with no visible
  chronological field on the card rows themselves — ordering builds
  by patch date would need either the build id's own numeric ordering
  (unverified as chronological) or an external patch-date mapping not
  currently in this project. This is the one card-only source in this
  project with genuine multi-snapshot history already on disk, without
  needing a dedicated archival step the way `spire_codex`/`gwent_one`/
  `scryfall` would.

## Known biases / known-unknowns

- **Card data only, no match/deck/ladder data in this container** —
  every metric here is card-intrinsic or card-relationship, never
  outcome-conditioned.
- **Non-collectible rows dominate the file** — Tavern Brawl/AI/dev-only
  cards (like the sampled `ART_BOT_Bundle_001`) should almost
  certainly be filtered to `collectible: true` before most metrics
  below, or they'll inject nonsensical 0/0 stat rows into any
  stat-prediction target.
- **Build-to-date ordering is unverified** — flagged above; any
  metric relying on "build N came before build M" needs that
  confirmed (e.g. against a known external patch-notes timeline)
  before being trusted.
- **`text` contains unresolved `$`/`#` placeholders** — needs a
  resolution/stripping policy (this project doesn't have one yet)
  before being a clean text-prediction input.
- **The same card (`id`) can have different stats across builds**
  (a genuine post-launch balance change) or identical stats (most
  builds, most cards) — any cross-build metric needs to actually diff
  builds rather than assume every appearance is a change.

## Candidate Metrics

### Single-card (13, single-build)

1. **Mana Cost Prediction** — Input: single-card. Label: count.
   Predict `cost` from `text`/`type`/`attack`/`health` — the
   Hearthstone analogue of every other source's cost-prediction idea.
2. **Attack/Health Prediction** — Input: single-card. Label: count
   (two values). For Minion cards, predict `attack`/`health` from
   `cost` + `text` + `race`.
3. **Card Class Prediction** — Input: single-card. Label:
   classification (fixed-set, over hero classes + Neutral). Predict
   `cardClass` from `text` + `type` + `cost`.
4. **Rarity Prediction** — Input: single-card. Label: classification
   (fixed-set: Free/Common/Rare/Epic/Legendary). Predict `rarity`
   from `text` + `cost` + `attack`/`health`.
5. **Type Prediction** — Input: single-card. Label: classification
   (fixed-set: Minion/Spell/Weapon/Hero/etc.). Predict `type` from
   `text` alone.
6. **Race/Tribe Prediction** — Input: single-card. Label:
   classification (fixed-set, over tribes + none). For Minions,
   predict `race` from `text` + `attack`/`health`/`cost`.
7. **Mechanic Tag Detection** — Input: single-card. Label:
   classification (variable-set, multi-label). Predict `mechanics`
   from `text` alone — structured ground truth already exists, no
   sourcing dependency (unlike cardvault_fabtcg's keyword problem).
8. **Spell School Prediction** — Input: single-card. Label:
   classification (fixed-set, over schools + none). For Spells,
   predict `spellSchool` from `text`.
9. **Hero Power Linkage Prediction** — Input: single-card (Hero
   cards). Label: classification (variable-set, over Hero Power
   cards). Predict a Hero's `heroPowerDbfId` target from the Hero's
   own `text`/`cardClass`.
10. **Collectibility Prediction** — Input: single-card. Label:
    probability. `P(collectible | attack, health, cost, text,
    set)` — whether a card "looks like" a real collectible vs. a
    dev/Tavern-Brawl-only card; a data-quality-flag classifier more
    than a gameplay signal.
11. **Set/Expansion Prediction** — Input: single-card. Label:
    classification (fixed-set, over set codes). Predict `set` from
    `text` + stats — a release-era style signal, similar to
    `gwent_one`'s set-prediction idea.
12. **Battlecry/Deathrattle Text Effect-Type Classification** — Input:
    single-card. Label: classification (fixed-set, over broad effect
    categories: damage/draw/buff/summon/etc.). Coarse-grained
    classification of what a card's effect *does*, derived from
    `text` + `mechanics` — a simpler, more tractable version of full
    free-text understanding.
13. **Flavor Text Presence Prediction** — Input: single-card. Label:
    probability. `P(has flavor text | set, collectible, rarity)` —
    weakest/most speculative, same role as cardvault_fabtcg's
    equivalent idea.

### Multi-card (9, mixing single-build and cross-build)

14. **Hero-Class Legality Pairing** — Input: multi-card (pair). Label:
    probability. `P(pair_can_share_a_deck | cardClass_1, cardClass_2)`
    — trivial deck-legality baseline (class + Neutral only), same
    role as other sources' equivalent idea.
15. **Mechanic Co-Occurrence / Synergy Signal** — Input: multi-card
    (pair). Label: probability. `P(share_a_mechanic | card_1,
    card_2)` — a card-text-derived synergy proxy, no real deck data
    exists in this container.
16. **Race/Tribe Synergy Pairing** — Input: multi-card (pair). Label:
    probability. `P(share_a_race | card_1, card_2)` — tribal-synergy
    detection, comparable to `pokemon_tcg`'s evolution-chain idea in
    spirit but for Hearthstone's tribal-typal structure.
17. **Hero + Hero Power Invariance Check** — Input: multi-card (a
    Hero card and its linked Hero Power, via `heroPowerDbfId`).
    Label: none (embedding-relationship evaluation, not a strict
    supervised target) — flag as a structural-consistency check
    rather than a prediction task.
18. **Cross-Build Stat-Change Detection** — Input: multi-card (the
    same `dbfId` across two builds). Label: probability. `P(card's
    attack/health/cost changed between build_1 and build_2 | card,
    build_1, build_2)` — this source's unique advantage: real balance-
    patch history already on disk, unlike any other card-only source
    in this project. Depends on confirming build chronological
    ordering (flagged above).
19. **Cross-Build Nerf/Buff Direction Classification** — Input:
    multi-card (same `dbfId`, two builds where a change is confirmed
    via #18). Label: classification (fixed-set: nerf/buff/rework).
    Given the before/after stat+text diff, classify the direction of
    change.
20. **Cost-Curve Complementarity** — Input: multi-card (pair). Label:
    probability (deterministic sanity-check). `P(cost(card_1) +
    cost(card_2) <= a fixed mana-curve budget)` — an arithmetic
    baseline, same role as other sources' equivalent idea.
21. **Class Representation in a Hypothetical Pool** — Input:
    multi-card (small synthetic group, since no real deck data exists
    in this container). Label: classification (fixed-set, plurality
    class). Predict the plurality `cardClass` across a synthetically-
    sampled small group — a stand-in until real deck/match data (not
    present in this project for Hearthstone) is available.
22. **Reworked-Card Text Similarity** — Input: multi-card (same
    `dbfId`, two builds with a confirmed text change beyond stat
    tweaks). Label: regression-continuous (a text-similarity score).
    How much a card's actual wording changes across a rework vs. a
    pure numeric nerf/buff.

### Multi-group / corpus-level (8)

23. **Class Representation Balance** — Input: multi-group (one build,
    collectible cards only). Label: count (distribution). Count of
    cards per `cardClass` — a dataset-balance stat before training #3.
24. **Mechanic Representation Balance** — Input: multi-group (one
    build). Label: count (distribution). Frequency of each mechanic
    tag across the corpus — needed before training #7.
25. **Cost Curve by Set** — Input: multi-group (one build, split by
    `set`). Label: count (distribution). Whether newer expansions
    ship a different mana-cost distribution than older ones — a
    power-creep/design-trend sanity check.
26. **Attack+Health-per-Cost Trend Across Builds** — Input:
    multi-group (all builds, each build's collectible Minions as a
    group). Label: regression-continuous (per-build averages).
    Whether average stats-per-mana creep upward release over release
    — the clearest power-creep metric this project can build, given
    this source's unique multi-snapshot history.
27. **Nerf/Buff Frequency by Class Over Time** — Input: multi-group
    (all cross-build changes detected by #18, grouped by `cardClass`
    and build-sequence position). Label: count (distribution). Which
    classes get balance-patched most often — a design/balance-history
    study.
28. **Non-Collectible Card Share by Set** — Input: multi-group (one
    build, split by `set`). Label: probability (per-set rate).
    `P(collectible=false | set)` — informs the filtering policy noted
    above before trusting any other metric's training data.
29. **Race Representation Balance** — Input: multi-group (one build,
    Minions only). Label: count (distribution). Count of Minions per
    `race`, including "no tribe" as its own bucket — a dataset-balance
    stat before training #6.
30. **Rarity Distribution by Set** — Input: multi-group (one build,
    split by `set`). Label: count (distribution). Whether rarity mix
    (e.g. Legendary density) shifts across sets/expansion types
    (core set vs. expansion vs. mini-set) — a sanity check for #4.
