# BRAINSTORM (spire_codex / Slay the Spire 2 card data)

A todo/idea list of candidate metrics for Slay the Spire 2's card
pool, produced by fetching and scanning the live 577-row
`cards.json` dump spire-codex publishes (see
`src/data_retrieval/spire_codex/card_downloader.py` and
`src/data_refinement/card_binder/spire_codex/ingestion_stage.py`).
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed against the live `cards.json` (577 rows):

- `color`: one of 11 values — 5 character colors at ~87-88 rows each
  (`ironclad`, `silent`, `defect`, `regent`, `necrobinder`), plus
  `colorless` (64), `event` (27), `curse` (18), `token` (14),
  `status` (12), `quest` (3). This is spire-codex's card-color-identity
  field, directly analogous to the "color identity" dependency the
  17lands brainstorm flags as unresolved for MTG.
- `type`: `Skill` (234), `Attack` (195), `Power` (111), `Curse` (18),
  `Status` (16), `Quest` (3).
- `rarity`: `Uncommon` (219), `Rare` (155), `Common` (100), `Basic`
  (19), `Event` (19), `Ancient` (18), `Curse` (18), `Status` (16),
  `Token` (10), `Quest` (3) — note `rarity` conflates true rarity
  tiers (Common/Uncommon/Rare) with card-category markers (Curse,
  Status, Token, Quest, Event) that aren't rarity in the gameplay
  sense; any rarity-prediction metric should probably restrict to the
  Basic/Common/Uncommon/Rare/Ancient subset.
- `cost`: mostly 0-3 (293 rows cost 1, 114 cost 0, 94 cost 2), a
  handful of outliers up to 9; `-1` (31 rows) marks X-cost cards
  (`is_x_cost`/`is_x_star_cost`/`star_cost` fields refine this further
  for star-cost cards).
- `damage`/`block`/`hit_count`/`cards_draw`/`energy_gain`/`hp_loss`:
  structured numeric effect fields, populated only where relevant
  (e.g. 212/577 rows have non-null `damage`). These are pre-parsed
  game-mechanical fields, not something that has to be regexed out of
  `description` — a real advantage over card-text-only sources.
- `powers_applied`: a list of `{power, amount, power_key}` objects for
  cards that grant player powers (Thorns, Dexterity, etc.) — the
  closest analogue to a structured "keyword ability" field.
- `keywords`: present on 156/577 rows; dominated by `Exhaust` (99),
  `Unplayable` (26), `Ethereal` (18), `Retain` (11), `Innate` (9),
  `Sly` (8), `Eternal` (7) — unlike Flesh and Blood's cardvault
  brainstorm, this keyword list is already structured data, no
  rules-text regex/sourcing needed.
- `upgrade`: present on 541/577 rows, a dict of field-name → delta
  (e.g. `{"thornspower": "+2"}`, `{"cost": 0}`) describing exactly
  what changes on upgrade; `upgrade_description` gives the upgraded
  card's full text. This is a genuinely rich, structured
  "predict the diff" target no other source in this project has.
- `description_raw` embeds unresolved template variables (e.g.
  `{CalculatedDamage:diff()}`) resolved via each row's own `vars`
  dict — `description` is the pre-resolved, human-readable text and
  is almost certainly the better model input of the two.
- `target`: values like `Self`, `AnyEnemy`, `AllEnemy` — a card's
  targeting shape, another structured field free of any text-parsing
  step.

## Known biases / known-unknowns

- **Card data only, no play/run data in this container** — run-level
  outcome data lives in the separate `sts_gg` source (a different
  project entirely, already resolving its own card references through
  this same `CardBinder` — see `card_binder/spire_codex/README.md`).
  Every metric here is therefore card-intrinsic or card-relationship,
  never outcome-conditioned; `sts_gg`'s own `BRAINSTORM.md` is where
  deck/outcome metrics belong.
- **`cards.json` reflects the current live card pool only** — the
  downloader always overwrites rather than versioning past pulls (see
  `card_downloader.py`'s docstring), so patch-history/balance-change
  metrics (unlike HearthstoneJSON's per-build snapshots) aren't
  possible from this source alone without deliberately archiving
  successive pulls over time.
- **`rarity` conflates rarity-tier and card-category** — flagged above;
  any rarity metric needs to filter to genuinely-rarity-bearing rows
  first.

## Candidate Metrics

### Single-card (16)

1. **Cost Prediction** — Input: single-card. Label: count. Predict
   `cost` from `description`/`type`/`damage`/`block`/etc., cost
   masked.
2. **Color Prediction from Effect Text** — Input: single-card. Label:
   classification (fixed-set, 11 colors). Predict `color` from
   `description` + structured effect fields alone.
3. **Type Prediction** — Input: single-card. Label: classification
   (fixed-set: Skill/Attack/Power/Curse/Status/Quest). Predict `type`
   from `description` + `damage`/`block`/`cost`.
4. **Rarity Prediction** — Input: single-card. Label: classification
   (fixed-set: Basic/Common/Uncommon/Rare/Ancient, category rows
   excluded). Predict `rarity` from `description` + `type` + `cost`
   — a power-level proxy analogous to the cardvault_fabtcg brainstorm's
   cost-prediction idea.
5. **Damage Value Prediction** — Input: single-card. Label: count.
   For Attack-type cards with non-null `damage`, predict the value
   from `cost`/`hit_count`/`description`.
6. **Block Value Prediction** — Input: single-card. Label: count. For
   cards with non-null `block`, predict the value from `cost`/
   `description`.
7. **Keyword Detection** — Input: single-card. Label: classification
   (variable-set, multi-label). Predict `keywords` (Exhaust/
   Unplayable/Ethereal/Retain/Innate/Sly/Eternal/etc.) from
   `description` alone — unlike cardvault_fabtcg's equivalent idea,
   the keyword vocabulary is already known/structured, no sourcing
   dependency.
8. **Applied-Power Prediction** — Input: single-card. Label:
   classification (variable-set, over the game's power vocabulary).
   Predict which `powers_applied` entries (power name + amount) a
   card grants, from `description` alone.
9. **Target Shape Prediction** — Input: single-card. Label:
   classification (fixed-set: Self/AnyEnemy/AllEnemy/etc.). Predict
   `target` from `description`.
10. **Upgrade-Diff Prediction** — Input: single-card. Label:
    classification (variable-set, over field names changed).
    Predict which fields change on upgrade (the keys of `upgrade`,
    e.g. `cost`, `thornspower`) from the base card's `description` +
    `type` + `cost`.
11. **Upgrade-Delta Magnitude Prediction** — Input: single-card. Label:
    count. For a specific known-upgraded field (e.g. `damage`),
    predict the numeric delta applied on upgrade.
12. **X-Cost Prediction** — Input: single-card. Label: probability.
    `P(is_x_cost | description, type)` — whether a card's cost scales
    with available resources.
13. **Compendium Order Prediction** — Input: single-card. Label:
    count. Predict `compendium_order` from other fields — weakest/
    most speculative entry, a UI-ordering artifact rather than a
    gameplay signal, included for completeness.
14. **Hit Count Prediction** — Input: single-card. Label: count. For
    multi-hit Attack cards, predict `hit_count` from `description` +
    `damage`.
15. **Energy Gain Prediction** — Input: single-card. Label: count.
    For Skill/Power cards, predict `energy_gain` from `description`.
16. **Card-Draw Prediction** — Input: single-card. Label: count.
    Predict `cards_draw` from `description` — a specific instance of
    the general "predict a structured numeric field from text" family
    this source supports unusually well.

### Multi-card (8)

17. **Color-Legality Pairing** — Input: multi-card (pair). Label:
    probability. `P(pair_playable_in_same_deck | color_1, color_2)` —
    trivial in Slay the Spire 2 (any character can normally only run
    their own color + colorless/curse/status/event cards), included
    as a sanity-check baseline like cardvault_fabtcg's pitch-
    sufficiency idea, not a research target.
18. **Same-Character Card Family Grouping** — Input: multi-card (pair).
    Label: probability. `P(same_color | card_1, card_2)` inferred
    purely from description-text similarity, without reading the
    `color` field directly — a free auxiliary/pretraining signal.
19. **Power-Synergy Pairing** — Input: multi-card (pair). Label:
    probability. `P(share_a_powers_applied entry | card_1, card_2)` —
    the Slay the Spire 2 analogue of cardvault_fabtcg's talent-
    co-occurrence idea, derived purely from card text/structured
    fields since no deck-level co-occurrence data lives in this
    container (see `sts_gg`/`sts2runs` for real deck data instead).
20. **Base-vs-Upgraded Invariance Check** — Input: multi-card (pair: a
    card and its own upgraded self, via `upgrade_description`). Label:
    none (embedding-similarity evaluation, not a supervised target) —
    an encoder should place these close but not identical, mirroring
    cardvault_fabtcg's reprint-invariance idea.
21. **Keyword Co-Occurrence** — Input: multi-card (pair). Label:
    probability. `P(share_a_keyword | card_1, card_2)`.
22. **Cost-Curve Pair Complementarity** — Input: multi-card (pair).
    Label: probability (deterministic sanity-check). `P(cost(card_1)
    + cost(card_2) <= energy_gain(card_2))` — an arithmetic baseline
    like cardvault_fabtcg's pitch-for-cost idea.
23. **Type Composition of a Hypothetical Small Pool** — Input:
    multi-card (small synthetic group, since no real draft/deck data
    exists in this container). Label: classification (fixed-set: the
    plurality type). Predict the plurality `type` across a
    synthetically-sampled small group — a stand-in until real deck
    data (see `sts_gg`/`sts2runs`) is joined in.
24. **Attack/Skill/Power Ratio Regression** — Input: multi-card (small
    synthetic group). Label: regression-continuous. Predict the
    ratio of Attack:Skill:Power cards in a group from a partial
    subset of that group — a compositional-inference task.

### Multi-group / corpus-level (6)

25. **Color Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Count of cards per `color`
    across the corpus — a dataset-balance stat before training any
    color-prediction metric (#2).
26. **Type-by-Color Distribution** — Input: multi-group (whole corpus,
    cross-tabulated by color × type). Label: count (distribution).
    Whether each character color favors Attack vs. Skill vs. Power at
    different rates — a stratification check for #3/#23.
27. **Keyword Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Frequency of each keyword
    (#7's target) across the corpus — 156/577 rows carry any keyword
    at all, and `Exhaust` alone accounts for 99 — flag severe
    imbalance before training #7.
28. **Description Length Distribution** — Input: multi-group (whole
    corpus, split by type/rarity). Label: count (distribution).
    Word/character-count distribution of `description` — validates
    whether the text-driven single-card metrics above (#1-#10) have
    enough signal to work with.
29. **Upgrade-Field Frequency Table** — Input: multi-group (all
    upgraded cards, 541/577 rows). Label: count (distribution). Which
    fields (`cost`, `damage`, a named power amount, etc.) most
    commonly change on upgrade across the whole corpus — a
    dataset-balance stat before training #10/#11.
30. **Cost Distribution by Rarity** — Input: multi-group (whole corpus,
    split by rarity). Label: count (distribution). Whether higher-
    rarity cards skew toward higher or more variable `cost` — a
    sanity check for #1/#4 before committing to them as training
    targets.
