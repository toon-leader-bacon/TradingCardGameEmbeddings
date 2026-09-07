# BRAINSTORM (gwent_one / Gwent card data)

A todo/idea list of candidate metrics for Gwent's card pool, produced
by scanning the live `data/raw/gwent_one/page_1.html` fragment (1,260
cards, one AJAX response — see
`src/data_retrieval/gwent_one/downloader.py`). None of this is
designed or built yet — this is raw material for picking what to
build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed by parsing every `div.card-wrap.card-data`'s attributes
(1,260 cards) plus its nested `.card-category`/`.card-body-ability`
text:

- `data-faction`: `neutral` (243), `syndicate` (177), `monster` (168),
  `scoiatael` (168), `nilfgaard` (168), `northern_realms` (168),
  `skellige` (168) — Gwent's 6 real factions plus neutral, a fairly
  even split.
- `data-color`: `gold` (705), `bronze` (513), `leader` (42) — card
  rarity-tier-by-power-budget, distinct from `data-rarity`.
- `data-type`: `unit` (937), `special` (205), `artifact` (64),
  `ability` (42, mostly leader ability cards), `stratagem` (12).
- `data-rarity`: `legendary` (406), `epic` (341), `common` (270),
  `rare` (243) — true collector rarity, independent of `data-color`.
- `data-set`: `baseset` (430) plus 11 named expansion sets (each
  ~70-110 cards), giving a real release-order/expansion signal.
- `data-power`/`data-armor`/`data-provision`: numeric stat fields
  present on every card (`0` where not applicable, e.g. non-unit
  cards' power).
- `.card-category` (e.g. `Human, Soldier`, `Witcher`, `Beast`,
  `Relict`, `Spell`, `Alchemy`, `Tactic`) is a comma-joined tribe/
  archetype tag, present on most cards (44 show `&nbsp;`, i.e. no
  category).
- `.card-body-ability` embeds structured `<span class="keyword
  X">` tags inline with the ability's prose — `deploy` (632), `order`
  (358), `spawn` (305), `melee`/`ranged` (147/133, row-placement
  keywords), `zeal` (98), `cooldown` (89), `charge` (87), `summon`
  (86), `armor` (84), `coin` (72), `doomed` (63), `bleeding` (60),
  `profit` (60), `fee` (47), `create` (45), `deathblow` (45),
  `deathwish` (43), `lock` (41), `resilient` (40), and others — a
  genuinely large, already-structured keyword vocabulary extractable
  without a text-parsing/regex step (each keyword is its own HTML
  span, not something that has to be guessed from prose).

## Known biases / known-unknowns

- **Card data only, no match/deck data in this container** — no
  ranked-ladder results, decklists, or draft logs exist for Gwent in
  this project (contrast `play_gwent`, which has curated guide decks
  with vote counts, a different raw source entirely). Every metric
  here is card-intrinsic or card-relationship, never outcome-
  conditioned.
- **This is the current live card pool only** — like `spire_codex`,
  a stale single snapshot with always-overwrite fetching (see
  `downloader.py`'s docstring); patch-history/balance-change metrics
  aren't possible from this source alone without deliberately
  archiving successive pulls.
- **`.card-category`'s `&nbsp;` rows need explicit "no category"
  handling**, not silent inclusion as a literal category value.
- **`power`/`armor`/`provision` being `0` is ambiguous between "not
  applicable to this card type" and "genuinely balanced to zero"** —
  worth checking per `data-type` before using these as raw regression
  targets.

## Candidate Metrics

### Single-card (12)

1. **Faction Prediction from Ability Text** — Input: single-card.
   Label: classification (fixed-set, 7 factions). Predict
   `data-faction` from `.card-body-ability` text + `data-type` +
   `data-color`.
2. **Provision Cost Prediction** — Input: single-card. Label: count.
   Predict `data-provision` from ability text + power/armor — the
   Gwent analogue of cardvault_fabtcg's cost-prediction "power level"
   proxy.
3. **Power Value Prediction** — Input: single-card. Label: count. For
   unit-type cards, predict `data-power` from ability text +
   provision.
4. **Color-Tier Prediction (Gold vs. Bronze)** — Input: single-card.
   Label: classification (fixed-set: gold/bronze/leader). Predict
   `data-color` from ability text + provision + faction — whether a
   card's power budget is inferable from its effect complexity.
5. **Rarity Prediction** — Input: single-card. Label: classification
   (fixed-set: legendary/epic/rare/common). Predict `data-rarity`
   from ability text + faction + type.
6. **Type Prediction** — Input: single-card. Label: classification
   (fixed-set: unit/special/artifact/ability/stratagem). Predict
   `data-type` from ability text alone.
7. **Category/Tribe Prediction** — Input: single-card. Label:
   classification (variable-set, multi-label over comma-joined
   categories). Predict `.card-category` from ability text + faction
   — genuinely multi-label given the comma-joined shape (e.g. "Human,
   Soldier").
8. **Keyword Ability Detection** — Input: single-card. Label:
   classification (variable-set, multi-label). Predict which
   keywords (deploy/order/spawn/zeal/etc.) appear in a card's own
   ability text, from the surrounding prose alone — unlike
   cardvault_fabtcg's equivalent idea, the keyword vocabulary is
   already structured/known here, no sourcing dependency.
9. **Row Placement Prediction (Melee/Ranged)** — Input: single-card.
   Label: classification (fixed-set: melee/ranged/agile-both/neither).
   For unit cards, predict row placement from ability text + power.
10. **Set/Expansion Prediction** — Input: single-card. Label:
    classification (fixed-set, 12 sets). Predict `data-set` from
    ability text + faction + provision — a release-era style/power-
    level signal, similar in spirit to cardvault_fabtcg's reprint-
    drift ideas but for original-release grouping rather than
    reprints.
11. **Armor Value Prediction** — Input: single-card. Label: count.
    For cards with non-zero `data-armor`, predict the value from
    ability text + power.
12. **Leader Ability Charge-Count Prediction** — Input: single-card
    (leader-color cards only, 42 rows). Label: count. Predict the
    leader ability's `Charges: N` value (visible in ability text,
    per the sample fragment) from the rest of the ability text — a
    low-data, narrowly-scoped metric flagged as such.

### Multi-card (10)

13. **Faction-Legality Pairing** — Input: multi-card (pair). Label:
    probability. `P(pair_legal_in_same_deck | faction_1, faction_2)`
    — a card is deck-legal only alongside its own faction or
    `neutral`; a trivial sanity-check baseline like cardvault_fabtcg's
    class-legality idea.
14. **Keyword Co-Occurrence / Synergy Signal** — Input: multi-card
    (pair). Label: probability. `P(share_a_keyword | card_1, card_2)`
    — a card-text-derived synergy proxy, since no real deck-co-
    occurrence data exists in this container (contrast `play_gwent`'s
    guide decks, a separate source).
15. **Category/Tribe Synergy Pairing** — Input: multi-card (pair).
    Label: probability. `P(share_a_category | card_1, card_2)` — the
    tribal-synergy analogue of #14, using `.card-category` instead of
    keywords.
16. **Provision-Curve Complementarity** — Input: multi-card (pair).
    Label: probability (deterministic sanity-check). `P(provision(card_1)
    + provision(card_2) <= a fixed provision-limit budget)` — an
    arithmetic baseline like cardvault_fabtcg's pitch-for-cost idea.
17. **Gold/Bronze Balance in a Hypothetical Pool** — Input: multi-card
    (small synthetic group, since no real deck data exists here).
    Label: regression-continuous. Predict the gold-to-bronze ratio
    across a synthetically-sampled small group — a stand-in until
    real deck data (see `play_gwent`) is joined in.
18. **Faction Representation in a Hypothetical Pool** — Input:
    multi-card (small synthetic group). Label: classification
    (fixed-set, plurality faction). Predict the plurality faction
    across a group.
19. **Reprint/Variant Invariance Check** — Input: multi-card (pair,
    same underlying card across `data-artid` art variants if any
    exist). Label: none (embedding-similarity evaluation, not a
    supervised target) — flagged as an eval metric, mirroring
    cardvault_fabtcg's reprint-invariance idea; needs confirming
    whether gwent.one's listing actually contains multiple `artid`
    variants per card `id` before committing to this.
20. **Set Adjacency Pairing** — Input: multi-card (pair). Label:
    probability. `P(same_set | card_1, card_2)` inferred from ability-
    text style/theme alone, without reading `data-set` directly — a
    free auxiliary/pretraining signal, similar to cardvault_fabtcg's
    family-grouping idea.
21. **Deploy/Order Ability Pairing** — Input: multi-card (pair).
    Label: probability. `P(card_2's order ability targets a
    card-type card_1 belongs to | card_1, card_2)` — a
    targeting-compatibility signal, speculative since it requires
    parsing target references out of ability prose, not just
    detecting keyword presence.
22. **Cross-Faction Neutral-Card Utility** — Input: multi-card (a
    neutral card + one faction's full card pool, as a comparison).
    Label: probability. `P(neutral card commonly thematically fits
    faction | neutral card, faction's typical keywords/categories)` —
    speculative, no ground-truth "fits" label exists without deck
    data.

### Multi-group / corpus-level (8)

23. **Faction Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Count of cards per faction
    — a dataset-balance stat before training any faction-prediction
    metric (#1).
24. **Color-Tier by Faction Distribution** — Input: multi-group (whole
    corpus, cross-tabulated by faction × color). Label: count
    (distribution). Whether some factions run gold-heavy vs.
    bronze-heavy card pools.
25. **Keyword Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Frequency of each keyword
    (#8's target) across the corpus — `deploy` alone appears on
    632/1,260 cards, flag the resulting imbalance before training.
26. **Category/Tribe Representation Balance** — Input: multi-group
    (whole corpus). Label: count (distribution). Count of cards per
    tribe/category tag, including the 44 `&nbsp;` (no-category) rows
    as their own bucket.
27. **Provision Distribution by Rarity** — Input: multi-group (whole
    corpus, split by rarity). Label: count (distribution). Whether
    higher-rarity cards skew toward higher/more variable provision
    cost — a sanity check for #2/#5 before training either.
28. **Ability Text Length Distribution** — Input: multi-group (whole
    corpus, split by type/color). Label: count (distribution).
    Word/character-count distribution of `.card-body-ability` text —
    validates whether the text-driven single-card metrics (#1, #6-9)
    have enough signal to work with.
29. **Set Size and Power-Level Trend** — Input: multi-group (whole
    corpus, grouped by `data-set` in release order). Label: count/
    regression-continuous (per-set averages). Whether average
    provision/power creeps upward across successive expansions — a
    power-creep sanity check.
30. **Faction Keyword Signature** — Input: multi-group (all cards of
    one faction, as a group, compared across factions). Label: count
    (distribution, per-faction keyword frequency table). Which
    keywords are over-represented in one faction relative to the
    corpus average (e.g. does Skellige skew toward `bleeding`,
    Nilfgaard toward `lock`) — a faction-identity/style summary useful
    for stratified sampling of #14/#20.
