# BRAINSTORM (play_gwent / playgwent.com community deck guides)

A todo/idea list of candidate metrics for playgwent.com's community-
submitted deck guides (`data/raw/play_gwent/guides.jsonl`, 60,138
records — see `src/data_retrieval/play_gwent/downloader.py`). None of
this is designed or built yet — this is raw material for picking what
to build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed by sampling `guides.jsonl` (first record, plus a 3,000-row
scan for distributions):

- Top-level guide fields: `id`, `name` (guide title, often non-English
  — the sampled record's title is Korean), `author`, `language`,
  `votes` (signed int, net upvotes minus downvotes — confirmed
  negative values exist, e.g. -1/-2), `published`/`new`/`invalid`
  (bools), `craftingCost`, `faction`, `leaderId`, `leader` (full
  embedded leader card object), `stratagem`, `content` (a `{"ops":
  [...]}` rich-text delta — the guide's actual write-up prose), `deck`
  (metadata: `cardsCount`, `unitsCount`, `provisionsCost`,
  `srcCardTemplates` — a flat list of card template ids), and `cards`
  — the deck's full card list as **fully embedded card objects**
  (`id`, `name`, `faction`, `cardGroup` [gold/bronze/leader],
  `power`, `provisionsCost`, `rarity`, `categoryName`/`categoryIds`,
  `type`, `armour`, `tooltip` [structured keyword+text ability
  segments, same shape as `gwent_one`'s ability text]).
- `votes` distribution (3,000-row sample): heavily right-skewed
  around 0 (1,154 rows at 0 votes, 565 at 1), with a real negative
  tail (147 rows at -1, 47 at -2) and a long positive tail (max
  observed in-sample: 51) — a genuine, if noisy, community
  quality/reception signal, unlike any vote-free source in this
  project.
- `faction` distribution (3,000-row sample): `nilfgaard` (649),
  `monsters` (603), `scoiatael` (519), `northernrealms` (511),
  `skellige` (453), `syndicate` (265) — reasonably balanced, skewed
  toward the newer/more popular factions in this sample.
- `language`: populated per-guide (e.g. `ko`), meaning this corpus is
  multilingual — a guide's `content` prose is not reliably English.
- `content.ops` is a Quill.js-style rich-text delta (a list of
  `{insert: ...}` operations) — the actual write-up text needs
  extracting from this delta format, not a plain string field.

## Known biases / known-unknowns

- **Self-selected, community-submitted guides, not a random sample of
  decks played** — guide authors are a motivated subset (people who
  write strategy content), and `votes` reflects guide-writing quality/
  popularity as much as (or more than) the underlying deck's actual
  competitive strength. A high-vote guide is not necessarily proof of
  a strong deck, and a low/negative-vote guide is not necessarily
  proof of a weak one.
- **No win/loss or ladder-rank outcome data exists in this source** —
  `votes` is the only quality-adjacent signal; it should not be
  conflated with a true win-rate label.
- **Multilingual corpus** — any text-based metric (on `content` or
  `name`) needs to either filter to one `language` or explicitly
  account for cross-language variation; treating the whole corpus as
  English-language text would be wrong.
- **`published`/`new`/`invalid` flags are unverified in their exact
  semantics from this review alone** — e.g. whether `invalid` means
  "removed by moderation" or "author-retracted" isn't confirmed;
  treat as a filter to investigate before relying on it, not to
  assume.
- **`craftingCost` reflects in-game currency cost to build the deck,
  not a power-level measure** — higher cost often just means more
  legendary/epic cards, not necessarily a stronger build.

## Candidate Metrics

### Single-card (10)

1. **Card Inclusion Rate** — Input: single-card. Label: probability.
   `P(card in deck | card is faction-legal for the deck)` — the base
   popularity rate this corpus can measure at real (if guide-biased)
   scale, unlike `gwent_one`'s card-catalog-only source.
2. **Card Vote-Weighted Popularity** — Input: single-card. Label:
   regression-continuous. Average guide `votes` across decks
   containing the card, vs. the corpus-wide average — surfaces cards
   associated with well-received guides specifically.
3. **Card Inclusion Rate by Faction** — Input: single-card. Label:
   probability. `P(card in deck | faction)` — a per-faction play-rate
   stat, comparable to `pitchstack`'s hero-conditioned idea but with
   full card visibility here.
4. **Card Provision-Efficiency Signal** — Input: single-card. Label:
   regression-continuous. Deck-average `votes` per unit of the card's
   own `provisionsCost` — a rough "is this card's provision cost
   justified" proxy.
5. **Card Recency/Language Popularity Split** — Input: single-card.
   Label: probability. `P(card in deck | language)` — whether card
   popularity differs across the game's regional communities
   represented in this corpus.
6. **Leader-Conditioned Card Inclusion** — Input: single-card. Label:
   probability. `P(card in deck | leaderId)` — the `play_gwent`
   analogue of `pitchstack`'s hero-conditioned rate, using the deck's
   specific leader rather than just faction.
7. **Card Copy-Count Distribution** — Input: single-card. Label:
   count (distribution). How many copies of a card guide decks
   typically run (Gwent decks generally cap at low copy counts per
   card, worth confirming against `repeatCount`/deck structure).
8. **Card Vote Sign Association** — Input: single-card. Label:
   probability. `P(guide has positive votes | card in deck)` — a
   coarser, sign-only version of #2, more robust to the corpus's long
   tail/outliers.
9. **Card Mention-in-Guide-Text Rate** — Input: single-card. Label:
   probability. `P(card's name appears in content.ops prose | card in
   deck)` — whether a card gets explicitly discussed by the guide
   author vs. just included silently; needs the delta-format text
   extraction noted above and is language-dependent per the corpus's
   multilingual nature.
10. **Rarity-vs-Vote Association** — Input: single-card. Label:
    regression-continuous. Average guide `votes` for decks, split by
    whether the card is legendary/epic/rare/common — whether flashier
    (higher-rarity) cards correlate with better-received guides.

### Multi-card (11)

11. **Guide Vote Prediction from Deck** — Input: multi-card (full
    `cards` list). Label: regression-continuous. Predict a guide's
    `votes` from its deck composition alone — the play_gwent analogue
    of `sts2runs`'s ascension-prediction idea, with the important
    caveat (see biases above) that this predicts guide reception, not
    competitive win rate.
12. **Masked Card-in-Deck Prediction** — Input: multi-card (deck with
    one card masked). Label: classification (variable-set, over the
    faction-legal card pool). Standard masking task, same family as
    `fabtcg_decklists`'/`pitchstack`'s versions but with guaranteed
    card-level data (no download-blocked gap here).
13. **Leader Prediction from Deck** — Input: multi-card (non-leader
    cards of a deck). Label: classification (fixed-set, over
    leaders). Given the rest of the deck, predict `leaderId` — this
    source's Commander-guessing analogue, directly comparable to
    `fabtcg_decklists`'s masked-Hero idea.
14. **Deck-Pair Same-Faction Classification** — Input: multi-card (two
    decks). Label: probability. `P(same_faction | deck_1, deck_2)` —
    a sanity-check baseline, same role as other sources' equivalent
    idea.
15. **Card Co-Occurrence / Synergy Statistic** — Input: multi-card
    (decks containing card A vs. card B, as two groups). Label:
    probability. `P(card_B in deck | card_A in deck)` — real deck
    co-occurrence data, comparable to `fabtcg_decklists`'s version but
    for Gwent and at much larger scale (60K decks vs. 1.7K).
16. **Gold/Bronze Ratio Prediction** — Input: multi-card (partial
    deck, e.g. just the bronze cards). Label: regression-continuous.
    Predict the deck's overall gold-to-bronze card-count ratio from a
    partial view — Gwent's deck-construction budget makes this a
    genuinely structured target (unlike a freeform TCG).
17. **Provision-Curve Prediction** — Input: multi-card (partial deck).
    Label: count (distribution). Predict the full deck's provision-
    cost curve (histogram across cost buckets) from a partial subset.
18. **High-Vote vs. Low-Vote Deck Discrimination** — Input: multi-card
    (two decks, same faction, one from the corpus's top-vote decile
    and one from the bottom). Label: probability. `P(deck_1 is the
    higher-voted deck | deck_1, deck_2)` — reframes #11's regression
    as a more robust pairwise-ranking task.
19. **Category/Tribe Composition Prediction** — Input: multi-card
    (partial deck). Label: count (distribution). Predict the full
    deck's tribal/category mix (`categoryName`, e.g. Human/Witcher/
    Beast counts) from a partial subset.
20. **Stratagem Prediction from Deck** — Input: multi-card (full deck,
    stratagem masked). Label: classification (fixed-set, over
    stratagems). Predict a deck's chosen `stratagem` from its card
    list.
21. **Cross-Language Deck Similarity** — Input: multi-card (two decks
    with guides in different `language`s). Label: probability.
    `P(decks are near-duplicate builds | deck_1, deck_2, different
    languages)` — whether the same underlying archetype gets
    independently rediscovered/reposted across language communities.

### Multi-group (9)

22. **Faction Metagame Share Over Time** — Input: multi-group (all
    guides in a faction, bucketed by `modified` time). Label: count
    (distribution). How a faction's guide-submission share shifts
    over time — a metagame-drift signal at community-content scale.
23. **Vote-Weighted Archetype Discovery** — Input: multi-group (all
    decks for one leader, clustered by card-set similarity, each
    cluster a group). Label: classification (variable-set, over
    discovered clusters). An unsupervised framing surfacing candidate
    archetypes per leader, weighted by `votes` to favor well-received
    exemplars — comparable to `fabtcg_decklists`'s archetype-discovery
    idea but with a real quality signal to weight by.
24. **Leader Popularity by Faction and Language** — Input: multi-group
    (leader counts, jointly grouped by faction and language). Label:
    count (distribution). A three-way breakdown of leader
    representation, analogous to `pitchstack`'s hero-by-format-by-tier
    idea.
25. **Guide-Quality-Conditioned Card Frequency Table** — Input:
    multi-group (decks split into vote-quartile groups). Label: count
    (distribution, per-quartile card frequency). Which cards are
    over-represented in top-vote-quartile decks vs. bottom-quartile —
    the vote-scale version of #2, exposing per-card signal more
    robustly than a raw average.
26. **Cross-Faction Card Portability** — Input: multi-group (a
    neutral/secondary-faction-eligible card's appearances, grouped by
    faction). Label: probability. `P(card appears across >= k
    factions | card)` — using `secondaryFactions`, confirmed present
    on embedded card objects.
27. **Guide Text Topic Clustering per Archetype** — Input: multi-group
    (guide `content` text for decks in the same discovered archetype
    cluster, as a group). Label: classification (variable-set, over
    discovered topics). What authors tend to write about for a given
    archetype (combo explanation, matchup advice, mulligan guide) —
    speculative, needs the delta-to-text extraction and is
    English-language-only unless multilingual text handling is added.
28. **Deck Similarity Network Density by Faction** — Input:
    multi-group (all decks in one faction, pairwise similarity as an
    implicit graph). Label: count/regression-continuous. How
    clustered vs. diverse one faction's submitted decks are relative
    to another — a metagame-diversity index, the play_gwent analogue
    of `pitchstack`'s field-diversity idea but at far larger scale.
29. **Language-Community Card Preference Divergence** — Input:
    multi-group (decks split by `language`, each language's card-
    frequency table a group). Label: regression-continuous (a
    divergence score, e.g. KL-divergence between language groups'
    card distributions). Whether regional communities favor
    meaningfully different cards/archetypes for the same faction.
30. **Guide Validity/Publication Filtering Impact** — Input:
    multi-group (guides split by `published`/`new`/`invalid` flag
    combinations, each combination a group). Label: count
    (distribution). How large each flag-combination bucket is and
    how their `votes`/deck-composition distributions differ — a
    data-quality-filtering study to run before trusting any of the
    metrics above on the full unfiltered corpus.
