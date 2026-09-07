# BRAINSTORM (sts_gg / Slay the Spire 2 run data, sts.gg source)

A todo/idea list of candidate metrics for sts.gg's Slay the Spire 2
run records (`data/raw/sts_gg/runs.jsonl` — see
`src/data_retrieval/sts_gg/run_downloader.py`). This is a *different*
raw source than `sts2runs` (spire-codex's run export) for the same
game — see `card_binder/spire_codex/README.md` for why this project
resolves both sources' card references through the same `CardBinder`.
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec. `legacy/sts_gg/README.md`
documents the one metric already built from this source
(`DeckOutcomeMetric`, a deliberately weak `P(win | final deck)`
baseline) — none of the ideas below repeat it as-is, though several
sharpen its framing per this project's own critique of it.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed against a live run record's full key set and nested field
samples:

- Top-level: `id`, `seed`, `character`, `ascension`, `win` (bool),
  `runTimeSeconds`, `acts` (list of act names actually reached, e.g.
  `["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"]`), `submittedAt`,
  `buildId` (game-version marker — a per-run analogue of
  `hearthstonejson`'s per-build snapshots, but stamped per-run rather
  than needing a separate file), `gameMode`, `deckSize`,
  `relicCount`, `killedBy` (null on a win), `player` (username/
  platform metadata), `maxPotionSlots`.
- `deck`: a flat list of `{id: "CARD.<id>", upgraded, floor}` — the
  `floor` field records **when each card was added**, which is what
  makes act-boundary-conditioned metrics (the ones the run's own
  known-bias section below cares about) actually buildable, not just
  a theoretical framing.
- `relics`/`potions`: `{id, floor}`/`{id, slot}` — same floor-stamped
  shape for relics; potions carry an inventory `slot` instead.
- `stats`: aggregate counters (`totalDamageTaken`,
  `totalGoldEarned`/`totalGoldSpent`, `totalCardsPicked`/
  `totalCardsSkipped`, `floorsCleared`, `totalCombats`, `totalTurns`,
  `elitesKilled`) plus a `perAct` breakdown of the same shape — a
  genuinely rich, pre-aggregated summary no manual accumulation is
  needed to reproduce.
- `bossFights`/`eliteFights`: one entry per named encounter (`name`,
  `encounterId`, `floor`, `hp`/`maxHp` after the fight, `damageTaken`)
  — per-fight granularity `sts2runs`' brainstorm doesn't have
  confirmed for its own raw shape.
- `hpPerFloor`/`turnsPerCombat`: per-floor time series (`floor`, `hp`/
  `maxHp`, `actIdx`, `type` for the former; `floor`, `turns`, `actIdx`,
  `type` for the latter) — a genuine turn-by-turn/floor-by-floor
  trajectory, richer than a single end-of-run snapshot.
- **What this source does NOT have, confirmed by its field list**:
  no `card_choices`/`relic_choices`/`potion_choices`/`event_choices`/
  `rest_site_choices` lists anywhere in the schema. Every "predict the
  pick given the offered options" metric from the `sts2runs`
  brainstorm has **no equivalent here** — this source only records
  what a run *ended up with*, never what was *offered and declined*.
  That distinction shapes this file's whole metric list: it leans
  toward deck-snapshot/outcome and per-fight/per-floor trajectory
  metrics, not draft-choice metrics.

## Known biases / known-unknowns

- **Same survivorship/self-selection bias as `sts2runs`** — this data
  comes only from players who submit runs to a tracking site, not a
  random sample of all play.
- **Naive "deck at end of run" conditioning is the exact contamination
  this project has already flagged for this source** (see
  `legacy/sts_gg/README.md`'s critique of `DeckOutcomeMetric`): an
  early-abandoned run's final deck looks close to the starting deck,
  so a model can learn "distance from starting deck" as a shortcut
  for "how far the run got" instead of learning anything about deck
  quality. This file's metrics deliberately use `floor`-stamped
  `deck`/`relics` fields to build sharper, act/floor-boundary-
  conditioned alternatives instead of repeating the naive framing.
- **No draft-choice data** — flagged above; any metric implying
  "the player chose X over Y" cannot be built from this source alone.
- **Two independent raw sources for the same underlying game**
  (`sts_gg` here, `spire_codex`'s run export in `sts2runs`) may have
  different player populations, submission biases, or even schema
  quirks — metrics here shouldn't be assumed to generalize to
  `sts2runs`'s corpus or vice versa without checking.

## Candidate Metrics

### Single-card (10)

1. **Card Win Rate, Sharply Conditioned** — Input: single-card. Label:
   probability. `P(win | card in deck_at_start_of_act_2)` — the
   direct fix for `DeckOutcomeMetric`'s flagged confound, using
   `deck`'s `floor` field to reconstruct the act-2-start deck rather
   than the naive final deck.
2. **Card Ascension-Weighted Win Rate** — Input: single-card. Label:
   probability. `P(win | card in deck_at_start_of_act_2, ascension
   >= k)` for a fixed high `k` — isolates whether a card holds up at
   harder difficulty.
3. **Card Average Floor Acquired** — Input: single-card. Label: count.
   Average `floor` at which a card is added to the deck, across all
   runs containing it — an early-staple vs. late-luxury-pick signal,
   buildable directly from the `floor`-stamped `deck` field with no
   accumulation trickery needed.
4. **Card Damage-Taken Association** — Input: single-card. Label:
   regression-continuous. Average `totalDamageTaken` (or the
   act-scoped `perAct.damageTaken`) for runs where the card was in the
   deck by that act's start, vs. the format's overall average — a
   survivability proxy.
5. **Card Boss-Fight Outcome Association** — Input: single-card.
   Label: regression-continuous. Average `damageTaken` in
   `bossFights` entries for runs where the card was in the deck at
   that boss's floor — a per-encounter power signal `sts2runs`'
   equivalent idea can only frame more speculatively (no confirmed
   per-fight granularity there).
6. **Card Elite-Fight Outcome Association** — Input: single-card.
   Label: regression-continuous. Same idea as #5, for `eliteFights`.
7. **Card Upgrade Rate** — Input: single-card. Label: probability.
   `P(card is upgraded by run end | card in final deck)`.
8. **Card Turns-Per-Combat Association** — Input: single-card. Label:
   regression-continuous. Average `turnsPerCombat` entries' `turns`
   value for combats occurring after a card was acquired, vs. before
   — an efficiency/tempo signal per card.
9. **Card Cause-of-Death Association** — Input: single-card. Label:
   probability. `P(killedBy == specific encounter/event | card absent
   from deck at that floor)` — whether lacking a card correlates with
   dying to a specific threat.
10. **Card Gold-Efficiency Association** — Input: single-card. Label:
    regression-continuous. Average `stats.totalGoldEarned` for runs
    containing the card vs. not — weak/speculative, included for
    completeness (gold economy is only loosely tied to any one card).

### Multi-card (11)

 1. **P(beat act 2 | deck at start of act 2)** — Input: multi-card
    (deck snapshot, reconstructed via `floor`). Label: probability.
    `P(win | deck_at_start_of_act_2)`.
 2. **P(beat act 3 | deck at start of act 3)** — Input: multi-card.
    Label: probability. `P(win | deck_at_start_of_act_3)`.
 3. **P(beat act 3 | deck at start of act 2)** — Input: multi-card.
    Label: probability. `P(win | deck_at_start_of_act_2)`, further
    conditioned on the run having reached act 3 at all — isolates
    deck quality from "did the run even continue," using `acts`.
 4. **P(win | deck_at_start_of_act_2, survived_to_final_boss)** —
    Input: multi-card. Label: probability. The user's own named
    example of a richer, bias-aware framing of this exact metric
    family — conditions out early-death noise entirely.
 5. **Masked-Card-in-Deck Prediction** — Input: multi-card (deck
    snapshot at a fixed floor, one card masked). Label: classification
    (variable-set, over the character's card pool). Predict the
    masked card from the rest of the deck.
 6. **Ascension Prediction from Deck Snapshot** — Input: multi-card
    (deck at a fixed floor). Label: classification (fixed-set,
    ascension levels 0-20+). Predict `ascension` from deck
    composition at that floor.
 7. **Character Prediction from Deck** — Input: multi-card (deck).
    Label: classification (fixed-set, one per playable character).
    Predict `character` from deck composition alone — a sanity-check
    baseline given character-specific card pools.
 8. **Floor-Reached Regression from Deck** — Input: multi-card (deck
    at start of act 1, i.e. essentially the starting deck plus any
    act-0 picks). Label: regression-continuous. Predict
    `stats.floorsCleared` — a softer, continuous alternative to the
    binary win/loss metrics above.
 9. **Deck-to-Relic-Count Association** — Input: multi-card (deck at a
    fixed floor). Label: count. Predict `relicCount` at that same
    floor from the deck snapshot — a "does this deck's shape correlate
    with picking up more relics" signal (e.g. more Skill-heavy decks
    lingering at shops).
10. **Deck Composition → Specific-Boss Outcome** — Input: multi-card
    (deck at the floor of a specific named boss, e.g. `THE_KIN_BOSS`).
    Label: regression-continuous. Predict `damageTaken` in that boss's
    `bossFights` entry — the matchup-specific framing `sts2runs`'
    brainstorm also proposes, buildable here with confirmed per-fight
    data.
11. **Turns-to-Clear-Act Prediction** — Input: multi-card (deck at act
    start). Label: count. Predict total turns taken to clear that
    act, summed from `turnsPerCombat` entries in that `actIdx`.
