# CURRENT_STATUS

A snapshot (2026-10-02, `main` at `ef95fae`) of what each
`data_retrieval` source has in `data_refinement` and the dojo catalog.
Update a row when its status changes. Bugs and chores live in each
container's own `TODO.md`. The prioritized pre-training work list is
`plans/pre_training_data.md` (local only; `plans/` is gitignored).

Legend: ✅ exists · ❌ missing · 🟥 brainstorm only · 🟨 partial ·
🟩 complete or mature. "Keys" counts `dojo_catalog.py` entries, which
are what a training config can name.

## Totals

- **Card binders:** 7 games (MTG, Pokemon, FaB, Gwent, StS2, Dominion,
  Hearthstone).
- **Published deck boxes:** 6 games, all but Hearthstone, which has no
  deck source.
- **Metrics:** 150 classes, 148 with a dojo
  (`docs/metric_dojo_inventory.csv`).
- **Catalog:** 151 keys. The first 123 each passed `run_training.py
  --check` on real data when added (2026-09-30 to 10-01);
  `sts_gg.card_character_prediction` and `cross_game.rarity_tier` passed
  on 2026-10-07. The 26 17lands keys were added 10-06; the four MTG deck
  keys wait on the multiset deck box and metric rebuild.

## Raw data was re-downloaded on 2026-10-02

Every `data/raw/` source was re-fetched today. **The binders, deck
boxes and metrics on disk were built from the older files.** Differences
that matter:

- **scryfall:** a newer dump (`oracle-cards-20261002...`, was 09-12).
- **spire_codex runs:** 40 pages / 6.1 GB (34 pages / 5.2 GB ingested).
- **play_gwent:** guides re-downloading in progress.
- **sts2runs:** `data/raw/sts2runs/` is **empty**. The download failed
  on a DNS error (`sts2runs.com`). Its 6,796 decks are still in
  `slay_the_spire_2.db` and in the `sts2_runs` metrics.
- **17lands:** none of the 303 CSVs is tar-wrapped any more. The old
  download had 19.

A binder re-ingest that changes `version_for(game)` invalidates that
game's deck box, metrics and splits. The refresh order is in the plan.

## Per-source matrix

| Source (retrieval) | Game | Card binder | Deck box | Metrics | Keys |
| --- | --- | --- | --- | --- | --- |
| `scryfall` | MTG | ✅ `ScryfallCardIngestionStage` | n/a (cards only) | 🟨 6 single-card masks (cmc, type, rarity, colors, power, toughness) | 6 |
| `seventeenlands` › `game_data` | MTG | (scryfall) | ✅ `seventeenlands_game_data` → `mtg.db`: every distinct decklist any game played (multiset hash; rebuild pending 2026-10-06, count TBD) | 🟨 11 built, all with dojos, all vectorized (chunk scan, slices 1-2). Partial run: 76 set/format dirs, no PremierDraft | 0 (outputs not merged) |
| `seventeenlands` › `draft_data` | MTG | (scryfall) | n/a (picks, not decks) | 🟨 6 built with dojos. Only one shakeout file run (OM1) | 0 |
| `seventeenlands` › `replay_data` | MTG | (scryfall) | n/a (same games as game_data) | 🟨 9 built with dojos. Only one shakeout file run (PIO; Arena ids miss the binder) | 0 |
| `pokemon_tcg` | Pokemon | ✅ `PokemonTcgCardIngestionStage` | ✅ `pokemon_tcg` → `pokemon.db`, 188 theme decks (prefabs, low value) | 🟨 5 single-card masks (HP, types, stage, retreat cost, weakness) | 5 |
| `hearthstonejson` | Hearthstone | ✅ `HearthstoneJsonCardIngestionStage` (newest build only, 6,187 collectible cards) | n/a (cards only) | 🟨 8 single-card masks (cost, attack, health, class, rarity, type, races, spell school) | 8 |
| `gwent_one` | Gwent | ✅ `GwentOneCardIngestionStage` | n/a (cards only) | 🟨 8 single-card masks | 8 |
| `play_gwent` | Gwent | (gwent_one) | ✅ `play_gwent` → `gwent.db`, 60k guide decks | 🟨 4: leader masked from deck, card inclusion rate, faction-conditioned inclusion, guide votes. All four only read the published box | 4 |
| `spire_codex` (cards) | StS2 | ✅ `SpireCodexCardIngestionStage` | n/a | 🟨 4 single-card masks (cost, type, rarity, color) | 4 |
| `spire_codex` (runs) + `sts2runs` | StS2 | (spire_codex) | ✅ `spire_codex_runs` (subclasses the sts2runs stage) + `sts2runs` → `slay_the_spire_2.db`, ~2.76M decks incl. abandoned runs | 🟩 24 in `metrics/sts2_runs/` over 1.36M scored runs, with losses. `card_reward_pick` (per-floor, B2) has a dojo and catalog key (`sts2_runs.card_reward_pick`, not yet scanned on the full corpus); the other per-floor metrics are not | 23 |
| `sts_gg` | StS2 | (spire_codex) | ✅ `sts_gg` (1,004 decks, into the same box) | 🟩 24 built with dojos. Wins only (a leaderboard), so the 4 win/killed-by labels are constant and have no key; `sts2_runs` covers them. `card_character_prediction` has a key (2026-10-07) | 20 |
| `cardvault_fabtcg` | FaB | ✅ `CardVaultFabtcgCardIngestionStage` | n/a (cards only) | 🟨 6 single-card masks (pitch, cost, power, defense, class, type) | 6 |
| `fabtcg_decklists` | FaB | (cardvault_fabtcg) | ✅ `fabtcg_decklists` → `flesh_and_blood.db`, 4,161 decks (cards keyed by name) | 🟨 3: hero masked from deck, card inclusion rate, hero-conditioned inclusion. Pitch-curve shape not built | 3 |
| `pitchstack` | FaB | (cardvault_fabtcg) | ❌ Blocked: `decks.jsonl` (1,686 decks) has metadata only; card lists need the unimplemented `/cards` endpoint | 🟥 brainstorm only | 0 |
| `dominiontabs` | Dominion | ✅ `DominionTabsCardIngestionStage` | n/a (cards only) | 🟩 all 3 viable ideas (cost regression, set, type) | 3 |
| `isotropic` | Dominion | (dominiontabs) | ✅ `isotropic` → `dominion.db`, 531,675 final decks (resigned players skipped) | 🟩 24 (13 summary, 11 games), re-run 2026-10-01. 22 with keys; `copies_bought_distribution` and `multiplayer_placement` have no dojo. Resignation metrics not built | 22 |
| `dominion/` | Dominion | Not a downloader (research notes and a prototype scraper) | n/a | n/a | n/a |
| none | Yu-Gi-Oh | ❌ `GameId.YUGIOH` exists, but there is no retrieval source | n/a | n/a | n/a |

Cross-source:

- **`final_decks`:** a held-out card metric over each published deck
  box (Pokemon, FaB, Gwent, Dominion, StS2, MTG): pick the held-out
  card from 8 candidates. 6 keys, outputs on disk. Splits are by deck, so
  a deck's several held-out rows never straddle TRAIN and TEST.
- **`cross_game` rarity tier:** `RarityTierMetric` labels MTG, Pokemon,
  Hearthstone, Gwent, FaB and StS2 cards on a shared four-step ladder
  (Dominion has none). One parquet for all games, regenerated 2026-10-07
  against the re-ingested MTG and Pokemon binders (each stores its lowest
  print rarity); trained by `RarityTierDojo` (key
  `cross_game.rarity_tier`, one head, TRAIN drawn evenly across games).
  The other MTG and Pokemon metrics, decks and splits still carry the old
  binder versions and are stale until regenerated.
- **`contrastive`:** one deck-contrastive key per deck-box game. 6 keys,
  no metric needed.

## Mods (`src/dojos/mods/`, `src/dojos/augmentation_defaults.py`)

- **Card-field augmentations:** `ShuffleKeysMod`, `RandomKeyMaskMod` and
  `WeightedFieldMaskMod`, with per-game defaults for all 7 games. They
  are applied to every catalog dojo, TRAIN only, after its own task
  mods. `mods:` in a run config replaces them.
- **Task mods:** `MaskTargetKeyMod`, `ShuffleDeckMod`, and
  `GroupSwapMod` (for symmetric deck pairs).
- **Deck thinning (opt-in):** `CardDropoutMod`, `CardSubsampleMod` and
  `DuplicateCollapseMod`, allowed per dojo and per group by
  `DECK_MOD_GROUPS` (14 dojos today).
- **Contrastive staple subsampling:** keep probability
  `min(1, sqrt(t / df))`, set per dojo with `staple_subsampling:`. The
  default `t = inf` is off; the ablation has not run.

## Open work, in priority order

The detail is in `plans/pre_training_data.md`. In short:

1. **Code fixes before any rebuild:** done 2026-10-02. Binder ingestion
   seeds the Unknown sentinel, `play_gwent` no longer writes `gwent.db`,
   and the held-out-card and isotropic per-kingdom dojos split by group.
2. **17lands:** vectorize draft_data and replay_data (slices 3-4), build the per-metric
   merge step, run the full corpus, add catalog keys.
3. **Refresh from the 10-02 re-download:** a scratch binder version
   check per game, then rebuild only the games that changed (Gwent and
   the StS2 spire_codex runs for sure; MTG only as one combined job).
4. **Metric breadth:** StS2 per-floor picks (B2), cross-game rarity, and
   the small dojo and catalog gaps above.
5. **Final gate:** regenerate the inventory, delete stale splits, and
   preflight every key.

Blocked or parked: pitchstack (needs `/cards`), Yu-Gi-Oh (no source),
older Hearthstone builds.
