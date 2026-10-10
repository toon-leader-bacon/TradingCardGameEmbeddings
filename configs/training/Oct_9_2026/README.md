# Series 1: model × diet (first draft, 2026-10-09)

Six run configs, one per cell of `EXPERIMENTS_TODO.md` series 1:

| | contrastive only | metric dojos only | 50/50 mixed |
|---|---|---|---|
| **single-card** | `single_contrastive.yaml` | `single_metric.yaml` | `single_mixed.yaml` |
| **multi-card** | `multi_contrastive.yaml` | `multi_metric.yaml` | `multi_mixed.yaml` |

Each config is self-contained. A single-card config and its multi-card
twin differ only in `model:` and the run directory, so each column
compares the two models on identical dojos and diets.

The `dojos:` lists are patterns per source (`"isotropic.*"`), so a dojo
added to the catalog later joins these configs. Each run's
`run_config.yaml` records the expanded names, so a finished run still
says exactly what it trained on.

## Shared by every cell

- **Model:** frozen ModernBERT-base, `residual_mlp` embedding head, 256-wide
  card embeddings, about 2.9M trainable parameters. Single-card: 9 MLP
  blocks. Multi-card: 3 MLP blocks plus 2 group-attention layers
  (feed-forward 1024, pre-norm).
- **Contrastive dojos:** all five training styles (single card,
  cross-slice card match, slice match, missing card, odd one out) for
  every game except Gwent (Dominion, Flesh and Blood, MTG, Pokemon, Slay
  the Spire 2): 25 dojos, the same for both models. The diet draws each
  style equally often, then each game equally. Card in contexts is left
  out: it is an evaluation probe (`src/evaluation/TODO.md`).
- **Metric dojos:** the 124 catalog keys that are not contrastive, not
  Gwent and not held out, weighted by TRAIN count ** 0.3.
- **Mixed diet:** a table diet, half contrastive (split by style as
  above) and half metric.
- **Holdout:** card seed 0, tiers 8/1/1. Gwent is a held-out game: every
  Gwent card is VALIDATION-tier, and no Gwent dojo is in any run.
- **Held-out dojos** (never trained, the same 13 in every cell):
  `cross_game.rarity_tier` (the intrinsic label set) plus 12 for extrinsic
  evaluation, spread across games and input shapes:
  - single-card: `scryfall.cmc_regression`, `hearthstonejson.class_mask`,
    `pokemon_tcg.hp_regression`, `cardvault_fabtcg.pitch_mask`,
    `spire_codex.card_type_mask`, `sts2_runs.card_win_rate` (and its
    sibling `card_win_rate_at_act2`, so it cannot leak through training),
    `seventeenlands_replay_data.average_turn_cast`;
  - multi-card: `seventeenlands_game_data.deck_win_prediction`,
    `isotropic.kingdom_veto_prediction`, `sts2_runs.card_reward_pick`,
    `fabtcg_decklists.hero_masked_from_deck`.
- **Budget:** 20 rounds × 1,000 steps = 20,000 steps. `patience_rounds`
  is above `max_rounds`, so no dojo saturates: the diet never changes and
  no cell ends early.
- **Loss:** Huber for regression dojos; every dojo's loss divided by its
  baseline (the default loss weighting).
- **Batches:** `max_batch_cost: 256` cards (the baseline survey's
  setting, which every key passed); fp16; 128 TEST examples per dojo per
  round.

## Open before the first run

- **Pilot run.** Step time, round time (the TEST pass covers up to 142
  dojos), VRAM at 256 cards and RAM with ~140 dojos built have not been
  measured. They decide whether 20,000 steps and `max_batch_cost: 256`
  hold.
- **`--check` on a mixed config.** `single_mixed.yaml` holds every dojo
  any cell uses; one passing check covers them all (plus building each
  model once).
- **Draft choices to confirm:** the held-out list, Huber over MSE,
  alpha 0.3 for the metric dojos, uniform over contrastive games (the
  Dominion staple caveat in `src/training/TODO.md` still applies), and
  the learning rates (1e-3, from the earlier runs).
