"""Runs src/data_refinement/metrics scanners — one raw-source metric
family at a time, or all of them — each converting already-ingested
CardBinder + raw source data into training-data parquet files under
data/metrics/.

Unlike run_card_binder_ingestion.py/run_deck_box_ingestion.py, this
container isn't one uniform shape (see src/data_refinement/metrics/
README.md): each family below is a direct, runnable copy of that
family's own README "How to run" recipe — sts_gg, gwent_one,
dominiontabs, and play_gwent each read exactly one fixed raw
file/corpus; isotropic_summary and isotropic_games each pool every
matching archive under data/raw/isotropic/ into one output per metric
(--raw-path narrows that to one archive), sharing one warm-started
metrics-private deck box; while seventeenlands's three families (draft_data/game_data/replay_data)
each have dozens of expansion x format CSVs
(data/raw/17lands/<family>/<Expansion>.<Format>.csv). For those three,
`--all` (or omitting --raw-path) scans every CSV found; every metric's
own DEFAULT_OUTPUT_PATH is namespaced by <expansion>/<format_code> so
scanning multiple sets never overwrites a previous one's output — the
metric classes themselves have no cross-file accumulation (each file's
header names different cards), so "one file, one output" is the actual
unit of work, not "one family, one output". sts2_runs reads two
fixed sources (spire_codex's ~1.7M-run pages and sts2runs' dump) in one
pass; it is the slow one (most of an hour).

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_metrics.py --list
    PYTHONPATH=. python3 scripts/run_metrics.py --source sts_gg
    PYTHONPATH=. python3 scripts/run_metrics.py --source seventeenlands_game_data
    PYTHONPATH=. python3 scripts/run_metrics.py --source seventeenlands_game_data \\
        --raw-path "data/raw/17lands/game_data/MSH.PremierDraft.csv"
    PYTHONPATH=. python3 scripts/run_metrics.py --all

`--all` runs every family in this process, one after another (same
reasoning as the other two scripts' `--all` — nothing here is a
multi-hour crawl, though sts2_runs takes most of an hour). One family's
failure is logged and does not stop the rest.
"""

import argparse
from functools import partial
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.sts_gg.extraction_stage import (
    StsGgDeckExtractionStage,
)
from src.data_refinement.metrics.generic.corpus_scan_metric import CorpusScanMetric
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.rowwise_metric import RowwiseMetric
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.schema.game_id import GameId

# --- sts_gg ---
from src.data_refinement.metrics.isotropic.deck_box_path import ISOTROPIC_DECK_BOX_PATH
from src.data_refinement.metrics.sts_gg.ascension_prediction_metric import (
    AscensionPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardDeckSizeMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardRelicCountMetric,
    CardTotalCardsPickedMetric,
    CardTotalCombatsMetric,
    CardTotalDamageTakenMetric,
    CardTotalTurnsMetric,
    CardWinRateMetric,
)
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.card_upgrade_rate_metric import (
    CardUpgradeRateMetric,
)
from src.data_refinement.metrics.sts_gg.card_win_rate_at_act2_metric import (
    CardWinRateAtAct2Metric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    ElitesKilledMetric,
    FloorsClearedMetric,
    KilledByMetric,
    RelicCountMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalCombatsMetric,
    TotalDamageTakenMetric,
    TotalTurnsMetric,
    WinMetric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric as DeckCharacterPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.deck_box_path import STS_GG_DECK_BOX_PATH
from src.data_refinement.metrics.sts_gg.scanner import scan_runs_jsonl

# --- final_decks ---
from src.data_refinement.metrics.final_decks.held_out_card_metrics import (
    DominionHeldOutCardMetric,
    FleshAndBloodHeldOutCardMetric,
    GwentHeldOutCardMetric,
    MtgHeldOutCardMetric,
    PokemonHeldOutCardMetric,
    SlayTheSpire2HeldOutCardMetric,
)
from src.data_refinement.metrics.generic.held_out_deck_card.metric import (
    HeldOutDeckCardMetric,
)

# --- sts2_runs ---
from src.data_refinement.metrics.sts2_runs import card_average_metrics as sts2_cards
from src.data_refinement.metrics.sts2_runs import deck_label_metrics as sts2_decks
from src.data_refinement.metrics.sts2_runs.run_record import Sts2Run
from src.data_refinement.metrics.sts2_runs.scanner import (
    default_run_sources,
    scan_sts2_runs,
)

# --- gwent_one ---
from src.data_refinement.metrics.gwent_one.armor_mask_metric import ArmorMaskMetric
from src.data_refinement.metrics.gwent_one.color_mask_metric import ColorMaskMetric
from src.data_refinement.metrics.gwent_one.faction_mask_metric import (
    FactionMaskMetric,
)
from src.data_refinement.metrics.gwent_one.power_mask_metric import PowerMaskMetric
from src.data_refinement.metrics.gwent_one.provision_mask_metric import (
    ProvisionMaskMetric,
)
from src.data_refinement.metrics.gwent_one.rarity_mask_metric import RarityMaskMetric
from src.data_refinement.metrics.gwent_one.set_mask_metric import SetMaskMetric
from src.data_refinement.metrics.gwent_one.type_mask_metric import TypeMaskMetric

# --- dominiontabs ---
from src.data_refinement.metrics.dominiontabs.cost_regression_metric import (
    CostRegressionMetric,
)
from src.data_refinement.metrics.dominiontabs.set_mask_metric import (
    SetMaskMetric as DominionSetMaskMetric,
)
from src.data_refinement.metrics.dominiontabs.type_mask_metric import (
    TypeMaskMetric as DominionTypeMaskMetric,
)

# --- single-card masks: scryfall, pokemon_tcg, cardvault_fabtcg, spire_codex ---
from src.data_refinement.metrics.cardvault_fabtcg import (
    card_mask_metrics as fabtcg_masks,
)
from src.data_refinement.metrics.hearthstonejson import (
    card_mask_metrics as hearthstone_masks,
)
from src.data_refinement.metrics.pokemon_tcg import card_mask_metrics as pokemon_masks
from src.data_refinement.metrics.scryfall import card_mask_metrics as scryfall_masks
from src.data_refinement.metrics.spire_codex import card_mask_metrics as sts2_masks

# --- isotropic/summary ---
from src.data_refinement.metrics.isotropic.summary.average_copies_bought_metric import (
    AverageCopiesBoughtMetric,
)
from src.data_refinement.metrics.isotropic.summary.copies_bought_distribution_metric import (  # noqa: E501
    CopiesBoughtDistributionMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_card_mask_metric import (
    WinningDeckMaskedCardMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_card_set_copy_count_metric import (  # noqa: E501
    DeckCardSetCopyCountMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_pair_winner_metric import (
    DeckPairWinnerMetric,
)
from src.data_refinement.metrics.isotropic.summary.full_deck_win_prediction_metric import (  # noqa: E501
    FullDeckWinPredictionMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_game_length_metric import (
    KingdomGameLengthMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_member_label_metrics import (
    WinningDeckCountMetric,
    WinningDeckMembershipMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_veto_prediction_metric import (  # noqa: E501
    KingdomVetoPredictionMetric,
)
from src.data_refinement.metrics.isotropic.summary.multiplayer_placement_metric import (
    MultiplayerPlacementMetric,
)
from src.data_refinement.metrics.isotropic.summary.scanner import (
    scan_isotropic_summary_archives,
)
from src.data_refinement.metrics.isotropic.summary.turn_count_association_metric import (  # noqa: E501
    TurnCountAssociationMetric,
)
from src.data_refinement.metrics.isotropic.summary.veto_rate_metric import (
    VetoRateMetric,
)

# --- isotropic/games ---
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog
from src.data_refinement.metrics.isotropic.games.header_parser import GameHeader
from src.data_refinement.metrics.isotropic.games.kingdom_ending_pile_prediction_metric import (  # noqa: E501
    KingdomEndingPilePredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.kingdom_ending_type_metric import (
    KingdomEndingTypeMetric,
)
from src.data_refinement.metrics.isotropic.games.kingdom_opening_buy_prediction_metric import (  # noqa: E501
    KingdomOpeningBuyPredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_deck_pair_winner_metric import (  # noqa: E501
    MidGameDeckPairWinnerMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_buy_metric import (
    NextBuyPredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_trashed_card_metric import (  # noqa: E501
    NextTrashedCardMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_turn_action_count_metric import (  # noqa: E501
    NextTurnActionCountMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_win_probability_metric import (  # noqa: E501
    EventualWinProbabilityMetric,
)
from src.data_refinement.metrics.isotropic.games.opening_buy_outcome_metric import (
    OpeningBuyOutcomeMetric,
)
from src.data_refinement.metrics.isotropic.games.opening_buy_rate_metric import (
    OpeningBuyRateMetric,
)
from src.data_refinement.metrics.isotropic.games.pile_exhaustion_rate_metric import (
    PileExhaustionRateMetric,
)
from src.data_refinement.metrics.isotropic.games.scanner import (
    scan_isotropic_game_log_archives,
    scan_isotropic_game_logs_archives,
)

# --- fabtcg_decklists ---
from src.data_refinement.metrics.fabtcg_decklists.card_inclusion_metrics import (
    CardInclusionRateMetric as FabCardInclusionRateMetric,
    HeroConditionedInclusionMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.hero_masked_from_deck_metric import (
    HeroMaskedFromDeckMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.scanner import (
    DEFAULT_RAW_PATH as FABTCG_DECKLISTS_DEFAULT_RAW_PATH,
    scan_decklist_files,
)

# --- play_gwent ---
from src.data_refinement.metrics.play_gwent.leader_deck_counts import (
    DEFAULT_RAW_PATH as PLAY_GWENT_DEFAULT_RAW_PATH,
)
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.data_refinement.metrics.play_gwent.card_inclusion_metrics import (
    CardInclusionRateMetric as GwentCardInclusionRateMetric,
    FactionConditionedInclusionMetric,
)
from src.data_refinement.metrics.play_gwent.guide_votes_metric import (
    GuideVotesMetric,
)
from src.data_refinement.metrics.play_gwent.scanner import scan_guides_jsonl

# --- seventeenlands/draft_data ---
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (
    PackToPickChoiceSetMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (
    PickNumberDecayCurveMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (
    PoolConditionedPickMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.scanner import (
    scan_draft_csv,
)

# --- seventeenlands/game_data ---
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import (
    scan_game_csv,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric as GameTutorTargetRateMetric,
)

# --- seventeenlands/replay_data ---
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.cast_rate_metric import (
    CastRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (  # noqa: E501
    CombatAggressionProfileMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.discard_rate_metric import (
    DiscardRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.scanner import (
    scan_replay_csv,
)
from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (  # noqa: E501
    TurnsToGameEndAfterCastMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric as ReplayTutorTargetRateMetric,
)

_MTG_BINDER_HINT = (
    "run 'python3 scripts/run_card_binder_ingestion.py --source scryfall' first."
)


def _require_binder(source_game: GameId, hint: str) -> CardBinder:
    binder_path = CardBinder.default_output_path(source_game)
    if not binder_path.exists():
        raise SystemExit(f"{binder_path} does not exist — {hint}")
    return CardBinder.load([binder_path])


# --- sts_gg ---


# Every sts_gg metric class now takes the same (card_binder, deck_box,
# output_path) shape - CardAverageMetric's own deck_box is unused but
# accepted for this exact reason (see its docstring) - so building the
# whole family is one call per class against the same two arguments,
# rather than a hand-picked arg list per metric.
_STS_GG_METRIC_CLASSES: tuple[Any, ...] = (
    CardUpgradeRateMetric,
    CardWinRateAtAct2Metric,
    AscensionPredictionMetric,
    RelicCountMetric,
    DeckCharacterPredictionMetric,
    TotalDamageTakenMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalTurnsMetric,
    ElitesKilledMetric,
    FloorsClearedMetric,
    TotalCombatsMetric,
    KilledByMetric,
    WinMetric,
    CardRelicCountMetric,
    CardTotalDamageTakenMetric,
    CardDeckSizeMetric,
    CardTotalCardsPickedMetric,
    CardTotalTurnsMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardTotalCombatsMetric,
    CardWinRateMetric,
    CardCharacterPredictionMetric,
)


def run_sts_gg(raw_path: Path | None) -> None:
    effective_raw_path = raw_path or StsGgDeckExtractionStage.DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.SLAY_THE_SPIRE_2,
        "run 'python3 scripts/run_card_binder_ingestion.py --source spire_codex' first.",
    )
    deck_box = DeckBox()  # metrics-private — see sts_gg/README.md's "Deck references"

    metrics: list[Metric[dict]] = [
        cls(binder, deck_box) for cls in _STS_GG_METRIC_CLASSES
    ]

    print(f"=== sts_gg: {effective_raw_path} ===")
    scan_runs_jsonl(effective_raw_path, metrics)
    deck_box.save(
        STS_GG_DECK_BOX_PATH,
        GameId.SLAY_THE_SPIRE_2,
        binder.version_for(GameId.SLAY_THE_SPIRE_2),
    )
    print(f"wrote {len(metrics)} metric outputs")


# --- sts2_runs ---


# Every sts2_runs metric takes (card_binder, output_path=None)
_STS2_RUNS_METRIC_CLASSES: tuple[
    type[sts2_decks.DeckLabelMetric] | type[sts2_cards.CardAverageMetric], ...
] = (
    sts2_decks.AscensionPredictionMetric,
    sts2_decks.CharacterPredictionMetric,
    sts2_decks.WinMetric,
    sts2_decks.KilledByMetric,
    sts2_decks.RelicCountMetric,
    sts2_decks.TotalDamageTakenMetric,
    sts2_decks.TotalCardsPickedMetric,
    sts2_decks.TotalCardsSkippedMetric,
    sts2_decks.TotalTurnsMetric,
    sts2_decks.ElitesKilledMetric,
    sts2_decks.FloorsClearedMetric,
    sts2_decks.TotalCombatsMetric,
    sts2_cards.CardRelicCountMetric,
    sts2_cards.CardTotalDamageTakenMetric,
    sts2_cards.CardDeckSizeMetric,
    sts2_cards.CardTotalCardsPickedMetric,
    sts2_cards.CardTotalTurnsMetric,
    sts2_cards.CardElitesKilledMetric,
    sts2_cards.CardFloorsClearedMetric,
    sts2_cards.CardTotalCombatsMetric,
    sts2_cards.CardWinRateMetric,
    sts2_cards.CardUpgradeRateMetric,
    sts2_cards.CardWinRateAtAct2Metric,
)


def run_sts2_runs(raw_path: Path | None) -> None:
    """Scan spire_codex's run pages and sts2runs' dump (both at their
    default paths) into data/metrics/sts2_runs/. The deck-level outputs
    point into the published deck box (data/final/decks/
    slay_the_spire_2.db); nothing is written to it."""
    _reject_raw_path("sts2_runs", raw_path)
    binder = _require_binder(
        GameId.SLAY_THE_SPIRE_2,
        "run 'python3 scripts/run_card_binder_ingestion.py --source spire_codex' first.",
    )
    metrics: list[Metric[Sts2Run]] = [cls(binder) for cls in _STS2_RUNS_METRIC_CLASSES]

    print("=== sts2_runs: spire_codex run pages + sts2runs dump ===")
    tally = scan_sts2_runs(default_run_sources(), binder, metrics)
    print(tally.as_dict())
    print(f"wrote {len(metrics)} metric outputs")


# --- gwent_one ---


def _reject_raw_path(name: str, raw_path: Path | None) -> None:
    """Exit if --raw-path was given to a family whose metrics read fixed
    inputs (an already-loaded CardBinder or DeckBox, or sts2_runs' two
    run sources) rather than one raw file."""
    if raw_path is not None:
        raise SystemExit(
            f"{name} metrics read fixed inputs, not one raw file "
            "— --raw-path is not applicable."
        )


def _scan_corpus_metrics(name: str, metrics: Sequence[CorpusScanMetric]) -> None:
    """scan() each CorpusScanMetric in turn, logging (not raising) any
    one metric's failure so the rest still run."""
    print(f"=== {name} ===")
    failed: list[str] = []
    for metric in metrics:
        try:
            metric.scan()
        except Exception:
            traceback.print_exc()
            failed.append(type(metric).__name__)
    print(f"wrote {len(metrics) - len(failed)}/{len(metrics)} metric outputs")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)


def run_gwent_one(raw_path: Path | None) -> None:
    _reject_raw_path("gwent_one", raw_path)
    binder = _require_binder(
        GameId.GWENT,
        "run 'python3 scripts/run_card_binder_ingestion.py --source gwent_one' first.",
    )
    metrics: list[CorpusScanMetric] = [
        FactionMaskMetric(binder),
        ColorMaskMetric(binder),
        TypeMaskMetric(binder),
        RarityMaskMetric(binder),
        SetMaskMetric(binder),
        ProvisionMaskMetric(binder),
        PowerMaskMetric(binder),
        ArmorMaskMetric(binder),
    ]

    _scan_corpus_metrics("gwent_one", metrics)


# --- dominiontabs ---


def run_dominiontabs(raw_path: Path | None) -> None:
    _reject_raw_path("dominiontabs", raw_path)
    binder = _require_binder(
        GameId.DOMINION,
        "run 'python3 scripts/run_card_binder_ingestion.py --source dominiontabs' "
        "first.",
    )
    metrics: list[CorpusScanMetric] = [
        CostRegressionMetric(binder),
        DominionTypeMaskMetric(binder),
        DominionSetMaskMetric(binder),  # also reads data/raw/dominiontabs/
    ]
    _scan_corpus_metrics("dominiontabs", metrics)


# --- single-card masks: scryfall, pokemon_tcg, cardvault_fabtcg, spire_codex ---


def run_scryfall(raw_path: Path | None) -> None:
    _reject_raw_path("scryfall", raw_path)
    binder = _require_binder(GameId.MTG, _MTG_BINDER_HINT)
    metrics: list[CorpusScanMetric] = [
        scryfall_masks.CmcRegressionMetric(binder),
        scryfall_masks.CardTypeMaskMetric(binder),
        scryfall_masks.RarityMaskMetric(binder),
        scryfall_masks.ColorsMaskMetric(binder),
        scryfall_masks.PowerRegressionMetric(binder),
        scryfall_masks.ToughnessRegressionMetric(binder),
    ]
    _scan_corpus_metrics("scryfall", metrics)


def run_pokemon_tcg(raw_path: Path | None) -> None:
    _reject_raw_path("pokemon_tcg", raw_path)
    binder = _require_binder(
        GameId.POKEMON,
        "run 'python3 scripts/run_card_binder_ingestion.py --source pokemon_tcg' "
        "first.",
    )
    metrics: list[CorpusScanMetric] = [
        pokemon_masks.HpRegressionMetric(binder),
        pokemon_masks.TypesMaskMetric(binder),
        pokemon_masks.StageMaskMetric(binder),
        pokemon_masks.RetreatCostRegressionMetric(binder),
        pokemon_masks.WeaknessMaskMetric(binder),
    ]
    _scan_corpus_metrics("pokemon_tcg", metrics)


def run_cardvault_fabtcg(raw_path: Path | None) -> None:
    _reject_raw_path("cardvault_fabtcg", raw_path)
    binder = _require_binder(
        GameId.FLESH_AND_BLOOD,
        "run 'python3 scripts/run_card_binder_ingestion.py --source "
        "cardvault_fabtcg' first.",
    )
    metrics: list[CorpusScanMetric] = [
        fabtcg_masks.PitchMaskMetric(binder),
        fabtcg_masks.CostRegressionMetric(binder),
        fabtcg_masks.PowerRegressionMetric(binder),
        fabtcg_masks.DefenseRegressionMetric(binder),
        fabtcg_masks.ClassMaskMetric(binder),
        fabtcg_masks.CardTypeMaskMetric(binder),
    ]
    _scan_corpus_metrics("cardvault_fabtcg", metrics)


def run_spire_codex(raw_path: Path | None) -> None:
    _reject_raw_path("spire_codex", raw_path)
    binder = _require_binder(
        GameId.SLAY_THE_SPIRE_2,
        "run 'python3 scripts/run_card_binder_ingestion.py --source spire_codex' "
        "first.",
    )
    metrics: list[CorpusScanMetric] = [
        sts2_masks.CostMaskMetric(binder),
        sts2_masks.CardTypeMaskMetric(binder),
        sts2_masks.RarityMaskMetric(binder),
        sts2_masks.ColorMaskMetric(binder),
    ]
    _scan_corpus_metrics("spire_codex", metrics)


def run_hearthstonejson(raw_path: Path | None) -> None:
    _reject_raw_path("hearthstonejson", raw_path)
    binder = _require_binder(
        GameId.HEARTHSTONE,
        "run 'python3 scripts/run_card_binder_ingestion.py --source "
        "hearthstonejson' first.",
    )
    metrics: list[CorpusScanMetric] = [
        hearthstone_masks.CostRegressionMetric(binder),
        hearthstone_masks.AttackRegressionMetric(binder),
        hearthstone_masks.HealthRegressionMetric(binder),
        hearthstone_masks.ClassMaskMetric(binder),
        hearthstone_masks.RarityMaskMetric(binder),
        hearthstone_masks.CardTypeMaskMetric(binder),
        hearthstone_masks.RacesMaskMetric(binder),
        hearthstone_masks.SpellSchoolMaskMetric(binder),
    ]
    _scan_corpus_metrics("hearthstonejson", metrics)


# --- play_gwent ---


def _require_published_deck_box(game: GameId) -> DeckBox:
    """The published deck box for game, for metrics that only read it.

    Inputs: game. Output: DeckBox (single-path load: callers must not
        write through it, and never save() it).
    Side effects: opens the box file. Exceptions: SystemExit if missing.
    """
    box_path = DeckBox.default_output_path(game)
    if not box_path.exists():
        raise SystemExit(f"{box_path} does not exist; run deck box ingestion first.")
    return DeckBox.load([box_path])


def run_play_gwent(raw_path: Path | None) -> None:
    """Leader masked from deck over guides.jsonl. Reads the published
    Gwent box and never writes or saves it (run deck box ingestion for
    play_gwent first)."""
    effective_raw_path = raw_path or PLAY_GWENT_DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.GWENT,
        "run 'python3 scripts/run_card_binder_ingestion.py --source gwent_one' first.",
    )
    deck_box = _require_published_deck_box(GameId.GWENT)
    metrics: list[Metric[dict]] = [LeaderMaskedFromDeckMetric(binder, deck_box)]

    print(f"=== play_gwent: {effective_raw_path} ===")
    scan_guides_jsonl(effective_raw_path, metrics)
    print(f"wrote {len(metrics)} metric outputs")


# --- fabtcg_decklists ---


def run_fabtcg_decklists(raw_path: Path | None) -> None:
    """Hero mask and card inclusion rates over the raw decklist files.
    Reads the published FaB deck box and never writes or saves it."""
    effective_raw_path = raw_path or FABTCG_DECKLISTS_DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.FLESH_AND_BLOOD,
        "run 'python3 scripts/run_card_binder_ingestion.py --source "
        "cardvault_fabtcg' first.",
    )
    deck_box = _require_published_deck_box(GameId.FLESH_AND_BLOOD)
    metrics: list[Metric[dict]] = [
        HeroMaskedFromDeckMetric(binder, deck_box),
        FabCardInclusionRateMetric(binder, deck_box),
        HeroConditionedInclusionMetric(binder, deck_box),
    ]

    print(f"=== fabtcg_decklists: {effective_raw_path} ===")
    scan_decklist_files(effective_raw_path, metrics)
    print(f"wrote {len(metrics)} metric outputs")


def run_play_gwent_guides(raw_path: Path | None) -> None:
    """Card inclusion rates and guide votes over guides.jsonl. Like
    run_play_gwent, this reads the published box and never writes or
    saves it."""
    effective_raw_path = raw_path or PLAY_GWENT_DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.GWENT,
        "run 'python3 scripts/run_card_binder_ingestion.py --source gwent_one' first.",
    )
    deck_box = _require_published_deck_box(GameId.GWENT)
    metrics: list[Metric[dict]] = [
        GwentCardInclusionRateMetric(binder, deck_box),
        FactionConditionedInclusionMetric(binder, deck_box),
        GuideVotesMetric(binder, deck_box),
    ]

    print(f"=== play_gwent_guides: {effective_raw_path} ===")
    scan_guides_jsonl(effective_raw_path, metrics)
    print(f"wrote {len(metrics)} metric outputs")


# --- isotropic ---

_ISOTROPIC_RAW_DIR = Path("data/raw/isotropic")
# ISOTROPIC_DECK_BOX_PATH is shared by both families and warm-started on every run (deck
# uuids are content-derived, so re-adding a deck is a no-op) - running
# one family never drops the other family's decks from the box.
# Flavor A: one games-YYYYMMDD.json JSONL member per day.
_ISOTROPIC_SUMMARY_GLOB = "*-summary.tar.bz2"
# Flavor B: bare "<year>_<YYYYMMDD>.tar.bz2" full-log archives. Deliberately
# excludes 201010_11_all.tar.bz2, a byte-identical legacy-URL duplicate of
# 2010_20101011.tar.bz2 (see metrics/isotropic/games/README.md).
_ISOTROPIC_GAME_LOG_GLOB = "????_????????.tar.bz2"
_DOMINION_BINDER_HINT = (
    "run 'python3 scripts/run_card_binder_ingestion.py --source dominiontabs' first."
)


def _isotropic_archive_paths(raw_path: Path | None, pattern: str) -> list[Path]:
    """[raw_path] if given, else every archive under _ISOTROPIC_RAW_DIR
    matching pattern; exits if none are found."""
    archive_paths = (
        [raw_path] if raw_path is not None else sorted(_ISOTROPIC_RAW_DIR.glob(pattern))
    )
    if not archive_paths:
        raise SystemExit(
            f"No {pattern} archives under {_ISOTROPIC_RAW_DIR} — run "
            "'python3 scripts/run_data_retrieval.py --source isotropic' first."
        )
    return archive_paths


def _load_isotropic_deck_box() -> DeckBox:
    return DeckBox.load(
        [ISOTROPIC_DECK_BOX_PATH] if ISOTROPIC_DECK_BOX_PATH.exists() else []
    )


def run_isotropic_summary(raw_path: Path | None) -> None:
    archive_paths = _isotropic_archive_paths(raw_path, _ISOTROPIC_SUMMARY_GLOB)
    binder = _require_binder(GameId.DOMINION, _DOMINION_BINDER_HINT)
    deck_box = _load_isotropic_deck_box()

    metrics: list[Metric[dict]] = [
        VetoRateMetric(binder),
        CopiesBoughtDistributionMetric(binder),
        AverageCopiesBoughtMetric(binder),
        TurnCountAssociationMetric(binder),
        FullDeckWinPredictionMetric(binder, deck_box),
        KingdomVetoPredictionMetric(binder, deck_box),
        KingdomGameLengthMetric(binder, deck_box),
        WinningDeckMembershipMetric(binder, deck_box),
        WinningDeckCountMetric(binder, deck_box),
        WinningDeckMaskedCardMetric(binder, deck_box),
        DeckCardSetCopyCountMetric(binder, deck_box),
        DeckPairWinnerMetric(binder, deck_box),
        MultiplayerPlacementMetric(binder, deck_box),
    ]

    print(f"=== isotropic_summary: {', '.join(p.name for p in archive_paths)} ===")
    scan_isotropic_summary_archives(archive_paths, metrics)
    deck_box.save(
        ISOTROPIC_DECK_BOX_PATH, GameId.DOMINION, binder.version_for(GameId.DOMINION)
    )
    print(f"wrote {len(metrics)} metric outputs")


def run_isotropic_games(raw_path: Path | None) -> None:
    archive_paths = _isotropic_archive_paths(raw_path, _ISOTROPIC_GAME_LOG_GLOB)
    binder = _require_binder(GameId.DOMINION, _DOMINION_BINDER_HINT)
    deck_box = _load_isotropic_deck_box()

    header_metrics: list[Metric[GameHeader]] = [
        OpeningBuyRateMetric(binder),
        PileExhaustionRateMetric(binder),
        KingdomOpeningBuyPredictionMetric(binder, deck_box),
        OpeningBuyOutcomeMetric(binder, deck_box),
        KingdomEndingPilePredictionMetric(binder, deck_box),
        KingdomEndingTypeMetric(binder, deck_box),
    ]
    game_log_metrics: list[Metric[GameLog]] = [
        NextBuyPredictionMetric(binder, deck_box),
        NextTrashedCardMetric(binder, deck_box),
        NextTurnActionCountMetric(binder, deck_box),
        EventualWinProbabilityMetric(binder, deck_box),
        MidGameDeckPairWinnerMetric(binder, deck_box),
    ]

    print(f"=== isotropic_games: {', '.join(p.name for p in archive_paths)} ===")
    # Two read passes over the same archives, one per scanner: header
    # metrics and mid-game metrics take different parsed row types.
    scan_isotropic_game_log_archives(archive_paths, header_metrics)
    scan_isotropic_game_logs_archives(archive_paths, game_log_metrics)
    deck_box.save(
        ISOTROPIC_DECK_BOX_PATH, GameId.DOMINION, binder.version_for(GameId.DOMINION)
    )
    print(f"wrote {len(header_metrics) + len(game_log_metrics)} metric outputs")


# --- seventeenlands ---


_METRICS_ROOT = Path("data/metrics")


def _namespaced_output_path(
    default_path: Path, expansion: str, format_code: str, output_root: Path | None
) -> Path:
    """default_path, relocated under <expansion>/<format_code> so one
    metric's output from one CSV never overwrites its output from
    another (see module docstring). When output_root is given, the path
    is also re-rooted from data/metrics/ to output_root: parity runs
    write to a scratch tree, never over the reference outputs.

    Inputs: default_path (a DEFAULT_OUTPUT_PATH under data/metrics/),
        expansion, format_code, output_root (None: keep data/metrics/).
    Output: Path. Side effects: none.
    Exceptions: ValueError if output_root is given and default_path is
        not under data/metrics/.
    """
    relocated = default_path.parent / expansion / format_code / default_path.name
    if output_root is None:
        return relocated
    if not relocated.is_relative_to(_METRICS_ROOT):
        raise ValueError(f"{default_path} is not under {_METRICS_ROOT}")
    return output_root / relocated.relative_to(_METRICS_ROOT)


def _parse_expansion_format(csv_path: Path) -> tuple[str, str]:
    """ "<Expansion>.<EventType>.csv" -> (expansion, format_code) - the
    naming convention every 17lands raw CSV in this project follows."""
    expansion, format_code = csv_path.stem.split(".", 1)
    return expansion, format_code


class RowMetricClass(Protocol):
    """A row metric class (Metric[dict]) built from one CSV's header."""

    DEFAULT_OUTPUT_PATH: Path

    def __call__(
        self,
        card_binder: CardBinder,
        header: pd.Index,
        source_game: GameId,
        output_path: Path | None = None,
    ) -> Metric[dict]: ...


class DeckBoxRowMetricClass(Protocol):
    """A row metric class that also writes decks into the family box."""

    DEFAULT_OUTPUT_PATH: Path

    def __call__(
        self,
        card_binder: CardBinder,
        header: pd.Index,
        source_game: GameId,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> Metric[dict]: ...


class ChunkMetricClass(Protocol):
    """A chunk metric class: built from the run's version metadata (no
    binder, no header)."""

    DEFAULT_OUTPUT_PATH: Path

    def __call__(
        self, version_metadata: MetricVersionMetadata, output_path: Path | None = None
    ) -> Metric[Any]: ...


class DeckBoxChunkMetricClass(Protocol):
    """A chunk metric class that also writes decks into the family box."""

    DEFAULT_OUTPUT_PATH: Path

    def __call__(
        self,
        version_metadata: MetricVersionMetadata,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> Metric[Any]: ...


@dataclass(frozen=True)
class RowMetricSpec:
    """A row metric a family scans. Transitional: deleted with the last
    row metric."""

    metric_class: RowMetricClass


@dataclass(frozen=True)
class DeckBoxRowMetricSpec:
    """A row metric that takes the family deck box. Transitional."""

    metric_class: DeckBoxRowMetricClass


@dataclass(frozen=True)
class ChunkMetricSpec:
    """A chunk metric a family scans."""

    metric_class: ChunkMetricClass


@dataclass(frozen=True)
class DeckBoxChunkMetricSpec:
    """A chunk metric that takes the family deck box."""

    metric_class: DeckBoxChunkMetricClass


RowSpec = RowMetricSpec | DeckBoxRowMetricSpec
ChunkSpec = ChunkMetricSpec | DeckBoxChunkMetricSpec
SeventeenLandsMetricSpec = RowSpec | ChunkSpec

# One CSV's scan, (csv_path, metrics) -> None.
FamilyScan = Callable[[Path, list[Metric[Any]]], None]

# Builds one CSV's scan from (header, binder, keep_source_frame). The
# driver computes keep_source_frame once from the family's specs.
ScanForCsv = Callable[[pd.Index, CardBinder, bool], FamilyScan]


@dataclass(frozen=True)
class RowScanning:
    """A family that still scans rows: the same row scanner for every
    CSV, and only row specs.

    scan: the family's row scanner.
    """

    scan: FamilyScan

    def scan_for_csv(
        self, header: pd.Index, binder: CardBinder, keep_source_frame: bool
    ) -> FamilyScan:
        """self.scan, whatever the CSV (header, binder and
        keep_source_frame are unused: a row scanner reads every column).

        Inputs: header, binder, keep_source_frame. Output: FamilyScan.
        Side effects: none. Exceptions: none.

        Example:
            >>> RowScanning(scan_draft_csv).scan_for_csv(header, binder, False)
        """
        return self.scan


@dataclass(frozen=True)
class ChunkScanning:
    """A family that scans typed chunks.

    scan_builder: builds one CSV's scan (game_data builds its
        GameDataChunkParser here).
    frame_of: set only while the family still wraps row metrics; they
        are then wrapped in RowwiseMetric(metric, frame_of). None for a
        family whose metrics are all chunk metrics, where a row spec is
        rejected before any CSV is read.
    """

    scan_builder: ScanForCsv
    frame_of: Callable[[Any], pd.DataFrame] | None

    def scan_for_csv(
        self, header: pd.Index, binder: CardBinder, keep_source_frame: bool
    ) -> FamilyScan:
        """One CSV's scan, from scan_builder.

        Inputs: header, binder, keep_source_frame (from
            _check_family_specs). Output: FamilyScan.
        Side effects: scan_builder's. Exceptions: scan_builder's.

        Example:
            >>> ChunkScanning(_scan_game_data_csv, None).scan_for_csv(
            ...     header, binder, False)
        """
        return self.scan_builder(header, binder, keep_source_frame)


@dataclass(frozen=True)
class _SeventeenLandsFamily:
    """Everything _run_seventeenlands_family needs to know about one
    family.

    scanning: how the family scans a CSV: rows, or typed chunks.
    """

    name: str
    family_dir: Path
    metric_specs: Sequence[SeventeenLandsMetricSpec]
    scanning: RowScanning | ChunkScanning
    deck_box_output_path: Path | None


_DRAFT_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    RowMetricSpec(CardTakeRateMetric),
    RowMetricSpec(FirstPickRateMetric),
    RowMetricSpec(RankStratifiedTakeRateMetric),
    RowMetricSpec(PickNumberDecayCurveMetric),
    RowMetricSpec(PackToPickChoiceSetMetric),
    RowMetricSpec(PoolConditionedPickMetric),
)

_GAME_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    ChunkMetricSpec(WinRateWhenInDeckMetric),
    ChunkMetricSpec(OpeningHandWinRateMetric),
    ChunkMetricSpec(DrawnWinRateMetric),
    ChunkMetricSpec(GameLengthAssociationMetric),
    ChunkMetricSpec(OnPlayWinRateDeltaMetric),
    ChunkMetricSpec(GameTutorTargetRateMetric),
    DeckBoxChunkMetricSpec(DeckWinPredictionMetric),
    DeckBoxChunkMetricSpec(DeckGameLengthPredictionMetric),
    DeckBoxChunkMetricSpec(DeckRankTierPredictionMetric),
    DeckBoxChunkMetricSpec(OnPlayWinRateSensitivityByDeckMetric),
    ChunkMetricSpec(TutorTargetPoolMetric),
)

_REPLAY_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    RowMetricSpec(AverageTurnCastMetric),
    RowMetricSpec(CastRateMetric),
    RowMetricSpec(TurnsToGameEndAfterCastMetric),
    RowMetricSpec(DiscardRateMetric),
    RowMetricSpec(ReplayTutorTargetRateMetric),
    RowMetricSpec(CombatKillInvolvementRateMetric),
    RowMetricSpec(CombatDamagePushThroughRateMetric),
    DeckBoxRowMetricSpec(CombatAggressionProfileMetric),
    RowMetricSpec(AttackerBlockerCombatOutcomeMetric),
)


@dataclass(frozen=True)
class _CsvMetricContext:
    """The per-CSV, per-run values every metric constructor draws from.

    binder/header/source_game: the row constructors' inputs.
    version_metadata: the chunk constructors' input, computed once per
        run.
    expansion/format_code/output_root: where outputs go.
    deck_box: the family box, for deck-box specs.
    """

    binder: CardBinder
    header: pd.Index
    source_game: GameId
    version_metadata: MetricVersionMetadata
    expansion: str
    format_code: str
    output_root: Path | None
    deck_box: DeckBox | None


def _namespaced_metrics(
    family: _SeventeenLandsFamily, context: _CsvMetricContext
) -> list[Metric[Any]]:
    """Construct every metric in specs for one 17lands CSV, each writing
    to its namespaced output path.

    Inputs: family (its specs already checked by _check_family_specs),
        context.
    Output: list of metrics, in family.metric_specs order. Chunk
        metrics come back as built. Row metrics are wrapped in
        RowwiseMetric when the family scans chunks with a frame_of, and
        come back as built in a row family.
    Side effects: none beyond the metric constructors'.
    Exceptions: ValueError if a deck-box spec meets a context without a
        deck box.

    Example:
        >>> _namespaced_metrics(game_data_family, context)
    """
    result: list[Metric[Any]] = []
    frame_of = _frame_of(family.scanning)

    for spec in family.metric_specs:
        output_path = _namespaced_output_path(
            spec.metric_class.DEFAULT_OUTPUT_PATH,
            context.expansion,
            context.format_code,
            context.output_root,
        )

        # Chunk metrics: built from the run's version metadata
        if isinstance(spec, (ChunkMetricSpec, DeckBoxChunkMetricSpec)):
            result.append(_build_chunk_metric(spec, context, output_path))
            continue

        # Row metrics: built as before, wrapped only in a chunk family
        row_metric = _build_row_metric(spec, context, output_path)
        if frame_of is None:
            result.append(row_metric)
        else:
            result.append(RowwiseMetric(row_metric, frame_of))

    return result


def _frame_of(
    scanning: RowScanning | ChunkScanning,
) -> Callable[[Any], pd.DataFrame] | None:
    """The frame_of row metrics are wrapped with: a chunk family's, or
    None for a row family (its row metrics run unwrapped).

    Inputs: scanning. Output: the callable, or None.
    Side effects: none. Exceptions: none.
    """
    return scanning.frame_of if isinstance(scanning, ChunkScanning) else None


def _build_chunk_metric(
    spec: ChunkSpec, context: _CsvMetricContext, output_path: Path
) -> Metric[Any]:
    """Construct one chunk metric from the run's version metadata (and
    the family deck box, for a deck-box spec).

    Inputs: spec, context, output_path. Output: the metric.
    Side effects: the constructor's.
    Exceptions: ValueError if spec is a DeckBoxChunkMetricSpec and
        context.deck_box is None.
    """
    if isinstance(spec, ChunkMetricSpec):
        return spec.metric_class(context.version_metadata, output_path)
    if context.deck_box is None:
        raise ValueError(f"{spec.metric_class!r} needs the family DeckBox")
    return spec.metric_class(context.version_metadata, context.deck_box, output_path)


def _build_row_metric(
    spec: RowSpec,
    context: _CsvMetricContext,
    output_path: Path,
) -> Metric[dict]:
    """Construct one row metric exactly as before this migration.

    Inputs: spec, context, output_path. Output: the metric.
    Side effects: the constructor's (row metrics still call
        binder.version_for themselves).
    Exceptions: ValueError if spec is a DeckBoxRowMetricSpec and
        context.deck_box is None.
    """
    if isinstance(spec, RowMetricSpec):
        return spec.metric_class(
            context.binder, context.header, context.source_game, output_path=output_path
        )
    if context.deck_box is None:
        raise ValueError(f"{spec.metric_class!r} needs the family DeckBox")
    return spec.metric_class(
        context.binder,
        context.header,
        context.source_game,
        context.deck_box,
        output_path=output_path,
    )


def _check_family_specs(family: _SeventeenLandsFamily) -> bool:
    """Reject an impossible family before any CSV is read, and report
    whether it still wraps row metrics (its chunks must then carry a
    source frame). This is the single source of that fact.

    Inputs: family.
    Output: True if the family scans chunks with a frame_of and has a
        row spec; False otherwise (game_data, whose metrics are all
        chunk metrics, is False).
    Side effects: none.
    Exceptions: ValueError if a row family registers a chunk spec, a
        chunk family without a frame_of registers a row spec, or a
        deck-box spec meets a family with no deck_box_output_path.
    """
    specs = family.metric_specs
    has_row_specs = any(
        isinstance(s, (RowMetricSpec, DeckBoxRowMetricSpec)) for s in specs
    )
    has_chunk_specs = any(
        isinstance(s, (ChunkMetricSpec, DeckBoxChunkMetricSpec)) for s in specs
    )
    has_deck_box_specs = any(
        isinstance(s, (DeckBoxRowMetricSpec, DeckBoxChunkMetricSpec)) for s in specs
    )
    frame_of = _frame_of(family.scanning)

    # Reject what can't run, before any CSV is read
    if isinstance(family.scanning, RowScanning) and has_chunk_specs:
        raise ValueError(f"{family.name} scans rows but registers a chunk metric")
    if (
        isinstance(family.scanning, ChunkScanning)
        and frame_of is None
        and has_row_specs
    ):
        raise ValueError(
            f"{family.name} scans chunks with no frame_of but registers a row metric"
        )
    if has_deck_box_specs and family.deck_box_output_path is None:
        raise ValueError(f"{family.name} has a deck-box metric but no deck box path")

    return frame_of is not None and has_row_specs


def _run_seventeenlands_family(
    family: _SeventeenLandsFamily, raw_path: Path | None, output_root: Path | None
) -> None:
    """Scan every CSV of one 17lands family, or just raw_path.

    Inputs:
        family: the family to scan.
        raw_path: one CSV to scan; None scans every CSV in
            family.family_dir.
        output_root: None writes under data/metrics/; a path is a
            scratch root for parity runs, deck box included.
    Output: none.
    Side effects: loads the MTG binder (once) and the family deck box;
        writes every metric's per-CSV output; saves the deck box.
    Exceptions: ValueError from _check_family_specs; SystemExit if no
        CSV is found; whatever a scan raises.

    Example:
        >>> _run_seventeenlands_family(game_data_family, Path(
        ...     "data/raw/17lands/game_data/KTK.TradDraft.csv"), None)
    """
    # Validate the family, and learn whether row metrics remain
    keep_source_frame = _check_family_specs(family)

    # The binder, loaded once, and its version, hashed once
    binder = _require_binder(GameId.MTG, _MTG_BINDER_HINT)
    version_metadata = MetricVersionMetadata(
        game=GameId.MTG, card_binder_version=binder.version_for(GameId.MTG)
    )

    csv_paths = (
        [raw_path] if raw_path is not None else sorted(family.family_dir.glob("*.csv"))
    )
    if not csv_paths:
        raise SystemExit(f"No CSVs found under {family.family_dir}.")

    deck_box_path = _deck_box_path(family.deck_box_output_path, output_root)
    deck_box = _load_family_deck_box(deck_box_path)

    # Each CSV: build its metrics and its scan from its own header
    for csv_path in csv_paths:
        expansion, format_code = _parse_expansion_format(csv_path)
        header = pd.read_csv(csv_path, nrows=0).columns
        context = _CsvMetricContext(
            binder=binder,
            header=header,
            source_game=GameId.MTG,
            version_metadata=version_metadata,
            expansion=expansion,
            format_code=format_code,
            output_root=output_root,
            deck_box=deck_box,
        )
        metrics = _namespaced_metrics(family, context)
        scan = family.scanning.scan_for_csv(header, binder, keep_source_frame)

        print(f"=== {family.name}: {csv_path} ({expansion}.{format_code}) ===")
        scan(csv_path, metrics)
        print(f"wrote {len(metrics)} metric outputs")

    if deck_box_path is not None:
        assert deck_box is not None
        deck_box.save(deck_box_path, GameId.MTG, version_metadata.card_binder_version)


def _deck_box_path(default_path: Path | None, output_root: Path | None) -> Path | None:
    """The family deck box path, re-rooted under output_root for a
    parity run so a parity run never writes the real box.

    Inputs: default_path, output_root. Output: Path or None.
    Side effects: none. Exceptions: none.
    """
    if default_path is None or output_root is None:
        return default_path
    return output_root / default_path.relative_to(_METRICS_ROOT)


def _load_family_deck_box(path: Path | None) -> DeckBox | None:
    """The family deck box at path (empty if the file doesn't exist
    yet), or None for a family without one.

    Inputs: path. Output: DeckBox or None.
    Side effects: reads path if it exists. Exceptions: DeckBox.load's.
    """
    if path is None:
        return None
    return DeckBox.load([path] if path.exists() else [])


def _scan_game_data_csv(
    header: pd.Index, binder: CardBinder, keep_source_frame: bool
) -> FamilyScan:
    """game_data's ScanForCsv: a GameDataChunkParser for this header,
    bound into scan_game_csv.

    Inputs: header, binder, keep_source_frame (from _check_family_specs;
        always False, since game_data has no row metrics).
    Output: FamilyScan.
    Side effects: none (no I/O).
    Exceptions: ValueError if keep_source_frame is True (a row metric
        registered in game_data, whose chunks carry no source frame);
        GameDataChunkParser.from_header's.
    """
    if keep_source_frame:
        raise ValueError("game_data chunks carry no source frame for row metrics")
    parser = GameDataChunkParser.from_header(list(header), binder, GameId.MTG)

    return partial(scan_game_csv, parser=parser)


def run_seventeenlands_draft_data(
    raw_path: Path | None, output_root: Path | None = None
) -> None:
    """Run the draft_data family (row metrics, row scanner).

    Inputs: raw_path (one CSV, or None for every draft_data CSV),
        output_root (None, or a scratch root for parity runs).
    Output: none.
    Side effects: writes per-CSV outputs.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_draft_data(None)
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_draft_data",
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "draft_data",
            metric_specs=_DRAFT_DATA_METRICS,
            scanning=RowScanning(scan_draft_csv),
            deck_box_output_path=None,
        ),
        raw_path,
        output_root,
    )


def run_seventeenlands_game_data(
    raw_path: Path | None, output_root: Path | None = None
) -> None:
    """Run the game_data family (chunk scan, chunk metrics only).

    Inputs: raw_path (one CSV, or None for every game_data CSV),
        output_root (None, or a scratch root for parity runs).
    Output: none.
    Side effects: writes per-CSV outputs and the family deck box.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_game_data(
        ...     Path("data/raw/17lands/game_data/KTK.TradDraft.csv"),
        ...     output_root=Path("scratch/parity"))
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_game_data",
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data",
            metric_specs=_GAME_DATA_METRICS,
            scanning=ChunkScanning(_scan_game_data_csv, frame_of=None),
            deck_box_output_path=Path(
                "data/metrics/seventeenlands/game_data/deck_box.db"
            ),
        ),
        raw_path,
        output_root,
    )


def run_seventeenlands_replay_data(
    raw_path: Path | None, output_root: Path | None = None
) -> None:
    """Run the replay_data family (row metrics, row scanner).

    Inputs: raw_path (one CSV, or None for every replay_data CSV),
        output_root (None, or a scratch root for parity runs).
    Output: none.
    Side effects: writes per-CSV outputs and the family deck box.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_replay_data(None)
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_replay_data",
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "replay_data",
            metric_specs=_REPLAY_DATA_METRICS,
            scanning=RowScanning(scan_replay_csv),
            deck_box_output_path=Path(
                "data/metrics/seventeenlands/replay_data/deck_box.db"
            ),
        ),
        raw_path,
        output_root,
    )


class _SeventeenLandsRunner(Protocol):
    """A seventeenlands family runner: --raw-path, and an optional
    --output-root. Also a plain _FAMILIES runner (raw_path only)."""

    def __call__(self, raw_path: Path | None, output_root: Path | None = None) -> None: ...


# The three seventeenlands runners, the only families --output-root
# applies to.
_SEVENTEENLANDS_RUNNERS: dict[str, _SeventeenLandsRunner] = {
    "seventeenlands_draft_data": run_seventeenlands_draft_data,
    "seventeenlands_game_data": run_seventeenlands_game_data,
    "seventeenlands_replay_data": run_seventeenlands_replay_data,
}


# --- final_decks ---

# Read-only scans of the published deck boxes (data/final/decks/<game>.db),
# smallest box first. Runnable with --source only, never by --all: they
# must run after deck-box ingestion (and play_gwent, which writes the
# Gwent box), and the StS2/MTG scans take long.
_FINAL_DECKS_METRICS: dict[str, type[HeldOutDeckCardMetric]] = {
    "final_decks_pokemon": PokemonHeldOutCardMetric,
    "final_decks_flesh_and_blood": FleshAndBloodHeldOutCardMetric,
    "final_decks_gwent": GwentHeldOutCardMetric,
    "final_decks_dominion": DominionHeldOutCardMetric,
    "final_decks_slay_the_spire_2": SlayTheSpire2HeldOutCardMetric,
    "final_decks_mtg": MtgHeldOutCardMetric,
}


def run_final_decks(name: str, raw_path: Path | None) -> None:
    """Scan one published deck box into its held-out card metric.

    Inputs: name (a _FINAL_DECKS_METRICS key), raw_path (must be None).
    Output: none. Side effects: reads the game's binder and deck box,
        writes the metric's parquet. Exceptions: SystemExit if raw_path
        is given or the binder or box is missing; ValueError from scan()
        on a stale box.
    """
    _reject_raw_path(name, raw_path)
    metric_class = _FINAL_DECKS_METRICS[name]
    game = metric_class.SOURCE_GAME
    binder = _require_binder(
        game, "run 'python3 scripts/run_card_binder_ingestion.py' for it first."
    )
    box_path = DeckBox.default_output_path(game)
    if not box_path.exists():
        raise SystemExit(f"{box_path} does not exist; run deck box ingestion first.")
    deck_box = DeckBox.load([box_path])

    print(f"=== {name}: {box_path} ===")
    output_path = metric_class(binder, deck_box).scan()
    print(f"wrote {output_path}")


_FAMILIES: dict[str, Callable[[Path | None], None]] = {
    "sts_gg": run_sts_gg,
    "sts2_runs": run_sts2_runs,
    "gwent_one": run_gwent_one,
    "dominiontabs": run_dominiontabs,
    "play_gwent": run_play_gwent,
    "fabtcg_decklists": run_fabtcg_decklists,
    "play_gwent_guides": run_play_gwent_guides,
    "scryfall": run_scryfall,
    "pokemon_tcg": run_pokemon_tcg,
    "cardvault_fabtcg": run_cardvault_fabtcg,
    "spire_codex": run_spire_codex,
    "hearthstonejson": run_hearthstonejson,
    "isotropic_summary": run_isotropic_summary,
    "isotropic_games": run_isotropic_games,
    **_SEVENTEENLANDS_RUNNERS,
}


def run_all() -> None:
    """Run every family in this process, isolating one family's failure
    (logged via traceback, not raised) from the rest."""
    failed: list[str] = []
    for name in sorted(_FAMILIES):
        try:
            _FAMILIES[name](None)
        except Exception:
            traceback.print_exc()
            failed.append(name)
        print()

    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available metric family names and exit.",
    )
    parser.add_argument(
        "--source",
        choices=[*sorted(_FAMILIES), *_FINAL_DECKS_METRICS],
        help="Run just this one metric family.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Override --source's default raw input. For the three seventeenlands "
        "families, restricts the scan to this one CSV instead of every CSV under "
        "its raw data directory. Ignored without --source.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Seventeenlands families only: write outputs (and the family deck box) "
        "under this root instead of data/metrics/, e.g. for a parity run against "
        "existing outputs. Requires --source.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every metric family, one after another, in this process (the "
        "default when no other flag is given).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for name in sorted(_FAMILIES):
            print(name)
        for name in _FINAL_DECKS_METRICS:
            print(f"{name} (--source only)")
        return

    if args.source in _FINAL_DECKS_METRICS:
        run_final_decks(args.source, args.raw_path)
        return
    if args.output_root is not None:
        # Only the seventeenlands families can redirect their outputs
        if args.source not in _SEVENTEENLANDS_RUNNERS:
            raise SystemExit("--output-root needs a seventeenlands --source.")
        _SEVENTEENLANDS_RUNNERS[args.source](args.raw_path, args.output_root)
        return
    if args.source:
        _FAMILIES[args.source](args.raw_path)
        return

    run_all()


if __name__ == "__main__":
    main()
