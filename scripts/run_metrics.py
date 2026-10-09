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
`--all` (or omitting --raw-path) scans every CSV found; every metric
writes one partition file per CSV,
data/metrics/seventeenlands/<family>/<stem>/<expansion>/<format_code>.parquet
(src/data_refinement/metrics/seventeenlands/partition.py), so scanning
multiple sets never overwrites a previous one's output — the
metric classes themselves have no cross-file accumulation (each file's
header names different cards), so "one file, one output" is the actual
unit of work, not "one family, one output". sts2_runs reads two
fixed source (spire_codex's ~1.7M-run pages) in one
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
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.sts_gg.extraction_stage import (
    StsGgDeckExtractionStage,
)
from src.data_refinement.metrics.generic.corpus_scan_metric import CorpusScanMetric
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    UnsupportedCsvLayout,
)
from src.data_refinement.seventeenlands.csv_header import read_csv_header
from src.data_refinement.metrics.seventeenlands.deck_box_path import (
    REPLAY_DATA_DECK_BOX_PATH,
)
from src.data_refinement.metrics.seventeenlands.partition import (
    METRICS_ROOT,
    SeventeenLandsPartition,
)
from src.data_retrieval.seventeenlands import refs
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
from src.data_refinement.metrics.sts2_runs.card_removal_pick_metric import CardRemovalPickMetric
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import CardRewardPickMetric
from src.data_refinement.metrics.sts2_runs.card_upgrade_pick_metric import CardUpgradePickMetric
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric
from src.data_refinement.metrics.sts2_runs.shop_purchase_pick_metric import ShopPurchasePickMetric
from src.data_refinement.metrics.sts2_runs import deck_label_metrics as sts2_decks
from src.data_refinement.metrics.sts2_runs.run_record import Sts2Run
from src.data_refinement.metrics.sts2_runs.scanner import (
    default_run_sources,
    scan_sts2_runs,
)

# --- cross_game ---
from src.data_refinement.metrics.cross_game.rarity.rarity_tier_metric import (
    RarityTierMetric,
)
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
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

# --- cross_game ---


def run_cross_game(raw_path: Path | None) -> None:
    """Label every translated game's cards with a RarityTier.

    Inputs: raw_path (must be None). Output: none.
    Side effects: loads every translated game's binder into one
        CardBinder; writes RarityTierMetric.DEFAULT_OUTPUT_PATH. A scan()
        failure (such as UnmappedRarityError) is logged by
        _scan_corpus_metrics, not raised.
    Exceptions: SystemExit if raw_path is given or a game's binder file
        is missing.

    Runtime: about 25 s (measured 2026-10-08).
    """
    _reject_raw_path("cross_game", raw_path)

    # One binder loaded from each translated game's file
    paths = [CardBinder.default_output_path(game) for game in RARITY_TRANSLATORS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing card binder file(s) {missing}; ingest those first.")
    binder = CardBinder.load(paths)

    _scan_corpus_metrics("cross_game", [RarityTierMetric(binder)])


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
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk_parser import (
    DraftDataChunkParser,
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
from src.data_refinement.metrics.seventeenlands.game_data.deck_occurrence_count_metric import (  # noqa: E501
    DeckOccurrenceCountMetric,
)
from src.data_refinement.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import (
    scan_game_csv,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_choice_rate_metric import (
    TutorChoiceRateMetric,
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
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk_parser import (
    ReplayDataChunkParser,
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
    """Runtime: about 35 s (measured 2026-10-08)."""
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
    type[sts2_decks.DeckLabelMetric]
    | type[sts2_cards.CardAverageMetric]
    | type[PickChoiceMetric],
    ...,
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
    CardRewardPickMetric,
    ShopPurchasePickMetric,
    CardRemovalPickMetric,
    CardUpgradePickMetric,
)


def run_sts2_runs(raw_path: Path | None) -> None:
    """Scan spire_codex's run pages (at their
    default paths) into data/metrics/sts2_runs/. The deck-level outputs
    point into the published deck box (data/final/decks/
    slay_the_spire_2.db); nothing is written to it.

    Runtime: about 1 h 37 min for 40 spire_codex run pages (6.1 GB, 1.58M
    scored runs, 27 outputs), the first hour alongside isotropic_summary
    (measured 2026-10-08).
    """
    _reject_raw_path("sts2_runs", raw_path)
    binder = _require_binder(
        GameId.SLAY_THE_SPIRE_2,
        "run 'python3 scripts/run_card_binder_ingestion.py --source spire_codex' first.",
    )
    metrics: list[Metric[Sts2Run]] = [cls(binder) for cls in _STS2_RUNS_METRIC_CLASSES]

    print("=== sts2_runs: spire_codex run pages ===")
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
    """Runtime: about 4 s (measured 2026-10-08)."""
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
    """Runtime: about 4 s (measured 2026-10-08)."""
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
    """Runtime: about 25 s (measured 2026-10-08)."""
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
    """Runtime: about 15 s (measured 2026-10-08)."""
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
    """Runtime: about 7 s (measured 2026-10-08)."""
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
    """Runtime: about 2 s (measured 2026-10-08)."""
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
    """Runtime: about 4 s (measured 2026-10-08)."""
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
    play_gwent first).

    Runtime: about 1 min (measured 2026-10-08).
    """
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
    Reads the published FaB deck box and never writes or saves it.

    Runtime: about 25 s (measured 2026-10-08).
    """
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
    saves it.

    Runtime: about 1 min 10 s (measured 2026-10-08).
    """
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
    """The shared isotropic box, file-backed (created if missing) so it
    commits batch by batch instead of growing in memory."""
    return DeckBox.load([ISOTROPIC_DECK_BOX_PATH])


def run_isotropic_summary(raw_path: Path | None) -> None:
    """Runtime: about 55 min for both summary archives (2010: 21 days, about 10
    min; 2013: 15 days at about 3 min per day), run alongside sts2_runs
    (measured 2026-10-08).
    """
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
    """Runtime: about 16 min for both game-log archives (measured 2026-10-08).
    """
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


def _partition_path(
    family: "_SeventeenLandsFamily",
    spec: "SeventeenLandsMetricSpec",
    context: "_CsvMetricContext",
) -> Path:
    """Where one metric's output for one CSV goes: its
    SeventeenLandsPartition path, under output_root when given (parity
    runs write to a scratch tree, never over the reference outputs).

    Inputs: family (its data_type), spec (its output_stem), context
        (expansion, format_code, output_root).
    Output: <root>/seventeenlands/<family>/<stem>/<SET>/<Format>.parquet.
    Side effects: none. Exceptions: none.

    Example:
        >>> _partition_path(game_data_family, ChunkMetricSpec(DrawnWinRateMetric), context)
        PosixPath('data/metrics/seventeenlands/game_data/drawn_win_rate/KTK/TradDraft.parquet')
    """
    partition = SeventeenLandsPartition(
        family=family.data_type,
        metric_stem=spec.output_stem,
        expansion=context.expansion,
        format=context.format_code,
    )
    return partition.path(context.output_root or METRICS_ROOT)


def _parse_expansion_format(
    csv_path: Path,
) -> tuple[refs.Expansion, refs.FormatCode]:
    """ "<Expansion>.<EventType>.csv" -> (expansion, format_code) - the
    naming convention every 17lands raw CSV in this project follows.

    Inputs: csv_path. Output: the CSV's set and format.
    Side effects: none.
    Exceptions: ValueError if the name has no "." or names an unknown
        set or format (a CSV this project never downloaded).
    """
    expansion, separator, format_name = csv_path.stem.partition(".")
    if not separator:
        raise ValueError(f"{csv_path.name} is not <Expansion>.<EventType>.csv")
    return refs.Expansion(expansion), refs.FormatCode(format_name)


class ChunkMetricClass(Protocol):
    """A chunk metric class: built from the run's version metadata (no
    binder, no header), writing one partition file."""

    OUTPUT_STEM: str

    def __call__(
        self, version_metadata: MetricVersionMetadata, output_path: Path
    ) -> Metric[Any]: ...


class DeckBoxChunkMetricClass(Protocol):
    """A chunk metric class that also writes decks into the family box."""

    OUTPUT_STEM: str

    def __call__(
        self,
        version_metadata: MetricVersionMetadata,
        deck_box: DeckBox,
        output_path: Path,
    ) -> Metric[Any]: ...


@dataclass(frozen=True)
class ChunkMetricSpec:
    """A chunk metric a family scans."""

    metric_class: ChunkMetricClass

    @property
    def output_stem(self) -> str:
        """The partition directory name: the metric's OUTPUT_STEM.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.
        """
        return self.metric_class.OUTPUT_STEM


@dataclass(frozen=True)
class DeckBoxChunkMetricSpec:
    """A chunk metric that takes the family deck box."""

    metric_class: DeckBoxChunkMetricClass

    @property
    def output_stem(self) -> str:
        """The partition directory name: the metric's OUTPUT_STEM.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.
        """
        return self.metric_class.OUTPUT_STEM


SeventeenLandsMetricSpec = ChunkMetricSpec | DeckBoxChunkMetricSpec

# One CSV's scan, (csv_path, metrics) -> None.
FamilyScan = Callable[[Path, list[Metric[Any]]], None]

# Builds one CSV's scan from (header, binder).
ScanForCsv = Callable[[Sequence[str], CardBinder], FamilyScan]


@dataclass(frozen=True)
class _SeventeenLandsFamily:
    """Everything _run_seventeenlands_family needs to know about one
    family.

    scan_for_csv: builds one CSV's scan from its header (the family's
        chunk parser, bound into its scan function).
    """

    name: str
    data_type: refs.DataType
    family_dir: Path
    metric_specs: Sequence[SeventeenLandsMetricSpec]
    scan_for_csv: ScanForCsv
    deck_box_output_path: Path | None


_DRAFT_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    ChunkMetricSpec(CardTakeRateMetric),
    ChunkMetricSpec(FirstPickRateMetric),
    ChunkMetricSpec(RankStratifiedTakeRateMetric),
    ChunkMetricSpec(PickNumberDecayCurveMetric),
    ChunkMetricSpec(PackToPickChoiceSetMetric),
    ChunkMetricSpec(PoolConditionedPickMetric),
)

_GAME_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    ChunkMetricSpec(WinRateWhenInDeckMetric),
    ChunkMetricSpec(OpeningHandWinRateMetric),
    ChunkMetricSpec(DrawnWinRateMetric),
    ChunkMetricSpec(GameLengthAssociationMetric),
    ChunkMetricSpec(OnPlayWinRateDeltaMetric),
    ChunkMetricSpec(GameTutorTargetRateMetric),
    ChunkMetricSpec(DeckWinPredictionMetric),
    ChunkMetricSpec(DeckGameLengthPredictionMetric),
    ChunkMetricSpec(DeckRankTierPredictionMetric),
    ChunkMetricSpec(OnPlayWinRateSensitivityByDeckMetric),
    ChunkMetricSpec(DeckOccurrenceCountMetric),
    ChunkMetricSpec(TutorChoiceRateMetric),
)

_REPLAY_DATA_METRICS: tuple[SeventeenLandsMetricSpec, ...] = (
    ChunkMetricSpec(AverageTurnCastMetric),
    ChunkMetricSpec(CastRateMetric),
    ChunkMetricSpec(TurnsToGameEndAfterCastMetric),
    ChunkMetricSpec(DiscardRateMetric),
    ChunkMetricSpec(ReplayTutorTargetRateMetric),
    ChunkMetricSpec(CombatKillInvolvementRateMetric),
    ChunkMetricSpec(CombatDamagePushThroughRateMetric),
    DeckBoxChunkMetricSpec(CombatAggressionProfileMetric),
    ChunkMetricSpec(AttackerBlockerCombatOutcomeMetric),
)


@dataclass(frozen=True)
class _CsvMetricContext:
    """The per-CSV, per-run values every metric constructor draws from.

    version_metadata: every constructor's input, computed once per run.
    expansion/format_code/output_root: where outputs go.
    deck_box: the family box, for deck-box specs.
    """

    version_metadata: MetricVersionMetadata
    expansion: refs.Expansion
    format_code: refs.FormatCode
    output_root: Path | None
    deck_box: DeckBox | None


def _namespaced_metrics(
    family: _SeventeenLandsFamily, context: _CsvMetricContext
) -> list[Metric[Any]]:
    """Construct every metric in specs for one 17lands CSV, each writing
    to its namespaced output path.

    Inputs: family (its specs already checked by _check_family_specs),
        context.
    Output: list of metrics, in family.metric_specs order.
    Side effects: none beyond the metric constructors'.
    Exceptions: ValueError if a deck-box spec meets a context without a
        deck box.

    Example:
        >>> _namespaced_metrics(game_data_family, context)
    """
    return [
        _build_chunk_metric(spec, context, _partition_path(family, spec, context))
        for spec in family.metric_specs
    ]


def _build_chunk_metric(
    spec: SeventeenLandsMetricSpec, context: _CsvMetricContext, output_path: Path
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


def _check_family_specs(family: _SeventeenLandsFamily) -> None:
    """Reject an impossible family before any CSV is read.

    Inputs: family.
    Output: none.
    Side effects: none.
    Exceptions: ValueError if a deck-box spec meets a family with no
        deck_box_output_path.
    """
    has_deck_box_specs = any(
        isinstance(spec, DeckBoxChunkMetricSpec) for spec in family.metric_specs
    )
    if has_deck_box_specs and family.deck_box_output_path is None:
        raise ValueError(f"{family.name} has a deck-box metric but no deck box path")


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
        writes every metric's per-CSV output; saves the deck box, even
        when a CSV failed. A CSV whose scan raises is logged (a "CSV
        FAILURE" line with its traceback) and skipped; the rest still
        run, and the run ends with one line per failed CSV. A CSV in a
        layout the family does not read is logged as "CSV SKIPPED" and
        is not a failure.
    Exceptions: ValueError from _check_family_specs; SystemExit if no
        CSV is found, or (after every CSV and the deck box save) if any
        CSV failed.

    Example:
        >>> _run_seventeenlands_family(game_data_family, Path(
        ...     "data/raw/17lands/game_data/KTK.TradDraft.csv"), None)
    """
    # Validate the family before any CSV is read
    _check_family_specs(family)

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

    # Each CSV: one failing file is logged and skipped, never fatal
    run = _FamilyRun(family, binder, version_metadata, deck_box, output_root)
    outcomes: dict[Path, CsvScanOutcome] = {}
    try:
        for csv_path in csv_paths:
            outcomes[csv_path] = _scan_one_csv(run, csv_path)
    finally:
        # Save the box whatever happened: written partitions point into it
        if deck_box_path is not None:
            assert deck_box is not None
            deck_box.save(
                deck_box_path, GameId.MTG, version_metadata.card_binder_version
            )

    failed_csvs = _print_outcome_summary(outcomes)
    if failed_csvs:
        raise SystemExit(f"{len(failed_csvs)} of {len(csv_paths)} CSVs failed")


def _print_outcome_summary(outcomes: dict[Path, "CsvScanOutcome"]) -> list[Path]:
    """Print one summary line per skipped, then per failed, CSV.

    Inputs: outcomes (each scanned CSV's outcome, in scan order).
    Output: the failed CSVs, in scan order.
    Side effects: prints to stdout. Exceptions: none.
    """
    skipped = [p for p, o in outcomes.items() if o is CsvScanOutcome.SKIPPED]
    failed = [p for p, o in outcomes.items() if o is CsvScanOutcome.FAILED]
    for csv_path in skipped:
        print(f"CSV SKIPPED summary: {csv_path}")
    for csv_path in failed:
        print(f"CSV FAILURE summary: {csv_path}")
    return failed


class CsvScanOutcome(Enum):
    """How one CSV's scan in a family run ended."""

    SCANNED = "scanned"
    SKIPPED = "skipped"  # a layout the family does not read
    FAILED = "failed"


@dataclass(frozen=True)
class _FamilyRun:
    """What every CSV of one family run shares.

    family: the family being run.
    binder: the MTG CardBinder, loaded once.
    version_metadata: its version, hashed once.
    deck_box: the family box, or None for a family without one.
    output_root: None writes under data/metrics/.
    """

    family: _SeventeenLandsFamily
    binder: CardBinder
    version_metadata: MetricVersionMetadata
    deck_box: DeckBox | None
    output_root: Path | None


def _scan_one_csv(run: _FamilyRun, csv_path: Path) -> CsvScanOutcome:
    """Scan one CSV: build its parser and its metrics from its header,
    then run every metric over it.

    Inputs: run, csv_path.
    Output: SCANNED; SKIPPED if the parser rejects the CSV's layout as
        unsupported (logged as a "CSV SKIPPED" line, nothing written);
        FAILED if anything else raised (logged with its traceback, as a
        "CSV FAILURE" line).
    Side effects: writes the CSV's partitions; adds decks to deck_box.
    Exceptions: none (every exception is caught and logged).
    """
    try:
        expansion, format_code = _parse_expansion_format(csv_path)
        header = read_csv_header(csv_path)
        # The parser first: an unsupported layout is skipped before any
        # metric is built
        scan = run.family.scan_for_csv(header, run.binder)
        context = _CsvMetricContext(
            version_metadata=run.version_metadata,
            expansion=expansion,
            format_code=format_code,
            output_root=run.output_root,
            deck_box=run.deck_box,
        )
        metrics = _namespaced_metrics(run.family, context)

        print(
            f"=== {run.family.name}: {csv_path} ({expansion.value}.{format_code.value}) ==="
        )
        scan(csv_path, metrics)
        print(f"wrote {len(metrics)} metric outputs")
        return CsvScanOutcome.SCANNED
    except UnsupportedCsvLayout as layout:
        print(f"CSV SKIPPED {csv_path}: {layout}")
        return CsvScanOutcome.SKIPPED
    except Exception:  # one bad CSV must not end a multi-hour family run
        print(f"CSV FAILURE {csv_path}:\n{traceback.format_exc()}")
        return CsvScanOutcome.FAILED


def _deck_box_path(default_path: Path | None, output_root: Path | None) -> Path | None:
    """The family deck box path, re-rooted under output_root for a
    parity run so a parity run never writes the real box.

    Inputs: default_path, output_root. Output: Path or None.
    Side effects: none. Exceptions: none.
    """
    if default_path is None or output_root is None:
        return default_path
    return output_root / default_path.relative_to(METRICS_ROOT)


def _load_family_deck_box(path: Path | None) -> DeckBox | None:
    """The family deck box at path (created empty if the file doesn't
    exist yet), or None for a family without one.

    Always file-backed, never ":memory:": the box commits to path batch
    by batch, so a full-corpus run's memory stays flat (an in-memory
    replay_data box outgrew a 32 GB machine a quarter of the way in).

    Inputs: path. Output: DeckBox or None.
    Side effects: connects to path, creating it if missing.
    Exceptions: DeckBox.load's.
    """
    if path is None:
        return None
    return DeckBox.load([path])


class _ChunkParserFactory(Protocol):
    """A chunk parser's from_header (GameDataChunkParser,
    DraftDataChunkParser, ReplayDataChunkParser)."""

    def __call__(
        self, header: Sequence[str], card_binder: CardBinder, source_game: GameId
    ) -> Any: ...


@dataclass(frozen=True)
class _ChunkScanBuilder:
    """A chunk family's ScanForCsv: per CSV, a parser built from its
    header, bound into the family's typed scan function.

    make_parser: the parser's from_header.
    scan_csv: the family's scan (scan_game_csv, scan_draft_csv,
        scan_replay_csv), called as scan_csv(path, metrics,
        parser=parser).
    """

    make_parser: _ChunkParserFactory
    scan_csv: Callable[..., None]

    def __call__(self, header: Sequence[str], binder: CardBinder) -> FamilyScan:
        """This CSV's scan.

        Inputs: header, binder.
        Output: FamilyScan.
        Side effects: none (no I/O).
        Exceptions: make_parser's.

        Example:
            >>> _GAME_DATA_SCAN(header, binder)(csv_path, metrics)
        """
        parser = self.make_parser(header, binder, GameId.MTG)
        return partial(self.scan_csv, parser=parser)


_GAME_DATA_SCAN = _ChunkScanBuilder(GameDataChunkParser.from_header, scan_game_csv)
_DRAFT_DATA_SCAN = _ChunkScanBuilder(DraftDataChunkParser.from_header, scan_draft_csv)
_REPLAY_DATA_SCAN = _ChunkScanBuilder(
    ReplayDataChunkParser.from_header, scan_replay_csv
)


def run_seventeenlands_draft_data(
    raw_path: Path | None, output_root: Path | None = None
) -> None:
    """Run the draft_data family (chunk scan, chunk metrics only).

    Inputs: raw_path (one CSV, or None for every draft_data CSV),
        output_root (None, or a scratch root for parity runs).
    Output: none.
    Side effects: writes per-CSV outputs.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_draft_data(None)

    Runtime: about 3 h 20 min for all 70 draft_data CSVs (244 GB, 6 outputs
    each, 6.1 GB written), peak memory about 5 GB (measured 2026-10-09).
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_draft_data",
            data_type=refs.DataType.DRAFT,
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "draft_data",
            metric_specs=_DRAFT_DATA_METRICS,
            scan_for_csv=_DRAFT_DATA_SCAN,
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
    Side effects: writes per-CSV outputs. No deck box of its own:
        deck_uuid outputs use the canonical box's identity
        (extraction_stage.py); see game_data/README.md for its coverage.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_game_data(
        ...     Path("data/raw/17lands/game_data/KTK.TradDraft.csv"),
        ...     output_root=Path("scratch/parity"))

    Runtime: about 1 h 20 min for all 133 game_data CSVs (85 GB, 12 outputs
    each, 1.9 GB written) (measured 2026-10-09).
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_game_data",
            data_type=refs.DataType.GAME,
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data",
            metric_specs=_GAME_DATA_METRICS,
            scan_for_csv=_GAME_DATA_SCAN,
            deck_box_output_path=None,
        ),
        raw_path,
        output_root,
    )


def run_seventeenlands_replay_data(
    raw_path: Path | None, output_root: Path | None = None
) -> None:
    """Run the replay_data family (chunk scan, chunk metrics only).

    Inputs: raw_path (one CSV, or None for every replay_data CSV),
        output_root (None, or a scratch root for parity runs).
    Output: none.
    Side effects: writes per-CSV outputs and the family deck box.
    Exceptions: see _run_seventeenlands_family.

    Example:
        >>> run_seventeenlands_replay_data(None)

    Runtime: about 2 h 40 min for the 100 replay_data CSVs (149 GB; the 4
    AFR/STX files are skipped as an unreadable layout), with a file-backed deck
    box of about 30 GB, peak memory about 3.6 GB (measured 2026-10-09).
    """
    _run_seventeenlands_family(
        _SeventeenLandsFamily(
            name="seventeenlands_replay_data",
            data_type=refs.DataType.REPLAY,
            family_dir=SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "replay_data",
            metric_specs=_REPLAY_DATA_METRICS,
            scan_for_csv=_REPLAY_DATA_SCAN,
            deck_box_output_path=REPLAY_DATA_DECK_BOX_PATH,
        ),
        raw_path,
        output_root,
    )


class _SeventeenLandsRunner(Protocol):
    """A seventeenlands family runner: --raw-path, and an optional
    --output-root. Also a plain _FAMILIES runner (raw_path only)."""

    def __call__(
        self, raw_path: Path | None, output_root: Path | None = None
    ) -> None: ...


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
    "cross_game": run_cross_game,
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
