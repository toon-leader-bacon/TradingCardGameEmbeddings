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
unit of work, not "one family, one output".

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_metrics.py --list
    PYTHONPATH=. python3 scripts/run_metrics.py --source sts_gg
    PYTHONPATH=. python3 scripts/run_metrics.py --source seventeenlands_game_data
    PYTHONPATH=. python3 scripts/run_metrics.py --source seventeenlands_game_data \\
        --raw-path "data/raw/17lands/game_data/MSH.PremierDraft.csv"
    PYTHONPATH=. python3 scripts/run_metrics.py --all

`--all` runs every family in this process, one after another (same
reasoning as the other two scripts' `--all` — nothing here is a
multi-hour crawl). One family's failure is logged and does not stop
the rest.
"""

import argparse
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.sts_gg.extraction_stage import (
    StsGgDeckExtractionStage,
)
from src.data_refinement.metrics.generic.corpus_scan_metric import CorpusScanMetric
from src.data_refinement.metrics.metric import Metric
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.schema.game_id import GameId

# --- sts_gg ---
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
from src.data_refinement.metrics.sts_gg.scanner import scan_runs_jsonl

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

# --- play_gwent ---
from src.data_refinement.metrics.play_gwent.leader_deck_counts import (
    DEFAULT_RAW_PATH as PLAY_GWENT_DEFAULT_RAW_PATH,
)
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
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
        Path("data/metrics/sts_gg/deck_box.db"),
        GameId.SLAY_THE_SPIRE_2,
        binder.version_for(GameId.SLAY_THE_SPIRE_2),
    )
    print(f"wrote {len(metrics)} metric outputs")


# --- gwent_one ---


def _reject_raw_path(name: str, raw_path: Path | None) -> None:
    """Exit if --raw-path was given to a family whose metrics scan an
    already-loaded CardBinder rather than one raw file."""
    if raw_path is not None:
        raise SystemExit(
            f"{name} metrics scan an already-loaded CardBinder, not a raw file "
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


# --- play_gwent ---


def run_play_gwent(raw_path: Path | None) -> None:
    effective_raw_path = raw_path or PLAY_GWENT_DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.GWENT,
        "run 'python3 scripts/run_card_binder_ingestion.py --source gwent_one' first.",
    )
    box_path = DeckBox.default_output_path(GameId.GWENT)
    deck_box = DeckBox.load([box_path] if box_path.exists() else [])  # warm start

    metrics: list[Metric[dict]] = [LeaderMaskedFromDeckMetric(binder, deck_box)]

    print(f"=== play_gwent: {effective_raw_path} ===")
    scan_guides_jsonl(effective_raw_path, metrics)
    deck_box.save(box_path, GameId.GWENT, binder.version_for(GameId.GWENT))
    print(f"wrote {len(metrics)} metric outputs")


# --- isotropic ---

_ISOTROPIC_RAW_DIR = Path("data/raw/isotropic")
# Shared by both isotropic families and warm-started on every run (deck
# uuids are content-derived, so re-adding a deck is a no-op) - running
# one family never drops the other family's decks from the box.
_ISOTROPIC_DECK_BOX_PATH = Path("data/metrics/isotropic/deck_box.db")
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
        [_ISOTROPIC_DECK_BOX_PATH] if _ISOTROPIC_DECK_BOX_PATH.exists() else []
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
        _ISOTROPIC_DECK_BOX_PATH, GameId.DOMINION, binder.version_for(GameId.DOMINION)
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
        _ISOTROPIC_DECK_BOX_PATH, GameId.DOMINION, binder.version_for(GameId.DOMINION)
    )
    print(f"wrote {len(header_metrics) + len(game_log_metrics)} metric outputs")


# --- seventeenlands ---


def _namespaced_output_path(
    default_path: Path, expansion: str, format_code: str
) -> Path:
    """default_path, relocated under <expansion>/<format_code> — keeps
    one metric's output from one CSV from overwriting its output from
    another (see module docstring)."""
    return default_path.parent / expansion / format_code / default_path.name


def _parse_expansion_format(csv_path: Path) -> tuple[str, str]:
    """ "<Expansion>.<EventType>.csv" -> (expansion, format_code) — the
    naming convention every 17lands raw CSV in this project follows."""
    expansion, format_code = csv_path.stem.split(".", 1)
    return expansion, format_code


@dataclass(frozen=True)
class _SeventeenLandsMetric:
    """One 17lands metric class a family scans, and whether its
    constructor takes the family's DeckBox (after binder/header/game)."""

    metric_class: Any  # a 17lands Metric subclass with DEFAULT_OUTPUT_PATH
    takes_deck_box: bool = False


_DRAFT_DATA_METRICS = (
    _SeventeenLandsMetric(CardTakeRateMetric),
    _SeventeenLandsMetric(FirstPickRateMetric),
    _SeventeenLandsMetric(RankStratifiedTakeRateMetric),
    _SeventeenLandsMetric(PickNumberDecayCurveMetric),
    _SeventeenLandsMetric(PackToPickChoiceSetMetric),
    _SeventeenLandsMetric(PoolConditionedPickMetric),
)

_GAME_DATA_METRICS = (
    _SeventeenLandsMetric(WinRateWhenInDeckMetric),
    _SeventeenLandsMetric(OpeningHandWinRateMetric),
    _SeventeenLandsMetric(DrawnWinRateMetric),
    _SeventeenLandsMetric(GameLengthAssociationMetric),
    _SeventeenLandsMetric(OnPlayWinRateDeltaMetric),
    _SeventeenLandsMetric(GameTutorTargetRateMetric),
    _SeventeenLandsMetric(DeckWinPredictionMetric, takes_deck_box=True),
    _SeventeenLandsMetric(DeckGameLengthPredictionMetric, takes_deck_box=True),
    _SeventeenLandsMetric(DeckRankTierPredictionMetric, takes_deck_box=True),
    _SeventeenLandsMetric(OnPlayWinRateSensitivityByDeckMetric, takes_deck_box=True),
    _SeventeenLandsMetric(TutorTargetPoolMetric),
)

_REPLAY_DATA_METRICS = (
    _SeventeenLandsMetric(AverageTurnCastMetric),
    _SeventeenLandsMetric(CastRateMetric),
    _SeventeenLandsMetric(TurnsToGameEndAfterCastMetric),
    _SeventeenLandsMetric(DiscardRateMetric),
    _SeventeenLandsMetric(ReplayTutorTargetRateMetric),
    _SeventeenLandsMetric(CombatKillInvolvementRateMetric),
    _SeventeenLandsMetric(CombatDamagePushThroughRateMetric),
    _SeventeenLandsMetric(CombatAggressionProfileMetric, takes_deck_box=True),
    _SeventeenLandsMetric(AttackerBlockerCombatOutcomeMetric),
)


def _namespaced_metrics(
    specs: Sequence[_SeventeenLandsMetric],
    binder: CardBinder,
    header: pd.Index,
    source_game: GameId,
    expansion: str,
    format_code: str,
    deck_box: DeckBox | None,
) -> list[Metric[dict]]:
    """Construct every metric in specs for one 17lands CSV, each writing
    to its DEFAULT_OUTPUT_PATH namespaced by expansion/format_code.

    Inputs: specs, plus the constructor arguments every 17lands metric
        shares (deck_box is passed only to specs that take it).
    Output: list[Metric[dict]], in specs order.
    Side effects: none beyond the metric constructors'.
    Exceptions: ValueError if a spec takes a deck box but deck_box is None.
    """
    metrics: list[Metric[dict]] = []
    for spec in specs:
        output_path = _namespaced_output_path(
            spec.metric_class.DEFAULT_OUTPUT_PATH, expansion, format_code
        )
        if not spec.takes_deck_box:
            args: tuple = (binder, header, source_game)
        elif deck_box is None:
            raise ValueError(f"{spec.metric_class.__name__} needs a DeckBox")
        else:
            args = (binder, header, source_game, deck_box)
        metrics.append(spec.metric_class(*args, output_path=output_path))
    return metrics


def _run_seventeenlands_family(
    name: str,
    family_dir: Path,
    metric_specs: Sequence[_SeventeenLandsMetric],
    scan: Callable[[Path, list[Metric[dict]]], None],
    deck_box_output_path: Path | None,
    raw_path: Path | None,
) -> None:
    binder = _require_binder(GameId.MTG, _MTG_BINDER_HINT)

    csv_paths = [raw_path] if raw_path is not None else sorted(family_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found under {family_dir}.")

    deck_box = None
    if deck_box_output_path is not None:
        deck_box = DeckBox.load(
            [deck_box_output_path] if deck_box_output_path.exists() else []
        )

    for csv_path in csv_paths:
        expansion, format_code = _parse_expansion_format(csv_path)
        header = pd.read_csv(csv_path, nrows=0).columns
        metrics = _namespaced_metrics(
            metric_specs, binder, header, GameId.MTG, expansion, format_code, deck_box
        )

        print(f"=== {name}: {csv_path} ({expansion}.{format_code}) ===")
        scan(csv_path, metrics)
        print(f"wrote {len(metrics)} metric outputs")

    if deck_box_output_path is not None:
        assert deck_box is not None
        deck_box.save(deck_box_output_path, GameId.MTG, binder.version_for(GameId.MTG))


def run_seventeenlands_draft_data(raw_path: Path | None) -> None:
    _run_seventeenlands_family(
        "seventeenlands_draft_data",
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "draft_data",
        _DRAFT_DATA_METRICS,
        scan_draft_csv,
        deck_box_output_path=None,
        raw_path=raw_path,
    )


def run_seventeenlands_game_data(raw_path: Path | None) -> None:
    _run_seventeenlands_family(
        "seventeenlands_game_data",
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data",
        _GAME_DATA_METRICS,
        scan_game_csv,
        deck_box_output_path=Path("data/metrics/seventeenlands/game_data/deck_box.db"),
        raw_path=raw_path,
    )


def run_seventeenlands_replay_data(raw_path: Path | None) -> None:
    _run_seventeenlands_family(
        "seventeenlands_replay_data",
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "replay_data",
        _REPLAY_DATA_METRICS,
        scan_replay_csv,
        deck_box_output_path=Path(
            "data/metrics/seventeenlands/replay_data/deck_box.db"
        ),
        raw_path=raw_path,
    )


_FAMILIES: dict[str, Callable[[Path | None], None]] = {
    "sts_gg": run_sts_gg,
    "gwent_one": run_gwent_one,
    "dominiontabs": run_dominiontabs,
    "play_gwent": run_play_gwent,
    "isotropic_summary": run_isotropic_summary,
    "isotropic_games": run_isotropic_games,
    "seventeenlands_draft_data": run_seventeenlands_draft_data,
    "seventeenlands_game_data": run_seventeenlands_game_data,
    "seventeenlands_replay_data": run_seventeenlands_replay_data,
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
        choices=sorted(_FAMILIES),
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
        return

    if args.source:
        _FAMILIES[args.source](args.raw_path)
        return

    run_all()


if __name__ == "__main__":
    main()
