"""Runs src/data_refinement/metrics scanners — one raw-source metric
family at a time, or all of them — each converting already-ingested
CardBinder + raw source data into training-data parquet files under
data/metrics/.

Unlike run_card_binder_ingestion.py/run_deck_box_ingestion.py, this
container isn't one uniform shape (see src/data_refinement/metrics/
README.md): each family below is a direct, runnable copy of that
family's own README "How to run" recipe — sts_gg, gwent_one, and
play_gwent each read exactly one fixed raw file/corpus, while
seventeenlands's three families (draft_data/game_data/replay_data)
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
from typing import Callable

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


def run_sts_gg(raw_path: Path | None) -> None:
    resolved_raw_path = raw_path or StsGgDeckExtractionStage.DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.SLAY_THE_SPIRE_2,
        "run 'python3 scripts/run_card_binder_ingestion.py --source spire_codex' first.",
    )
    deck_box = DeckBox()  # metrics-private — see sts_gg/README.md's "Deck references"

    metrics: list[Metric[dict]] = [
        CardUpgradeRateMetric(binder),
        CardWinRateAtAct2Metric(binder),
        AscensionPredictionMetric(binder, deck_box),
        RelicCountMetric(binder, deck_box),
        DeckCharacterPredictionMetric(binder, deck_box),
        TotalDamageTakenMetric(binder, deck_box),
        TotalCardsPickedMetric(binder, deck_box),
        TotalCardsSkippedMetric(binder, deck_box),
        TotalTurnsMetric(binder, deck_box),
        ElitesKilledMetric(binder, deck_box),
        FloorsClearedMetric(binder, deck_box),
        TotalCombatsMetric(binder, deck_box),
        KilledByMetric(binder, deck_box),
        WinMetric(binder, deck_box),
        CardRelicCountMetric(binder),
        CardTotalDamageTakenMetric(binder),
        CardDeckSizeMetric(binder),
        CardTotalCardsPickedMetric(binder),
        CardTotalTurnsMetric(binder),
        CardElitesKilledMetric(binder),
        CardFloorsClearedMetric(binder),
        CardTotalCombatsMetric(binder),
        CardWinRateMetric(binder),
        CardCharacterPredictionMetric(binder),
    ]

    print(f"=== sts_gg: {resolved_raw_path} ===")
    scan_runs_jsonl(resolved_raw_path, metrics)
    deck_box.save(
        Path("data/metrics/sts_gg/deck_box.jsonl"),
        GameId.SLAY_THE_SPIRE_2,
        binder.version_for(GameId.SLAY_THE_SPIRE_2),
    )
    print(f"wrote {len(metrics)} metric outputs")


# --- gwent_one ---


def run_gwent_one(raw_path: Path | None) -> None:
    if raw_path is not None:
        raise SystemExit(
            "gwent_one metrics scan an already-loaded CardBinder, not a raw file "
            "— --raw-path is not applicable."
        )
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

    print("=== gwent_one ===")
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


# --- play_gwent ---


def run_play_gwent(raw_path: Path | None) -> None:
    resolved_raw_path = raw_path or PLAY_GWENT_DEFAULT_RAW_PATH
    binder = _require_binder(
        GameId.GWENT,
        "run 'python3 scripts/run_card_binder_ingestion.py --source gwent_one' first.",
    )
    box_path = DeckBox.default_output_path(GameId.GWENT)
    deck_box = DeckBox.load([box_path] if box_path.exists() else [])  # warm start

    metrics: list[Metric[dict]] = [LeaderMaskedFromDeckMetric(binder, deck_box)]

    print(f"=== play_gwent: {resolved_raw_path} ===")
    scan_guides_jsonl(resolved_raw_path, metrics)
    deck_box.save(box_path, GameId.GWENT, binder.version_for(GameId.GWENT))
    print(f"wrote {len(metrics)} metric outputs")


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


def _draft_data_metrics(
    binder: CardBinder,
    header: pd.Index,
    source_game: GameId,
    expansion: str,
    format_code: str,
    deck_box: DeckBox | None,
) -> list[Metric[dict]]:
    return [
        CardTakeRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                CardTakeRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        FirstPickRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                FirstPickRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        RankStratifiedTakeRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                RankStratifiedTakeRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        PickNumberDecayCurveMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                PickNumberDecayCurveMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        PackToPickChoiceSetMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                PackToPickChoiceSetMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        PoolConditionedPickMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                PoolConditionedPickMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
    ]


def _game_data_metrics(
    binder: CardBinder,
    header: pd.Index,
    source_game: GameId,
    expansion: str,
    format_code: str,
    deck_box: DeckBox | None,
) -> list[Metric[dict]]:
    assert deck_box is not None

    return [
        WinRateWhenInDeckMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                WinRateWhenInDeckMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        OpeningHandWinRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                OpeningHandWinRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        DrawnWinRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                DrawnWinRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        GameLengthAssociationMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                GameLengthAssociationMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        OnPlayWinRateDeltaMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                OnPlayWinRateDeltaMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        GameTutorTargetRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                GameTutorTargetRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        DeckWinPredictionMetric(
            binder,
            header,
            source_game,
            deck_box,
            output_path=_namespaced_output_path(
                DeckWinPredictionMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        DeckGameLengthPredictionMetric(
            binder,
            header,
            source_game,
            deck_box,
            output_path=_namespaced_output_path(
                DeckGameLengthPredictionMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        DeckRankTierPredictionMetric(
            binder,
            header,
            source_game,
            deck_box,
            output_path=_namespaced_output_path(
                DeckRankTierPredictionMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        OnPlayWinRateSensitivityByDeckMetric(
            binder,
            header,
            source_game,
            deck_box,
            output_path=_namespaced_output_path(
                OnPlayWinRateSensitivityByDeckMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        TutorTargetPoolMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                TutorTargetPoolMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
    ]


def _replay_data_metrics(
    binder: CardBinder,
    header: pd.Index,
    source_game: GameId,
    expansion: str,
    format_code: str,
    deck_box: DeckBox | None,
) -> list[Metric[dict]]:
    assert deck_box is not None

    return [
        AverageTurnCastMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                AverageTurnCastMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        CastRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                CastRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        TurnsToGameEndAfterCastMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                TurnsToGameEndAfterCastMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        DiscardRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                DiscardRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        ReplayTutorTargetRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                ReplayTutorTargetRateMetric.DEFAULT_OUTPUT_PATH, expansion, format_code
            ),
        ),
        CombatKillInvolvementRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                CombatKillInvolvementRateMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        CombatDamagePushThroughRateMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                CombatDamagePushThroughRateMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        CombatAggressionProfileMetric(
            binder,
            header,
            source_game,
            deck_box,
            output_path=_namespaced_output_path(
                CombatAggressionProfileMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
        AttackerBlockerCombatOutcomeMetric(
            binder,
            header,
            source_game,
            output_path=_namespaced_output_path(
                AttackerBlockerCombatOutcomeMetric.DEFAULT_OUTPUT_PATH,
                expansion,
                format_code,
            ),
        ),
    ]


def _run_seventeenlands_family(
    name: str,
    family_dir: Path,
    build_metrics: Callable[
        [CardBinder, pd.Index, GameId, str, str, DeckBox | None], list[Metric[dict]]
    ],
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
        metrics = build_metrics(
            binder, header, GameId.MTG, expansion, format_code, deck_box
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
        _draft_data_metrics,
        scan_draft_csv,
        deck_box_output_path=None,
        raw_path=raw_path,
    )


def run_seventeenlands_game_data(raw_path: Path | None) -> None:
    _run_seventeenlands_family(
        "seventeenlands_game_data",
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data",
        _game_data_metrics,
        scan_game_csv,
        deck_box_output_path=Path(
            "data/metrics/seventeenlands/game_data/deck_box.jsonl"
        ),
        raw_path=raw_path,
    )


def run_seventeenlands_replay_data(raw_path: Path | None) -> None:
    _run_seventeenlands_family(
        "seventeenlands_replay_data",
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "replay_data",
        _replay_data_metrics,
        scan_replay_csv,
        deck_box_output_path=Path(
            "data/metrics/seventeenlands/replay_data/deck_box.jsonl"
        ),
        raw_path=raw_path,
    )


_FAMILIES: dict[str, Callable[[Path | None], None]] = {
    "sts_gg": run_sts_gg,
    "gwent_one": run_gwent_one,
    "play_gwent": run_play_gwent,
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
