"""Exercise the training/TODO.md section-C "first-run dojo set" once each,
before pointing an unattended Trainer.run() at them.

Constructs the gwent_one mask dojos and the sts_gg per-metric dojos
against real, already-ingested/scanned data, then runs
src.training.preflight.preflight_dojo on each: pull one TRAIN batch,
run compute_loss against random embeddings (no encoder, no GPU needed),
confirm the loss is a finite scalar. This is exactly the kind of check
that would have caught the staleness smoke_test_training_loop.py hit
(gwent_one's metrics built against a pre-re-ingest CardBinder, so every
dojo silently yielded zero examples and got quarantined with no
exception) - here it prints a clear failure per dojo instead.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/preflight_dojos.py
    PYTHONPATH=. python3 scripts/preflight_dojos.py --card-embedding-size 64
"""

import argparse
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.gwent_one.masked_field_dojos import (
    ArmorMaskDojo,
    ColorMaskDojo,
    FactionMaskDojo,
    PowerMaskDojo,
    ProvisionMaskDojo,
    RarityMaskDojo,
    SetMaskDojo,
    TypeMaskDojo,
)
from src.dojos.sts_gg.card_average_dojos import (
    CardDeckSizeDojo,
    CardElitesKilledDojo,
    CardFloorsClearedDojo,
    CardRelicCountDojo,
    CardTotalCardsPickedDojo,
    CardTotalCombatsDojo,
    CardTotalDamageTakenDojo,
    CardTotalTurnsDojo,
    CardWinRateDojo,
)
from src.dojos.sts_gg.deck_label_dojos import (
    CharacterDojo,
    DeckElitesKilledDojo,
    DeckFloorsClearedDojo,
    DeckRelicCountDojo,
    DeckTotalCardsPickedDojo,
    DeckTotalCardsSkippedDojo,
    DeckTotalCombatsDojo,
    DeckTotalDamageTakenDojo,
    DeckTotalTurnsDojo,
    WinDojo,
)
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.preflight import preflight_dojo

_STS_DECK_BOX_PATH = Path("data/metrics/sts_gg/deck_box.db")


def build_gwent_dojos(holdout: HoldoutSpec, card_embedding_size: int) -> list[Dojo]:
    """The 8 gwent_one masked-field dojos - "ready today" per
    training/TODO.md section C, no regeneration needed."""
    binder_path = CardBinder.default_output_path(GameId.GWENT)
    if not binder_path.exists():
        print(f"  skipping gwent_one: {binder_path} does not exist")
        return []
    binder = CardBinder.load([binder_path])
    dojo_classes = [
        ColorMaskDojo,
        FactionMaskDojo,
        RarityMaskDojo,
        SetMaskDojo,
        TypeMaskDojo,
        ArmorMaskDojo,
        ProvisionMaskDojo,
        PowerMaskDojo,
    ]
    return [
        dojo_class(binder, holdout, card_embedding_size) for dojo_class in dojo_classes
    ]


def build_sts_dojos(holdout: HoldoutSpec, card_embedding_size: int) -> list[Dojo]:
    """The 9 card-average dojos (need only a CardBinder) plus the
    deck-level dojos (need the metrics-private sts_gg deck box too, per
    scripts/run_metrics.py's run_sts_gg) - "ready today" per
    training/TODO.md section C."""
    binder_path = CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)
    if not binder_path.exists():
        print(f"  skipping sts_gg: {binder_path} does not exist")
        return []
    binder = CardBinder.load([binder_path])
    card_dojo_classes = [
        CardRelicCountDojo,
        CardTotalDamageTakenDojo,
        CardDeckSizeDojo,
        CardTotalCardsPickedDojo,
        CardTotalTurnsDojo,
        CardElitesKilledDojo,
        CardFloorsClearedDojo,
        CardTotalCombatsDojo,
        CardWinRateDojo,
    ]
    dojos: list[Dojo] = [
        dojo_class(binder, holdout, card_embedding_size)
        for dojo_class in card_dojo_classes
    ]
    if not _STS_DECK_BOX_PATH.exists():
        print(
            f"  skipping sts_gg deck-level dojos: {_STS_DECK_BOX_PATH} does not exist"
        )
        return dojos
    deck_box = DeckBox.load([_STS_DECK_BOX_PATH])
    deck_dojo_classes = [
        DeckRelicCountDojo,
        DeckTotalDamageTakenDojo,
        DeckTotalCardsPickedDojo,
        DeckTotalCardsSkippedDojo,
        DeckTotalTurnsDojo,
        DeckElitesKilledDojo,
        DeckFloorsClearedDojo,
        DeckTotalCombatsDojo,
        WinDojo,
        CharacterDojo,
    ]
    dojos.extend(
        dojo_class(binder, holdout, deck_box, card_embedding_size)
        for dojo_class in deck_dojo_classes
    )
    return dojos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--card-embedding-size",
        type=int,
        default=32,
        help="width the dojos' decoder heads are built for (default: 32, "
        "matching scripts/smoke_test_training_loop.py)",
    )
    parser.add_argument(
        "--max-batch-cost",
        type=int,
        default=8,
        help="BatchBudget.max_cost for the one TRAIN batch pulled per dojo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    holdout = HoldoutSpec.no_holdout()
    budget = BatchBudget(max_cost=args.max_batch_cost, cost_of=lambda card: 1)

    print("Building dojos...")
    dojos = build_gwent_dojos(holdout, args.card_embedding_size)
    dojos += build_sts_dojos(holdout, args.card_embedding_size)
    if not dojos:
        print("No dojos could be built (no card binder found) - nothing to check.")
        return 1

    print(
        f"\nChecking {len(dojos)} dojos (card_embedding_size={args.card_embedding_size}):"
    )
    results = [preflight_dojo(dojo, budget, args.card_embedding_size) for dojo in dojos]
    for result in results:
        status = "[ OK ]" if result.ok else "[FAIL]"
        print(
            f"{status} {result.dojo_name}: train={result.train_count} test={result.test_count}"
        )
        if result.ok:
            print(f"       sample loss: {result.sample_loss:.4g}")
        else:
            print(f"       {result.error}")

    passed = sum(result.ok for result in results)
    print(f"\n{passed}/{len(results)} dojos passed preflight")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
