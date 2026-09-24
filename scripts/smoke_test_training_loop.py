"""End-to-end smoke test of the actual training loop (Trainer.run()),
against real data - the first time this project's Trainer has been run
against anything other than fake dojos/a fake encoder.

Unlike smoke_test.py (which only proves ingestion -> metric -> dojo
batches wiring, echoing data to stdout with no model), this script
builds a real model and calls Trainer.run() on it: encoder -> dojo
heads -> loss -> optimizer step -> checkpoint, exactly what a real run
does, just tiny (few steps, few rounds, frozen encoder, CPU-sized
batches) and pointed at a throwaway run directory.

Uses two already-ingested, already-scanned gwent_one dojos
(ColorMaskDojo, FactionMaskDojo - see src/training/TODO.md section C
for why these were picked: ready today, no regeneration needed) against
a frozen LinearProjectionCardModel (src/training/TODO.md section B:
ModernBERT-base + a linear head). No holdout, since this only checks
that the loop runs, not that the model learns anything.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/smoke_test_training_loop.py
"""

import argparse
import logging
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.gwent_one.masked_field_dojos import ColorMaskDojo, FactionMaskDojo
from src.encoder_model.reference_singlecard_models import LinearProjectionCardModel
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.plan import (
    FaultPolicy,
    HardwareLimits,
    Phase,
    SaturationSpec,
    TrainingPlan,
    Uniform,
)
from src.training.recording.checkpointer import DirectoryCheckpointer
from src.training.recording.run_listener import LoggingRunListener
from src.training.trainer import Trainer

_DEFAULT_BINDER_PATH = Path("data/final/cards/gwent.jsonl")
_DEFAULT_RUN_DIRECTORY = Path("data/tmp/smoke_test/runs/training_loop")
_EMBED_DIM = 32


def build_plan(dojo_names: tuple[str, ...], holdout: HoldoutSpec) -> TrainingPlan:
    """A single tiny, frozen-encoder phase over dojo_names.

    Inputs: dojo_names (must match the constructed dojos' own .name),
        holdout (must match every dojo's own .holdout).
    Output: a TrainingPlan with one Phase: 5 steps/round, 3 rounds,
        never saturating early (patience_rounds exceeds max_rounds) -
        this smoke test only needs the loop to run its full course,
        not to demonstrate convergence.
    Side effects: none. Exceptions: whatever TrainingPlan/Phase's own
        validation raises for a malformed input.
    """
    phase = Phase(
        name="smoke",
        dojo_names=dojo_names,
        diet_rule=Uniform(),
        encoder_trainable=False,
        encoder_lr=0.0,
        head_lr=1e-3,
        steps_per_round=5,
        max_rounds=3,
        saturation=SaturationSpec(
            epsilon=0.0,
            patience_rounds=10,
            reactivation_delta=0.0,
            target_saturated_fraction=1.0,
        ),
    )
    return TrainingPlan(
        phases=(phase,),
        holdout=holdout,
        held_out_dojos=frozenset(),
        eval_examples_per_dojo=32,
        seed=0,
        faults=FaultPolicy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binder-path", type=Path, default=_DEFAULT_BINDER_PATH)
    parser.add_argument("--run-directory", type=Path, default=_DEFAULT_RUN_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    print(f"[1/3] Loading gwent CardBinder from {args.binder_path}")
    card_binder = CardBinder.load([args.binder_path])
    print(f"      {len(list(card_binder.all_cards(GameId.GWENT)))} cards loaded")

    print("[2/3] Building dojos (color_mask, faction_mask) and the model")
    holdout = HoldoutSpec.no_holdout()
    dojos = [
        ColorMaskDojo(card_binder, holdout, _EMBED_DIM, rng_seed=0),
        FactionMaskDojo(card_binder, holdout, _EMBED_DIM, rng_seed=0),
    ]
    model = LinearProjectionCardModel(embed_dim=_EMBED_DIM)
    plan = build_plan(tuple(dojo.name for dojo in dojos), holdout)

    print("[3/3] Running Trainer.run()")
    result = Trainer(
        model=model,
        dojos=dojos,
        plan=plan,
        limits=HardwareLimits(max_batch_cost=8),
        checkpointer=DirectoryCheckpointer(args.run_directory),
        listeners=[LoggingRunListener()],
    ).run()

    print(f"stopped_early_reason: {result.stopped_early_reason}")
    if result.final_report is not None:
        print(
            f"final round per-dojo test loss: {result.final_report.per_dojo_test_loss}"
        )
    if result.best_checkpoint is not None:
        print(f"best checkpoint: {result.best_checkpoint.path}")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
