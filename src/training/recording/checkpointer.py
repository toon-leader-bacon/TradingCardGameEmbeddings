"""Persisting a run: a weights snapshot plus the encoder-only artifact,
and loading a checkpoint's encoder weights and HoldoutSpec back (the
module that writes a checkpoint also reads it)."""

import json
import re
import shutil
from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch import nn

from src.dojos.dojo import Dojo
from src.schema.holdout import HoldoutSpec
from src.training.plan import TrainingPlan
from src.training.recording.reports import CheckpointRecord, RoundReport
from src.training.trainable_encoder import TrainableEncoder

_STATE_FILE = "state.pt"
_ENCODER_FILE = "encoder.pt"
_MANIFEST_FILE = "manifest.json"
_HOLDOUT_FILE = "holdout.json"


class Checkpointer(Protocol):
    def save(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write a checkpoint.

        Inputs: model, dojos, optimizer, plan, report.
        Output: CheckpointRecord. Side effects: writes files.
        Exceptions: OSError on write failure (Trainer catches, logs and
            carries on). Implementations write to a temp directory and
            rename, so a crash never leaves a half-written checkpoint.
        """
        ...


class DirectoryCheckpointer:
    """Writes one subdirectory per checkpoint under a run directory.

    Each holds `state.pt` (a weights snapshot: model + optimizer + dojo
    heads, keyed by dojo name; NOT yet resumable mid-run since tracker/rng/
    phase position are not saved) and
    `encoder.pt` (encoder_only_state_dict(), the seam for Hugging Face
    export), `holdout.json` (plan.holdout, via HoldoutSpec.to_json, so
    evaluation can label cards by the tiers the run trained with), plus
    the plan/report manifest.

    Inputs (constructor): run_directory (Path).
    """

    def __init__(self, run_directory: Path) -> None:
        self._run_directory = run_directory

    def save(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write a checkpoint.

        Inputs: model, dojos (their trainable_parameters are saved),
            optimizer, plan, report.
        Output: CheckpointRecord with both paths.
        Side effects: creates files under run_directory.
        Exceptions: OSError on write failure.

        Example:
            >>> DirectoryCheckpointer(Path("runs/a")).save(m, ds, opt, plan, rep)
        """
        # Choose this checkpoint's directory from the report
        directory = self._directory_for(report)
        staging = directory.with_name(directory.name + ".tmp")
        # Write into a staging directory, then rename: a crash mid-write
        # never leaves a half-written checkpoint under the final name
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            self._write_state(staging, model, dojos, optimizer)
            self._write_encoder(staging, model)
            self._write_manifest(staging, plan, report)
            self._write_holdout(staging, plan.holdout)
            shutil.rmtree(directory, ignore_errors=True)
            staging.rename(directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return CheckpointRecord(
            directory / _STATE_FILE, directory / _ENCODER_FILE, report
        )

    def _directory_for(self, report: RoundReport) -> Path:
        """run_directory/<phase>_round<NNNN>, phase name made filename-safe."""
        phase = re.sub(r"[^A-Za-z0-9_.-]", "_", report.phase)
        return self._run_directory / f"{phase}_round{report.round_index:04d}"

    def _write_state(
        self,
        directory: Path,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """model + optimizer state and each dojo's head weights (positional
        lists keyed by dojo name)."""
        heads = {
            dojo.name: [p.detach().cpu() for p in dojo.trainable_parameters()]
            for dojo in dojos
        }
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dojo_heads": heads,
        }
        torch.save(state, directory / _STATE_FILE)

    def _write_encoder(self, directory: Path, model: TrainableEncoder) -> None:
        """The encoder-only weights: the published artifact."""
        torch.save(dict(model.encoder_only_state_dict()), directory / _ENCODER_FILE)

    def _write_manifest(
        self, directory: Path, plan: TrainingPlan, report: RoundReport
    ) -> None:
        """manifest.json: the report plus repr(plan), for reproducibility."""
        manifest = {
            "phase": report.phase,
            "round_index": report.round_index,
            "step": report.step,
            "elapsed_seconds": report.elapsed_seconds,
            "per_dojo_test_loss": dict(report.per_dojo_test_loss),
            "statuses": {n: s.value for n, s in report.statuses.items()},
            "quarantined": sorted(report.quarantined),
            "plan": repr(plan),
        }
        (directory / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    def _write_holdout(self, directory: Path, holdout: HoldoutSpec) -> None:
        """holdout.json: the run's HoldoutSpec, losslessly."""
        (directory / _HOLDOUT_FILE).write_text(holdout.to_json())


def load_checkpoint_holdout(checkpoint_dir: Path) -> HoldoutSpec:
    """The HoldoutSpec the checkpoint's run trained with.

    Inputs: checkpoint_dir (Path), one directory DirectoryCheckpointer wrote.
    Output: HoldoutSpec parsed from checkpoint_dir/holdout.json.
    Side effects: reads the file.
    Exceptions: FileNotFoundError if holdout.json is absent (checkpoints
        written before it existed); ValueError if it does not parse.

    Example:
        >>> load_checkpoint_holdout(Path("runs/a/pretrain_round0007"))
        HoldoutSpec(seed=0, tier_ratios=(8.0, 1.0, 1.0), held_out_games=frozenset())
    """
    return HoldoutSpec.from_json((checkpoint_dir / _HOLDOUT_FILE).read_text())


def load_encoder_weights(checkpoint_dir: Path, model: nn.Module) -> None:
    """Load the checkpoint's encoder.pt into a caller-constructed model.

    The caller builds the same architecture the run trained; the weights
    are loaded onto the CPU, then copied onto whatever device model is on.

    Inputs: checkpoint_dir (Path), model (nn.Module).
    Output: None.
    Side effects: overwrites model's weights in place.
    Exceptions: FileNotFoundError if encoder.pt is absent; RuntimeError if
        its keys or shapes do not match model (strict load).

    Example:
        >>> model = SingleCardModel(text_encoder, head)
        >>> load_encoder_weights(Path("runs/a/pretrain_round0007"), model)
    """
    # Read on the CPU with weights_only (no pickled code), then strict-load
    weights = torch.load(
        checkpoint_dir / _ENCODER_FILE, map_location="cpu", weights_only=True
    )
    model.load_state_dict(weights, strict=True)
