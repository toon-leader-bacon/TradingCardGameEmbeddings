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
_LATEST_DIRECTORY = "latest"


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

    def save_latest(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write the most recent round's weights, replacing the previous
        latest checkpoint.

        Inputs, output, exceptions: as save.
        Side effects: replaces the latest checkpoint only; bests are never
            touched.
        """
        ...


class DirectoryCheckpointer:
    """Writes checkpoint subdirectories under a run directory, keeping only
    the best checkpoint of each phase (`<phase>_round<NNNN>/`; a new best
    deletes the phase's previous one) and the latest round (`latest/`,
    replaced every round). Each checkpoint holds the full model, so without
    pruning a run would grow by about 1.2 GB (ModernBERT-base) or more
    (optimizer state when the encoder trains) per round.
    The checkpoints CSV still lists every best ever written; only the last
    per phase is on disk.

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
        # Phase name -> the directory of its current best checkpoint
        self._best_by_phase: dict[str, Path] = {}

    def save(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write a new best checkpoint for report's phase and delete that
        phase's previous best.

        Inputs: model, dojos (their trainable_parameters are saved),
            optimizer, plan, report.
        Output: CheckpointRecord with both paths.
        Side effects: creates files under run_directory; deletes the
            phase's previous best directory once the new one is complete.
        Exceptions: OSError on write failure (the previous best is kept).

        Example:
            >>> DirectoryCheckpointer(Path("runs/a")).save(m, ds, opt, plan, rep)
        """
        directory = self._directory_for(report)
        result = self._write_checkpoint(
            directory, model, dojos, optimizer, plan, report
        )

        # Only once the new best is complete: drop the phase's previous one
        previous = self._best_by_phase.get(report.phase)
        if previous is not None and previous != directory:
            shutil.rmtree(previous, ignore_errors=True)
        self._best_by_phase[report.phase] = directory
        return result

    def save_latest(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write run_directory/latest/, replacing the previous latest.

        Inputs, output, exceptions: as save.
        Side effects: replaces run_directory/latest/.

        Example:
            >>> checkpointer.save_latest(m, ds, opt, plan, rep).path.parent.name
            'latest'
        """
        directory = self._run_directory / _LATEST_DIRECTORY
        return self._write_checkpoint(directory, model, dojos, optimizer, plan, report)

    def _write_checkpoint(
        self,
        directory: Path,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        optimizer: torch.optim.Optimizer,
        plan: TrainingPlan,
        report: RoundReport,
    ) -> CheckpointRecord:
        """Write every checkpoint file into directory, replacing it whole.

        Inputs: directory, then as save. Output: CheckpointRecord.
        Side effects: replaces directory. Exceptions: OSError on write
            failure. A failure while writing files leaves directory as it
            was; a failure in the final swap (old directory deleted, rename
            fails) loses it - for latest/ that means no latest checkpoint
            until the next round.
        """
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
        """manifest.json: the report (each dojo's TEST loss, baseline and
        normalized loss) plus repr(plan), for reproducibility."""
        manifest = {
            "phase": report.phase,
            "round_index": report.round_index,
            "step": report.step,
            "elapsed_seconds": report.elapsed_seconds,
            "per_dojo_test_loss": {
                name: {
                    "loss": split_loss.loss,
                    "baseline_loss": split_loss.baseline_loss,
                    "normalized": split_loss.normalized,
                }
                for name, split_loss in report.per_dojo_test_loss.items()
            },
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
