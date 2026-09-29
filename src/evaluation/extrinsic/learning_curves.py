"""plot_learning_curves: overlay every encoder's per-round TEST loss, one
chart per dojo, from the rounds CSVs that run_extrinsic wrote.

Reads the CSVs only through read_rounds_csv - the writer's module owns the
format; evaluation never parses it. Unequal curve lengths plot fine:
rounds-to-saturation is itself a signal.
"""

import re
from pathlib import Path
from typing import Mapping

from src.evaluation.extrinsic.curve_renderer import CurveRenderer, XAxis
from src.training.recording.run_listener import RoundRow, read_rounds_csv

# Characters a plot file name may not contain. The same rule the
# checkpointer applies to phase names inline; promote it to one shared
# helper if a third use appears.
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def plot_learning_curves(
    curves: Mapping[str, Path],
    output_dir: Path,
    x_axis: XAxis = "step",
    *,
    renderer: CurveRenderer = CurveRenderer(),
) -> tuple[Path, ...]:
    """One overlay chart per dojo of every encoder's TEST loss vs x_axis.

    Inputs: curves (encoder label -> its rounds CSV, e.g.
        ExtrinsicResult.rounds_csv; the mapping's order fixes each
        encoder's color on every chart); output_dir (the plots directory
        itself); x_axis ("step" or "elapsed_seconds"); renderer (Strategy:
        how a chart is drawn; the reference palette by default).
    Output: the PNG paths written, one per dojo, in dojo-name order; each
        named after its dojo (made filename-safe). An encoder whose CSV has
        no rows (a run with no completed round) appears on no chart but
        keeps its color slot.
    Side effects: creates output_dir if missing; overwrites existing plot
        files of the same name (plots are cheap to regenerate).
    Exceptions: ValueError if curves is empty or has more encoders than
        renderer.style.max_encoders (checked before anything is read or
        written); FileNotFoundError / ValueError from read_rounds_csv for a
        missing or malformed CSV; ValueError if every CSV is empty
        (nothing to plot); ValueError if two dojo names make the
        same file name (e.g. "a b" and "a/b") - all before anything is
        written; OSError on a write failure.

    Example:
        >>> plot_learning_curves({"trained": a.rounds_csv,
        ...     "untrained": b.rounds_csv}, out / "extrinsic" / "plots")
        (PosixPath('.../pick.png'), PosixPath('.../winner.png'))
    """
    result: list[Path] = []
    _require_encoder_count(len(curves), renderer.style.max_encoders)

    # Read and group every CSV, and name every plot, before writing anything
    rows_by_encoder = {
        encoder: read_rounds_csv(path) for encoder, path in curves.items()
    }
    curves_by_dojo = _group_by_dojo(rows_by_encoder)
    if not curves_by_dojo:
        raise ValueError(f"no rows in any rounds CSV: {sorted(curves)}")
    plot_paths = _plot_paths(list(curves_by_dojo), output_dir)
    encoder_slots = {encoder: slot for slot, encoder in enumerate(curves)}

    # One chart per dojo seen in any CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    for dojo, dojo_curves in curves_by_dojo.items():
        result.append(
            renderer.draw_dojo_curves(
                dojo, dojo_curves, encoder_slots, x_axis, plot_paths[dojo]
            )
        )
    return tuple(result)


def _require_encoder_count(count: int, max_encoders: int) -> None:
    """Inputs: the encoder count, the renderer's limit. Output: None. Side
    effects: none. Exceptions: ValueError if count is 0 or above
    max_encoders (more lines than validated colors: facet instead)."""
    if not 1 <= count <= max_encoders:
        raise ValueError(
            f"can plot 1 to {max_encoders} encoders (one per color), got {count}"
        )


def _group_by_dojo(
    rows_by_encoder: Mapping[str, list[RoundRow]],
) -> dict[str, dict[str, list[RoundRow]]]:
    """Group every encoder's rows by dojo, in one pass.

    Inputs: encoder -> rows (file order).
    Output: dojo -> encoder -> that encoder's rows for the dojo (file
        order); dojos sorted by name; every encoder present under every
        dojo (empty list if it never scored that dojo). All phases' rows
        are kept: steps count up across phases, so a multi-phase training
        CSV still plots as one curve (an extrinsic CSV has one phase).
    Side effects: none. Exceptions: none.
    """
    result: dict[str, dict[str, list[RoundRow]]] = {}
    for encoder, rows in rows_by_encoder.items():
        for row in rows:
            result.setdefault(row.dojo, {}).setdefault(encoder, []).append(row)
    # Every encoder under every dojo, dojos in name order
    return {
        dojo: {encoder: result[dojo].get(encoder, []) for encoder in rows_by_encoder}
        for dojo in sorted(result)
    }


def _plot_paths(dojos: list[str], output_dir: Path) -> dict[str, Path]:
    """Name every dojo's plot before any is written.

    Inputs: dojos, the plots directory.
    Output: dojo -> output_dir / "<filename-safe dojo>.png", every
        character outside [A-Za-z0-9_.-] replaced by "_".
    Side effects: none.
    Exceptions: ValueError naming both dojos if two map to one file name,
        compared case-folded (the second chart would silently overwrite
        the first; "Pick" and "pick" collide on case-insensitive file
        systems such as macOS's default, as TrainingPlan's phase names do).
    """
    result: dict[str, Path] = {}
    owners: dict[str, str] = {}
    for dojo in dojos:
        name = _UNSAFE.sub("_", dojo)
        key = name.casefold()
        if key in owners:
            raise ValueError(
                f"dojos {owners[key]!r} and {dojo!r} would share the plot file {name}.png"
            )
        owners[key] = dojo
        result[dojo] = output_dir / f"{name}.png"
    return result
