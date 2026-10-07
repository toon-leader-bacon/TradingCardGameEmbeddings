"""Regenerates docs/metric_dojo_inventory.csv from the code: every metric
class that writes a parquet output, and the dojo(s) that read it.

A dojo is paired with a metric when the dojo class names that metric in
its own body (every per-metric wrapper reads its paired metric's
DEFAULT_OUTPUT_PATH/LABEL_COLUMN ClassVars, directly or via METRIC, or
passes it to seventeenlands_training_path), or
when a src/training/dojo_catalog.py recipe points the dojo class at that
metric's output (metric_output, e.g. the sts2_runs keys). A metric with
no dojo gets one row with an empty dojo column. A 17lands sliced metric
(OUTPUT_STEM instead of DEFAULT_OUTPUT_PATH) is listed with its "all"
slice file as its output. Needs no data on disk.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/report_metric_dojo_inventory.py
    PYTHONPATH=. python3 scripts/report_metric_dojo_inventory.py --check
"""

import argparse
import ast
import csv
import importlib
import inspect
import io
import sys
from dataclasses import dataclass
from typing import cast
from pathlib import Path

from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.slice_file import (
    SeventeenLandsSliceFile,
)
from src.data_refinement.metrics.seventeenlands.sliced_metric import (
    SlicedMetricClass,
)
from src.dojos.generic.generic_dojo import GenericDojo

_METRICS_ROOT = Path("src/data_refinement/metrics")
_DOJOS_ROOT = Path("src/dojos")
_OUTPUT_PATH = Path("docs/metric_dojo_inventory.csv")
_COLUMNS = ["metric_class", "metric_module", "metric_output", "dojo_class", "dojo_cell"]


@dataclass(frozen=True)
class _DojoClass:
    name: str
    cell: str
    metric_classes: tuple[type, ...]


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _metric_classes() -> list[type]:
    """Every class defined under metrics/ with a Path DEFAULT_OUTPUT_PATH,
    or (a 17lands sliced metric) a str OUTPUT_STEM."""
    metrics: list[type] = []
    for path in sorted(_METRICS_ROOT.rglob("*.py")):
        module = importlib.import_module(_module_name(path))
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            output = getattr(cls, "DEFAULT_OUTPUT_PATH", None)
            if isinstance(output, Path) or _is_sliced(cls):
                metrics.append(cls)
    return metrics


def _is_sliced(cls: type) -> bool:
    """Whether cls is a concrete 17lands sliced metric (a str OUTPUT_STEM;
    the abstract bases only annotate it)."""
    return isinstance(getattr(cls, "OUTPUT_STEM", None), str)


def _dojo_classes(metric_classes: list[type]) -> list[_DojoClass]:
    """Every concrete GenericDojo subclass outside dojos/generic/, with the
    metric classes its body references (names resolved through the dojo
    module's own imports, so same-named metrics in two games stay apart)."""
    dojos: list[_DojoClass] = []
    for path in sorted(_DOJOS_ROOT.rglob("*.py")):
        if "generic" in path.parts:
            continue
        module = importlib.import_module(_module_name(path))
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(module, node.name)
            if not issubclass(cls, GenericDojo):
                continue
            referenced = [getattr(module, n, None) for n in _paired_metric_names(node)]
            paired = tuple(m for m in metric_classes if any(m is r for r in referenced))
            dojos.append(_DojoClass(cls.__name__, _cell_of(cls), paired))
    return dojos


def _cell_of(dojo_class: type) -> str:
    """The generic cell (src/dojos/generic/<cell>/dojo.py) dojo_class
    builds on."""
    return next(
        base.__name__
        for base in dojo_class.__mro__[1:]
        if base.__module__.startswith("src.dojos.generic.")
        and base.__module__.endswith(".dojo")
    )


def _catalog_dojo_classes(metric_classes: list[type]) -> list[_DojoClass]:
    """One entry per dojo_catalog recipe that points a dojo class at
    another metric's output (metric_output), paired with that metric.

    Inputs: metric_classes (every metric class found).
    Output: list[_DojoClass].
    Side effects: imports the training catalog. Exceptions: ImportError.
    """
    # Imported here: only this pairing needs the training catalog
    from src.training.dojo_catalog import (
        DOJO_CATALOG,
        CardDojoRecipe,
        DeckDojoRecipe,
    )

    dojos: list[_DojoClass] = []
    for recipe in DOJO_CATALOG.values():
        if not isinstance(recipe, (CardDojoRecipe, DeckDojoRecipe)):
            continue
        if recipe.metric_output is None:
            continue
        # Every catalog dojo constructor is a dojo class
        dojo_class = cast(type, recipe.dojo_class)
        paired = tuple(
            m for m in metric_classes if _output_path(m) == recipe.metric_output
        )
        dojos.append(_DojoClass(dojo_class.__name__, _cell_of(dojo_class), paired))
    return dojos


def _paired_metric_names(class_node: ast.ClassDef) -> set[str]:
    """Names a dojo class reads its training file from: X in
    `X.DEFAULT_OUTPUT_PATH`, in `METRIC = X` (paired_metric_dojos), or in
    `seventeenlands_training_path(X, ...)`."""
    names: set[str] = set()
    for node in ast.walk(class_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "seventeenlands_training_path"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            names.add(node.args[0].id)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "DEFAULT_OUTPUT_PATH"
            and isinstance(node.value, ast.Name)
        ):
            names.add(node.value.id)
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "METRIC" for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            names.add(node.value.id)
    return names


def _output_path(metric: type) -> Path:
    """metric's output: its DEFAULT_OUTPUT_PATH, or a sliced metric's
    "all" slice file."""
    if _is_sliced(metric):
        sliced = cast(SlicedMetricClass, metric)
        return SeventeenLandsSliceFile(sliced).path_for(SeventeenLandsSlice())
    return Path(getattr(metric, "DEFAULT_OUTPUT_PATH"))


def inventory_rows() -> list[dict[str, str]]:
    """One row per (metric, dojo) pairing, plus one per unpaired metric.

    Inputs: none (reads the source tree under src/).
    Output: list of dicts keyed by _COLUMNS, sorted by metric output path.
    Side effects: imports every metric and dojo module.
    Exceptions: ImportError if a module fails to import.
    """
    metrics = _metric_classes()
    dojos = _dojo_classes(metrics) + _catalog_dojo_classes(metrics)
    rows: list[dict[str, str]] = []
    for metric in sorted(metrics, key=lambda m: (str(_output_path(m)), m.__name__)):
        paired: list[_DojoClass | None] = [
            dojo for dojo in dojos if metric in dojo.metric_classes
        ]
        for dojo in paired or [None]:
            rows.append(
                {
                    "metric_class": metric.__name__,
                    "metric_module": metric.__module__,
                    "metric_output": _output_path(metric).as_posix(),
                    "dojo_class": dojo.name if dojo else "",
                    "dojo_cell": dojo.cell if dojo else "",
                }
            )
    return rows


def _as_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed CSV is out of date, instead of writing it",
    )
    args = parser.parse_args()
    content = _as_csv(inventory_rows())
    if args.check:
        current = _OUTPUT_PATH.read_text() if _OUTPUT_PATH.exists() else ""
        if current != content:
            sys.exit(f"{_OUTPUT_PATH} is out of date; re-run without --check")
        print(f"{_OUTPUT_PATH} is up to date")
        return
    _OUTPUT_PATH.write_text(content)
    print(f"wrote {content.count(chr(10)) - 1} rows to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
