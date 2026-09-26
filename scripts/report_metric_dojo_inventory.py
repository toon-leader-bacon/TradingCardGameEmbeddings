"""Regenerates docs/metric_dojo_inventory.csv from the code: every metric
class that writes a parquet output, and the dojo(s) that read it.

A dojo is paired with a metric when the dojo class names that metric in
its own body (every per-metric wrapper reads its paired metric's
DEFAULT_OUTPUT_PATH/LABEL_COLUMN ClassVars, directly or via METRIC). A
metric with no dojo gets one row with an empty dojo column. Needs no
data on disk.

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
from pathlib import Path

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
    """Every class defined under metrics/ with a Path DEFAULT_OUTPUT_PATH."""
    metrics: list[type] = []
    for path in sorted(_METRICS_ROOT.rglob("*.py")):
        module = importlib.import_module(_module_name(path))
        for _, cls in inspect.getmembers(module, inspect.isclass):
            output = getattr(cls, "DEFAULT_OUTPUT_PATH", None)
            if cls.__module__ == module.__name__ and isinstance(output, Path):
                metrics.append(cls)
    return metrics


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
            cell = next(
                base.__name__
                for base in cls.__mro__[1:]
                if base.__module__.startswith("src.dojos.generic.")
                and base.__module__.endswith(".dojo")
            )
            dojos.append(_DojoClass(cls.__name__, cell, paired))
    return dojos


def _paired_metric_names(class_node: ast.ClassDef) -> set[str]:
    """Names a dojo class reads its training file from: X in
    `X.DEFAULT_OUTPUT_PATH`, or in `METRIC = X` (paired_metric_dojos)."""
    names: set[str] = set()
    for node in ast.walk(class_node):
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
    return Path(getattr(metric, "DEFAULT_OUTPUT_PATH"))


def inventory_rows() -> list[dict[str, str]]:
    """One row per (metric, dojo) pairing, plus one per unpaired metric.

    Inputs: none (reads the source tree under src/).
    Output: list of dicts keyed by _COLUMNS, sorted by metric output path.
    Side effects: imports every metric and dojo module.
    Exceptions: ImportError if a module fails to import.
    """
    metrics = _metric_classes()
    dojos = _dojo_classes(metrics)
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
