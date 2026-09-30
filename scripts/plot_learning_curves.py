"""Plot learning curves from rounds CSVs, independent of any run.

Any rounds CSV that CsvRunListener writes works here. That includes an
extrinsic run's (data/evaluations/<name>/extrinsic/<encoder>/rounds.csv) and
a training run's (models/.../rounds.csv). A run that was stopped or crashed
partway still has a usable CSV, one row per completed round. The script
draws one overlay chart per dojo, with one curve per labeled CSV.

Usage (from the project root):

    PYTHONPATH=. python scripts/plot_learning_curves.py \\
        --curve single=data/evaluations/v1/extrinsic/single/rounds.csv \\
        --curve untrained=data/evaluations/v1/extrinsic/untrained/rounds.csv \\
        --output-dir data/evaluations/v1/extrinsic/plots
"""

import argparse
from pathlib import Path
from typing import get_args

from src.evaluation.extrinsic.curve_renderer import XAxis
from src.evaluation.extrinsic.learning_curves import plot_learning_curves


def parse_curve(text: str) -> tuple[str, Path]:
    """One --curve argument, "label=path", as (label, path).

    Inputs: text. Output: (label, Path).
    Side effects: none.
    Exceptions: argparse.ArgumentTypeError if there is no "=" or either
        side is empty.

    Example:
        >>> parse_curve("single=runs/a/rounds.csv")
        ('single', PosixPath('runs/a/rounds.csv'))
    """
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def curves_by_label(pairs: list[tuple[str, Path]]) -> dict[str, Path]:
    """The --curve pairs as a label -> CSV mapping, in command-line order
    (which fixes each label's color).

    Inputs: pairs. Output: dict[str, Path].
    Side effects: none.
    Exceptions: ValueError if a label repeats.

    Example:
        >>> curves_by_label([("a", Path("a.csv")), ("b", Path("b.csv"))])
        {'a': PosixPath('a.csv'), 'b': PosixPath('b.csv')}
    """
    result: dict[str, Path] = {}
    for label, path in pairs:
        if label in result:
            raise ValueError(f"curve label {label!r} given twice")
        result[label] = path
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line. Inputs: argv (None: sys.argv). Output: Namespace
    with curve (list of (label, Path)), output_dir and x_axis.
    Side effects: none. Exceptions: SystemExit on bad arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--curve",
        type=parse_curve,
        action="append",
        required=True,
        help="label=path/to/rounds.csv (repeat for each curve)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-axis", choices=get_args(XAxis), default="step")
    return parser.parse_args(argv)


def main() -> None:
    """Plot every --curve into --output-dir and print the files written.
    Side effects: writes one PNG per dojo. Exceptions: as
    plot_learning_curves (e.g. a missing or malformed CSV)."""
    args = parse_args()
    plots = plot_learning_curves(
        curves_by_label(args.curve), args.output_dir, args.x_axis
    )
    for path in plots:
        print(path)


if __name__ == "__main__":
    main()
