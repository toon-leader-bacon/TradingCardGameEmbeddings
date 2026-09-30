import argparse
from pathlib import Path

import pytest

from scripts.plot_learning_curves import curves_by_label, parse_args, parse_curve


def test_a_curve_is_a_label_and_a_path() -> None:
    assert parse_curve("single=runs/a/rounds.csv") == (
        "single",
        Path("runs/a/rounds.csv"),
    )


def test_only_the_first_equals_sign_splits() -> None:
    assert parse_curve("a=dir=x/rounds.csv") == ("a", Path("dir=x/rounds.csv"))


@pytest.mark.parametrize("text", ["rounds.csv", "=rounds.csv", "single="])
def test_a_curve_needs_both_a_label_and_a_path(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_curve(text)


def test_curves_keep_command_line_order() -> None:
    result = curves_by_label([("b", Path("b.csv")), ("a", Path("a.csv"))])
    assert list(result) == ["b", "a"]


def test_a_repeated_label_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        curves_by_label([("a", Path("1.csv")), ("a", Path("2.csv"))])


def test_the_command_line_collects_every_curve() -> None:
    args = parse_args(
        ["--curve", "a=1.csv", "--curve", "b=2.csv", "--output-dir", "out"]
    )
    assert args.curve == [("a", Path("1.csv")), ("b", Path("2.csv"))]
    assert args.output_dir == Path("out")
    assert args.x_axis == "step"
