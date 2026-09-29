import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.analyses.card_sample import AllCards, PerLabelCap
from src.evaluation.analyses.projection_plot import ProjectionPlot
from src.evaluation.analyses.projection_renderer import FigureStyle, ProjectionRenderer
from tests.evaluation.analyses.tables import labeled_table

_FAST = {"perplexity": 5.0, "max_iter": 250}


class _RecordingRenderer(ProjectionRenderer):
    """Draws as usual, remembering what each figure was asked to show."""

    def __init__(self, style: FigureStyle = FigureStyle()) -> None:
        super().__init__(style)
        self.highlighted: list[str] = []
        self.panel_labels: list[str] = []

    def draw_overview(self, coordinates, labels, highlighted, path):  # type: ignore[no-untyped-def]
        self.highlighted = list(highlighted)
        return super().draw_overview(coordinates, labels, highlighted, path)

    def draw_label_panels(  # type: ignore[no-untyped-def]
        self, coordinates, labels, panel_labels, path
    ):
        self.panel_labels = list(panel_labels)
        return super().draw_label_panels(coordinates, labels, panel_labels, path)


def _table(tmp_path: Path, sizes: dict[str, int]):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    vectors, labels = [], []
    for name, count in sizes.items():
        vectors.append(rng.standard_normal(8) * 3 + rng.standard_normal((count, 8)))
        labels += [name] * count
    return labeled_table(tmp_path / "t.db", np.vstack(vectors), labels)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_writes_both_figures_the_coordinates_and_scalars(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 12, "b": 10, "c": 8})
    out = tmp_path / "projection"
    with table:
        result = ProjectionPlot(labels, AllCards(), seed=0, **_FAST).run(table, out)
    names = [
        "projection.png",
        "projection_by_label.png",
        "projection.csv",
        "scalars.json",
    ]
    assert result.files == tuple(out / name for name in names)
    rows = _csv_rows(out / "projection.csv")
    assert len(rows) == 30
    assert list(rows[0]) == ["nocab_uuid", "source_game", "label", "x", "y"]
    assert {row["source_game"] for row in rows} == {"mtg"}
    assert {row["label"] for row in rows} == {"a", "b", "c"}
    scalars = json.loads((out / "scalars.json").read_text())
    assert scalars["n_cards"] == 30 and scalars["n_labels"] == 3
    assert math.isfinite(scalars["kl_divergence"])


def test_the_same_seed_gives_the_same_coordinates(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 12, "b": 12})
    plot = ProjectionPlot(labels, AllCards(), seed=4, **_FAST)
    with table:
        plot.run(table, tmp_path / "first")
        plot.run(table, tmp_path / "second")
    first = (tmp_path / "first" / "projection.csv").read_text()
    assert first == (tmp_path / "second" / "projection.csv").read_text()


def test_by_default_the_largest_labels_are_colored(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"small": 6, "big": 14, "mid": 10, "tiny": 5})
    renderer = _RecordingRenderer()
    with table:
        ProjectionPlot(labels, AllCards(), seed=0, renderer=renderer, **_FAST).run(
            table, tmp_path / "out"
        )
    assert renderer.highlighted == ["big", "mid", "small"]
    assert renderer.panel_labels == ["big", "mid", "small", "tiny"]


def test_tied_counts_rank_by_name(tmp_path: Path) -> None:
    # PerLabelCap(5) makes every label tie at five cards
    table, labels = _table(tmp_path, {"d": 9, "b": 9, "c": 9, "a": 9})
    renderer = _RecordingRenderer()
    with table:
        ProjectionPlot(labels, PerLabelCap(5), seed=0, renderer=renderer, **_FAST).run(
            table, tmp_path / "out"
        )
    assert renderer.highlighted == ["a", "b", "c"]
    assert renderer.panel_labels == ["a", "b", "c", "d"]


def test_panels_stop_at_max_panels_largest_first(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 6, "b": 12, "c": 9, "d": 7})
    renderer = _RecordingRenderer()
    with table:
        ProjectionPlot(
            labels, AllCards(), seed=0, max_panels=2, renderer=renderer, **_FAST
        ).run(table, tmp_path / "out")
    assert renderer.panel_labels == ["b", "c"]


def test_named_labels_are_colored_as_given(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 10, "b": 10, "c": 10})
    renderer = _RecordingRenderer()
    with table:
        ProjectionPlot(
            labels,
            AllCards(),
            seed=0,
            highlighted=("c", "a"),
            renderer=renderer,
            **_FAST
        ).run(table, tmp_path / "out")
    assert renderer.highlighted == ["c", "a"]


def test_a_custom_style_sets_the_limit(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 10, "b": 10, "c": 10})
    renderer = _RecordingRenderer(FigureStyle(series=("#2a78d6", "#eb6834")))
    with table:
        ProjectionPlot(labels, AllCards(), seed=0, renderer=renderer, **_FAST).run(
            table, tmp_path / "out"
        )
    assert renderer.highlighted == ["a", "b"]
    with pytest.raises(ValueError):
        ProjectionPlot(labels, AllCards(), seed=0, highlighted=3, renderer=renderer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"perplexity": 0.0},
        {"perplexity": float("nan")},
        {"max_iter": 249},
        {"max_panels": 0},
        {"highlighted": 0},
        {"highlighted": 4},
        {"highlighted": True},
        {"highlighted": ()},
        {"highlighted": ("a", "a")},
        {"highlighted": ("a", "b", "c", "d")},
    ],
)
def test_invalid_settings_are_rejected_on_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ProjectionPlot(None, AllCards(), seed=0, **kwargs)  # type: ignore[arg-type]


def test_too_few_cards_for_the_perplexity_writes_nothing(tmp_path: Path) -> None:
    table, labels = _table(tmp_path, {"a": 3, "b": 3})
    out = tmp_path / "out"
    with table, pytest.raises(ValueError, match="perplexity"):
        ProjectionPlot(labels, AllCards(), seed=0, perplexity=6.0).run(table, out)
    assert not out.exists()


def test_a_non_finite_kl_divergence_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, labels = _table(tmp_path, {"a": 10, "b": 10})

    def project(self: ProjectionPlot, vectors: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros((len(vectors), 2)), float("nan")

    monkeypatch.setattr(ProjectionPlot, "_project", project)
    with table:
        result = ProjectionPlot(labels, AllCards(), seed=0, **_FAST).run(
            table, tmp_path / "out"
        )
    assert "kl_divergence" not in result.scalars
    assert (tmp_path / "out" / "projection.png").exists()
