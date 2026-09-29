from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.evaluation.analyses.projection_renderer import FigureStyle, ProjectionRenderer

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestFigureStyle:
    def test_defaults_allow_three_highlighted_labels(self) -> None:
        assert FigureStyle().max_highlighted == 3

    def test_the_limit_follows_the_series(self) -> None:
        assert FigureStyle(series=("#000000", "#ffffff")).max_highlighted == 2

    @pytest.mark.parametrize(
        "overrides",
        [{"series": ()}, {"marker_area": 0.0}, {"max_panel_columns": 0}],
    )
    def test_invalid_fields_are_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            FigureStyle(**overrides)

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            FigureStyle().marker_area = 10.0  # type: ignore[misc]


class TestProjectionRenderer:
    def test_style_is_read_only(self) -> None:
        renderer = ProjectionRenderer()
        with pytest.raises(AttributeError):
            renderer.style = FigureStyle()  # type: ignore[misc]

    @pytest.mark.parametrize("highlighted", [(), ("a", "b", "c", "d"), ("a", "a")])
    def test_invalid_highlighted_labels_are_refused(self, highlighted: tuple) -> None:
        with pytest.raises(ValueError):
            ProjectionRenderer().require_highlightable(highlighted)

    @pytest.mark.parametrize(
        ("panels", "shape"),
        [
            (1, (1, 1)),
            (2, (1, 2)),
            (4, (2, 2)),
            (5, (2, 3)),
            (12, (3, 4)),
            (20, (5, 4)),
        ],
    )
    def test_grid_is_as_square_as_the_column_cap_allows(
        self, panels: int, shape: tuple[int, int]
    ) -> None:
        assert ProjectionRenderer()._grid_shape(panels) == shape

    def test_legend_lists_the_fold_last(self) -> None:
        renderer = ProjectionRenderer()
        axes = Figure().subplots()
        points = np.zeros((1, 2))
        renderer._scatter(axes, points, "#c3c2b7", "Other (5)")
        renderer._scatter(axes, points, "#2a78d6", "mtg (9)")
        renderer._scatter(axes, points, "#eb6834", "gwent (7)")
        renderer._add_legend(axes, fold_drawn=True)
        texts = [text.get_text() for text in axes.get_legend().get_texts()]
        assert texts == ["mtg (9)", "gwent (7)", "Other (5)"]

    def test_a_real_label_named_other_is_not_mistaken_for_the_fold(self) -> None:
        renderer = ProjectionRenderer()
        axes = Figure().subplots()
        points = np.zeros((1, 2))
        renderer._scatter(axes, points, "#2a78d6", "Other (9)")  # a real label
        renderer._scatter(axes, points, "#eb6834", "gwent (7)")
        renderer._add_legend(axes, fold_drawn=False)
        texts = [text.get_text() for text in axes.get_legend().get_texts()]
        assert texts == ["Other (9)", "gwent (7)"]

    def test_both_figures_are_written_as_png(self, tmp_path: Path) -> None:
        renderer = ProjectionRenderer()
        coordinates = np.random.default_rng(0).standard_normal((30, 2))
        labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 5 + ["d"] * 5
        overview = renderer.draw_overview(
            coordinates, labels, ["a", "b", "absent"], tmp_path / "o.png"
        )
        panels = renderer.draw_label_panels(
            coordinates, labels, ["a", "b", "c", "d", "e"], tmp_path / "p.png"
        )
        for path in (overview, panels):
            assert path.read_bytes()[:8] == _PNG_MAGIC
