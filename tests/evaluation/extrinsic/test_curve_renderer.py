from pathlib import Path

import pytest
from matplotlib.figure import Figure

from src.evaluation.extrinsic.curve_renderer import CurveRenderer, CurveStyle
from src.training.recording.run_listener import RoundRow


def _rows(losses: list[float]) -> list[RoundRow]:
    return [
        RoundRow(
            "extrinsic",
            index,
            index * 10,
            float(index),
            "pick",
            loss,
            loss,
            None,
            False,
        )
        for index, loss in enumerate(losses)
    ]


def _would_label_ends(renderer: CurveRenderer, ends: dict, ylim: tuple) -> bool:
    axes = Figure().subplots()
    axes.set_ylim(*ylim)
    return renderer._should_label_ends(axes, ends)


class TestCurveStyle:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"line_width": 0.0},
            {"end_marker_size": -1.0},
            {"end_marker_ring": 0.0},
            {"figure_size": (0.0, 4.0)},
            {"max_direct_labels": -1},
            {"min_label_gap": 1.0},
            {"min_label_gap": -0.1},
        ],
    )
    def test_invalid_fields_are_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            CurveStyle(**overrides)

    def test_zero_direct_labels_is_allowed(self) -> None:
        assert CurveStyle(max_direct_labels=0).max_direct_labels == 0

    def test_max_encoders_follows_the_theme(self) -> None:
        assert CurveStyle().max_encoders == 8


class TestEndLabels:
    def test_few_well_separated_ends_are_labeled(self) -> None:
        ends = {"a": (5.0, 0.2), "b": (5.0, 0.6)}
        assert _would_label_ends(CurveRenderer(), ends, (0.0, 1.0))

    def test_crowded_ends_fall_back_to_the_legend(self) -> None:
        ends = {"a": (5.0, 0.50), "b": (5.0, 0.51)}
        assert not _would_label_ends(CurveRenderer(), ends, (0.0, 1.0))

    def test_too_many_encoders_fall_back_to_the_legend(self) -> None:
        ends = {name: (5.0, 0.1 * i) for i, name in enumerate("abcde")}
        assert not _would_label_ends(CurveRenderer(), ends, (0.0, 1.0))

    def test_zero_max_direct_labels_turns_them_off(self) -> None:
        ends = {"a": (5.0, 0.2), "b": (5.0, 0.8)}
        renderer = CurveRenderer(CurveStyle(max_direct_labels=0))
        assert not _would_label_ends(renderer, ends, (0.0, 1.0))


class TestDrawDojoCurves:
    def test_one_encoder_has_no_legend_and_is_named_in_the_title(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = CurveRenderer()
        seen: dict[str, object] = {}
        original_save = renderer._save

        def spy_save(figure: Figure, path: Path) -> Path:
            axes = figure.axes[0]
            seen["title"] = axes.get_title(loc="left")
            seen["legend"] = axes.get_legend()
            return original_save(figure, path)

        monkeypatch.setattr(renderer, "_save", spy_save)
        renderer.draw_dojo_curves(
            "pick",
            {"trained": _rows([1.0, 0.5])},
            {"trained": 0},
            "step",
            tmp_path / "p.png",
        )
        assert seen["title"] == "pick: trained"
        assert seen["legend"] is None

    def test_two_encoders_get_a_legend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = CurveRenderer()
        seen: dict[str, object] = {}
        original_save = renderer._save

        def spy_save(figure: Figure, path: Path) -> Path:
            axes = figure.axes[0]
            seen["title"] = axes.get_title(loc="left")
            legend = axes.get_legend()
            seen["legend"] = [text.get_text() for text in legend.get_texts()]
            seen["end_labels"] = sorted(text.get_text() for text in axes.texts)
            return original_save(figure, path)

        monkeypatch.setattr(renderer, "_save", spy_save)
        curves = {"a": _rows([1.0, 0.5]), "b": _rows([1.2, 0.9])}
        renderer.draw_dojo_curves(
            "pick", curves, {"a": 0, "b": 1}, "step", tmp_path / "p.png"
        )
        assert seen["title"] == "pick"
        assert seen["legend"] == ["a", "b"]
        # Two well-separated ends: each line is also labeled at its end
        assert seen["end_labels"] == ["a", "b"]

    def test_each_line_takes_its_slot_color_even_when_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Encoder "a" never scored this dojo: "b" is drawn alone but must
        # keep slot 1's color, as on every other chart
        renderer = CurveRenderer()
        seen: dict[str, object] = {}
        original_save = renderer._save

        def spy_save(figure: Figure, path: Path) -> Path:
            axes = figure.axes[0]
            seen["title"] = axes.get_title(loc="left")
            seen["colors"] = {
                line.get_label(): line.get_color()
                for line in axes.get_lines()
                if not line.get_label().startswith("_")  # skip end dots
            }
            return original_save(figure, path)

        monkeypatch.setattr(renderer, "_save", spy_save)
        curves = {"a": [], "b": _rows([1.0, 0.5])}
        renderer.draw_dojo_curves(
            "pick", curves, {"a": 0, "b": 1}, "step", tmp_path / "p.png"
        )
        assert seen["title"] == "pick: b"
        assert seen["colors"] == {"b": renderer.style.theme.categorical[1]}

    @pytest.mark.parametrize(
        ("slots", "match"),
        [
            ({"a": 0}, "slot None"),  # b has no slot
            ({"a": 0, "b": 8}, "slot 8"),  # beyond the eight colors
            ({"a": 0, "b": -1}, "slot -1"),  # negative
            ({"a": 1, "b": 1}, "share a slot"),
        ],
    )
    def test_bad_slots_are_refused(
        self, tmp_path: Path, slots: dict, match: str
    ) -> None:
        curves = {"a": _rows([1.0]), "b": _rows([1.0])}
        with pytest.raises(ValueError, match=match):
            CurveRenderer().draw_dojo_curves(
                "pick", curves, slots, "step", tmp_path / "p.png"
            )
        assert not (tmp_path / "p.png").exists()

    def test_no_rows_at_all_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no encoder"):
            CurveRenderer().draw_dojo_curves(
                "pick", {"a": []}, {"a": 0}, "step", tmp_path / "p.png"
            )


class TestYAxis:
    def _drawn_axes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        curves: dict[str, list[RoundRow]],
    ) -> dict[str, object]:
        renderer = CurveRenderer()
        seen: dict[str, object] = {}
        original_save = renderer._save

        def spy_save(figure: Figure, path: Path) -> Path:
            axes = figure.axes[0]
            seen["ylabel"] = axes.get_ylabel()
            seen["horizontal_at_one"] = any(
                list(line.get_ydata()) == [1.0, 1.0] for line in axes.get_lines()
            )
            seen["first_line_ys"] = list(axes.get_lines()[-2].get_ydata())
            return original_save(figure, path)

        monkeypatch.setattr(renderer, "_save", spy_save)
        slots = {encoder: index for index, encoder in enumerate(curves)}
        renderer.draw_dojo_curves("pick", curves, slots, "step", tmp_path / "p.png")
        return seen

    def test_normalized_rows_plot_normalized_loss_with_a_baseline_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            RoundRow("e", 0, 10, 0.0, "pick", 4.0, 0.8, None, False),
            RoundRow("e", 1, 20, 1.0, "pick", 2.0, 0.4, None, False),
        ]
        seen = self._drawn_axes(tmp_path, monkeypatch, {"a": rows})
        assert seen["ylabel"] == "TEST loss / baseline"
        assert seen["horizontal_at_one"]
        assert seen["first_line_ys"] == [0.8, 0.4]

    def test_a_legacy_row_falls_back_to_raw_loss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legacy = [RoundRow("e", 0, 10, 0.0, "pick", 4.0, None, None, False)]
        seen = self._drawn_axes(tmp_path, monkeypatch, {"a": legacy})
        assert seen["ylabel"] == "TEST loss"
        assert not seen["horizontal_at_one"]
        assert seen["first_line_ys"] == [4.0]
