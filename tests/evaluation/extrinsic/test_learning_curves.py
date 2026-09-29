from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from src.evaluation.chart_theme import ChartTheme
from src.evaluation.extrinsic.curve_renderer import CurveRenderer, CurveStyle, XAxis
from src.evaluation.extrinsic.learning_curves import plot_learning_curves
from src.training.recording.reports import RoundReport
from src.training.recording.run_listener import CsvRunListener, RoundRow

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _RecordingRenderer(CurveRenderer):
    """Draws as usual, remembering each chart's inputs."""

    def __init__(self, style: CurveStyle = CurveStyle()) -> None:
        super().__init__(style)
        self.charts: dict[str, tuple[dict[str, int], dict[str, int], XAxis]] = {}

    def draw_dojo_curves(
        self,
        dojo: str,
        curves: Mapping[str, Sequence[RoundRow]],
        encoder_slots: Mapping[str, int],
        x_axis: XAxis,
        path: Path,
    ) -> Path:
        row_counts = {encoder: len(rows) for encoder, rows in curves.items()}
        self.charts[dojo] = (row_counts, dict(encoder_slots), x_axis)
        return super().draw_dojo_curves(dojo, curves, encoder_slots, x_axis, path)


def _rounds_csv(path: Path, losses_per_round: list[dict[str, float]]) -> Path:
    listener = CsvRunListener(path)
    for index, losses in enumerate(losses_per_round):
        report = RoundReport(
            "extrinsic", index, (index + 1) * 10, float(index), losses, {}, frozenset()
        )
        listener.on_round_end(report)
    return path


def test_one_chart_per_dojo_with_fixed_colors(tmp_path: Path) -> None:
    trained = _rounds_csv(
        tmp_path / "t.csv", [{"pick": 1.0, "winner": 2.0}, {"pick": 0.5, "winner": 1.5}]
    )
    untrained = _rounds_csv(tmp_path / "u.csv", [{"pick": 1.2}] * 3)
    renderer = _RecordingRenderer()

    paths = plot_learning_curves(
        {"trained": trained, "untrained": untrained},
        tmp_path / "plots",
        renderer=renderer,
    )

    assert paths == (tmp_path / "plots" / "pick.png", tmp_path / "plots" / "winner.png")
    assert all(path.read_bytes()[:8] == _PNG_MAGIC for path in paths)
    pick_rows, pick_slots, _ = renderer.charts["pick"]
    winner_rows, winner_slots, _ = renderer.charts["winner"]
    assert pick_rows == {"trained": 2, "untrained": 3}
    # An encoder that never scored a dojo is present with no rows...
    assert winner_rows == {"trained": 2, "untrained": 0}
    # ...and keeps its slot, so colors follow the encoder on every chart
    assert pick_slots == winner_slots == {"trained": 0, "untrained": 1}


def test_the_x_axis_is_passed_through(tmp_path: Path) -> None:
    csv = _rounds_csv(tmp_path / "t.csv", [{"pick": 1.0}, {"pick": 0.5}])
    renderer = _RecordingRenderer()
    plot_learning_curves(
        {"a": csv}, tmp_path / "plots", "elapsed_seconds", renderer=renderer
    )
    assert renderer.charts["pick"][2] == "elapsed_seconds"


def test_existing_plots_are_overwritten(tmp_path: Path) -> None:
    csv = _rounds_csv(tmp_path / "t.csv", [{"pick": 1.0}])
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "pick.png").write_bytes(b"old")
    plot_learning_curves({"a": csv}, tmp_path / "plots")
    assert (tmp_path / "plots" / "pick.png").read_bytes()[:8] == _PNG_MAGIC


@pytest.mark.parametrize(("first", "second"), [("a b", "a/b"), ("Pick", "pick")])
def test_colliding_file_names_write_nothing(
    tmp_path: Path, first: str, second: str
) -> None:
    csv = _rounds_csv(tmp_path / "t.csv", [{first: 1.0, second: 2.0}])
    with pytest.raises(ValueError, match="share the plot file"):
        plot_learning_curves({"a": csv}, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


def test_too_many_or_no_encoders_are_refused(tmp_path: Path) -> None:
    csv = _rounds_csv(tmp_path / "t.csv", [{"pick": 1.0}])
    two_colors = CurveRenderer(
        CurveStyle(theme=ChartTheme(categorical=("#000", "#fff")))
    )
    with pytest.raises(ValueError, match="1 to 2"):
        plot_learning_curves(
            {"a": csv, "b": csv, "c": csv}, tmp_path / "plots", renderer=two_colors
        )
    with pytest.raises(ValueError):
        plot_learning_curves({}, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


def test_empty_csvs_skip_their_encoder_and_all_empty_is_refused(tmp_path: Path) -> None:
    full = _rounds_csv(tmp_path / "full.csv", [{"pick": 1.0}])
    empty = tmp_path / "empty.csv"
    # The header only, built from RoundRow: the format's one definition
    empty.write_text(",".join(field.name for field in fields(RoundRow)) + "\n")
    renderer = _RecordingRenderer()
    plot_learning_curves({"e": empty, "f": full}, tmp_path / "plots", renderer=renderer)
    assert renderer.charts["pick"][0] == {"e": 0, "f": 1}
    with pytest.raises(ValueError, match="no rows"):
        plot_learning_curves({"e": empty}, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_a_missing_csv_fails_before_writing(tmp_path: Path) -> None:
    csv = _rounds_csv(tmp_path / "t.csv", [{"pick": 1.0}])
    with pytest.raises(FileNotFoundError):
        plot_learning_curves(
            {"a": csv, "b": tmp_path / "absent.csv"}, tmp_path / "plots"
        )
    assert not (tmp_path / "plots").exists()
