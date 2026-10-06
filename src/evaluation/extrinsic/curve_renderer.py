"""CurveRenderer: draws one learning-curve chart - every encoder's TEST
loss on one dojo - in a CurveStyle.

The y axis is normalized TEST loss (loss / the dojo's baseline), with a
hairline reference at 1.0 ("learned nothing"), whenever every plotted row
has it; a chart with any row from a rounds CSV written before that column
existed falls back to raw TEST loss.

Follows the dataviz skill's line specs: 2px lines, >= 8px end dots with a
2px surface-colored ring, hairline solid gridlines, recessive axes, a
legend for two or more encoders (one encoder: the title names it), text in
text colors. Three of the eight colors (aqua, yellow, magenta) sit below 3:1
contrast on the surface; their relief is the legend, the direct end labels
and the rounds CSVs themselves (the table view). Encoder colors
come from the theme's categorical slots in the fixed validated order -
the eight reference colors are validated for adjacent-pair forms such as
lines - and follow the encoder, never its rank, so an encoder has the same
color on every dojo's chart.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.evaluation.chart_canvas import ChartCanvas
from src.evaluation.chart_theme import ChartTheme
from src.training.recording.run_listener import RoundRow

XAxis = Literal["step", "elapsed_seconds"]

# How each x axis reads a row: a typed lookup, not getattr on a string
_X_VALUES: dict[XAxis, Callable[[RoundRow], float]] = {
    "step": lambda row: float(row.step),
    "elapsed_seconds": lambda row: row.elapsed_seconds,
}

_NORMALIZED_Y_TITLE = "TEST loss / baseline"
_RAW_Y_TITLE = "TEST loss"

_X_AXIS_TITLES: dict[XAxis, str] = {
    "step": "Optimizer step",
    "elapsed_seconds": "Elapsed time (s)",
}


@dataclass(frozen=True)
class CurveStyle:
    """How a learning-curve chart looks: the shared ChartTheme plus what a
    line chart needs. Sizes are in points (1 pt = dpi / 72 px; at the
    default 150 dpi, 1 pt is about 2 px).

    theme: colors and resolution; its categorical length is how many
        encoders one chart can show.
    line_width: 1.0 pt (2 px).
    end_marker_size: end-dot diameter, 4.0 pt (8 px).
    end_marker_ring: the surface-colored ring around it, 1.0 pt (2 px).
    max_direct_labels: with at most this many encoders, line ends are
        also labeled with the encoder name (the legend stays); 0 turns
        end labels off.
    min_label_gap: direct labels are dropped (legend only) if two line
        ends sit closer than this fraction of the y range - stacking
        nudged labels would detach them from their lines.
    figure_size: inches (width, height).

    Exceptions: ValueError on construction unless line_width,
        end_marker_size, end_marker_ring and both figure_size values are
        > 0, max_direct_labels >= 0, and min_label_gap is in [0, 1).
    """

    theme: ChartTheme = field(default_factory=ChartTheme)
    line_width: float = 1.0
    end_marker_size: float = 4.0
    end_marker_ring: float = 1.0
    max_direct_labels: int = 4
    min_label_gap: float = 0.04
    figure_size: tuple[float, float] = (7.5, 4.5)

    def __post_init__(self) -> None:
        """Inputs: none. Output: None. Side effects: none. Exceptions:
        ValueError on an out-of-range field (see class docstring)."""
        sizes = (self.line_width, self.end_marker_size, self.end_marker_ring)
        if not all(size > 0 for size in sizes + self.figure_size):
            raise ValueError(
                "line_width, end_marker_size, end_marker_ring and figure_size "
                f"must be > 0, got {sizes}, {self.figure_size}"
            )
        if self.max_direct_labels < 0:
            raise ValueError(
                f"max_direct_labels must be >= 0, got {self.max_direct_labels}"
            )
        if not 0 <= self.min_label_gap < 1:
            raise ValueError(
                f"min_label_gap must be in [0, 1), got {self.min_label_gap}"
            )

    @property
    def max_encoders(self) -> int:
        """The most encoders one chart can color: one per categorical slot.
        Inputs: none. Output: int. Side effects: none. Exceptions: none."""
        return len(self.theme.categorical)


class CurveRenderer(ChartCanvas[CurveStyle]):
    """Draws learning-curve charts in one read-only style. Stateless apart
    from that style, so one instance can be shared (e.g. as a default
    argument).

    Line-specific: everything here. The blank-figure/legend/save
    scaffolding this shares with ProjectionRenderer is ChartCanvas.

    Inputs (constructor): style (CurveStyle; the reference palette by
        default).
    """

    def __init__(self, style: CurveStyle = CurveStyle()) -> None:
        """Inputs: style. Output: none (constructor). Side effects: none.
        Exceptions: none."""
        super().__init__(style)

    def draw_dojo_curves(
        self,
        dojo: str,
        curves: Mapping[str, Sequence[RoundRow]],
        encoder_slots: Mapping[str, int],
        x_axis: XAxis,
        path: Path,
    ) -> Path:
        """One chart: each encoder's TEST loss on this dojo against x_axis
        (normalized, with a 1.0 reference line, unless a row lacks it - see
        the module docstring).

        Inputs: dojo (the chart title); curves (encoder label -> that
            encoder's rows for this dojo, in file order; an encoder with no
            rows here is left out of this chart); encoder_slots (encoder
            label -> categorical slot, fixed across every dojo's chart so
            colors follow the encoder); x_axis ("step" or
            "elapsed_seconds"); path.
        Output: path.
        Side effects: writes a PNG to path.
        Exceptions: ValueError if an encoder in curves has no slot, a slot
            is outside the theme's categorical range, two encoders share a
            slot (they would share a color), or no encoder has any rows;
            OSError on a write failure.

        Example:
            >>> renderer.draw_dojo_curves("pick", {"trained": rows_a,
            ...     "untrained": rows_b}, {"trained": 0, "untrained": 1},
            ...     "step", out / "pick.png")
        """
        self._require_slots(curves, encoder_slots)
        drawn = {encoder: rows for encoder, rows in curves.items() if rows}
        normalized = _all_rows_normalized(drawn)
        figure, axes = self._blank_chart(
            _chart_title(dojo, list(drawn)),
            x_axis,
            _NORMALIZED_Y_TITLE if normalized else _RAW_Y_TITLE,
        )
        if normalized:
            self._draw_baseline_reference(axes)

        # One line per encoder, in its fixed color, ending in a ringed dot
        ends: dict[str, tuple[float, float]] = {}
        for encoder, rows in drawn.items():
            color = self._style.theme.categorical[encoder_slots[encoder]]
            ys = [_y_value(row, normalized) for row in rows]
            ends[encoder] = self._draw_curve(axes, rows, ys, x_axis, color, encoder)

        # Two or more encoders: a legend always, plus direct end labels when
        # few and well apart. One encoder: the title names it, no legend.
        if len(ends) >= 2:
            self._add_legend(axes)
            if self._should_label_ends(axes, ends):
                self._label_ends(axes, ends)
        return self._save(figure, path)

    def _require_slots(
        self, curves: Mapping[str, Sequence[RoundRow]], encoder_slots: Mapping[str, int]
    ) -> None:
        """Inputs: curves, encoder_slots. Output: None. Side effects: none.
        Exceptions: ValueError naming the encoder if it has no slot or its
        slot is not in range(style.max_encoders), or naming the encoders
        that share a slot."""
        for encoder in curves:
            slot = encoder_slots.get(encoder)
            if slot is None or not 0 <= slot < self._style.max_encoders:
                raise ValueError(
                    f"encoder {encoder!r} has slot {slot}; need 0 to "
                    f"{self._style.max_encoders - 1}"
                )
        by_slot: dict[int, list[str]] = {}
        for encoder in curves:
            by_slot.setdefault(encoder_slots[encoder], []).append(encoder)
        shared = [encoders for encoders in by_slot.values() if len(encoders) > 1]
        if shared:
            raise ValueError(f"encoders share a slot (and a color): {shared}")
        if not any(curves.values()):
            raise ValueError("no encoder has any rows to plot")

    def _blank_chart(
        self, title: str, x_axis: XAxis, y_title: str
    ) -> tuple[Figure, Axes]:
        """Inputs: the chart title, x_axis (its axis title), y_title. Output:
        a Figure on the theme's surface and its one Axes: title in
        text_primary, axis titles in text_secondary, tick labels in
        muted, hairline solid y gridlines in gridline, axis spines in axis
        color, top/right spines hidden. Side effects: none. Exceptions:
        none."""
        theme = self._style.theme
        figure = self._new_figure(self._style.figure_size)
        axes = figure.subplots()
        axes.set_facecolor(theme.surface)
        axes.set_title(title, color=theme.text_primary, loc="left")
        axes.set_xlabel(_X_AXIS_TITLES[x_axis], color=theme.text_secondary)
        axes.set_ylabel(y_title, color=theme.text_secondary)
        axes.tick_params(colors=theme.muted, labelcolor=theme.muted)
        # Recessive frame: hairline solid y gridlines, no top/right spines
        axes.grid(axis="y", color=theme.gridline, linewidth=0.5, linestyle="-")
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(theme.axis)
            axes.spines[side].set_linewidth(0.5)
        return figure, axes

    def _draw_baseline_reference(self, axes: Axes) -> None:
        """Inputs: axes. Output: None. Side effects: draws a hairline dashed
        horizontal line at y = 1.0 (normalized loss of a dojo that learned
        nothing) in the axis color, behind the curves and out of the
        legend. Exceptions: none."""
        axes.axhline(
            1.0,
            color=self._style.theme.axis,
            linewidth=0.75,
            linestyle="--",
            zorder=1,
        )

    def _draw_curve(
        self,
        axes: Axes,
        rows: Sequence[RoundRow],
        ys: Sequence[float],
        x_axis: XAxis,
        color: str,
        encoder: str,
    ) -> tuple[float, float]:
        """Inputs: axes, one encoder's rows (non-empty), their y values (same
        order), x_axis, its color, its label (legend entry). Output: the
        line's last (x, y).
        Side effects: draws the line (style.line_width, round joins/caps)
        and a ringed end dot (style.end_marker_size, surface-colored
        style.end_marker_ring). Exceptions: none."""
        xs = [_X_VALUES[x_axis](row) for row in rows]
        style = self._style
        axes.plot(
            xs,
            ys,
            color=color,
            linewidth=style.line_width,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=encoder,
        )
        # The end dot, ringed in the surface color so crossings stay legible
        axes.plot(
            xs[-1:],
            ys[-1:],
            linestyle="none",
            marker="o",
            markersize=style.end_marker_size,
            markerfacecolor=color,
            markeredgecolor=style.theme.surface,
            markeredgewidth=style.end_marker_ring,
        )
        return float(xs[-1]), float(ys[-1])

    def _should_label_ends(
        self, axes: Axes, ends: Mapping[str, tuple[float, float]]
    ) -> bool:
        """Inputs: axes (for its y range), encoder -> line end (two or
        more). Output: True if there are at most style.max_direct_labels
        ends and every pair of end y values is at least
        style.min_label_gap of the axes' y range apart. Side effects:
        none. Exceptions: none."""
        if len(ends) > self._style.max_direct_labels:
            return False
        low, high = axes.get_ylim()
        min_gap = self._style.min_label_gap * (high - low)
        end_ys = sorted(y for _, y in ends.values())
        # Sorted, so only neighbours can be the closest pair
        return all(upper - lower >= min_gap for lower, upper in zip(end_ys, end_ys[1:]))

    def _label_ends(self, axes: Axes, ends: Mapping[str, tuple[float, float]]) -> None:
        """Inputs: axes, encoder -> line end. Output: None. Side effects:
        writes each encoder's name just right of its end dot, in
        text_secondary (never the series color). Exceptions: none."""
        for encoder, (x, y) in ends.items():
            axes.annotate(
                encoder,
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=self._style.theme.text_secondary,
                annotation_clip=False,
            )

    def _add_legend(self, axes: Axes) -> None:
        """Inputs: axes with labeled lines. Output: None. Side effects: adds
        a frameless legend outside the plot area, text in text_secondary.
        Exceptions: none."""
        self._outside_legend(axes)


def _chart_title(dojo: str, encoders: list[str]) -> str:
    """Inputs: the dojo, the encoders drawn (non-empty). Output: the dojo
    name, or "<dojo>: <encoder>" when one encoder is drawn - a single
    series has no legend, so the title names it. Side effects: none.
    Exceptions: ValueError if encoders is empty."""
    if not encoders:
        raise ValueError("a chart needs at least one encoder")
    return dojo if len(encoders) > 1 else f"{dojo}: {encoders[0]}"


def _all_rows_normalized(curves: Mapping[str, Sequence[RoundRow]]) -> bool:
    """Whether every row of every curve has a normalized_test_loss.

    Inputs: encoder -> rows. Output: bool (False if any row comes from a
    rounds CSV written before that column). Side effects: none.
    Exceptions: none.
    """
    return all(
        row.normalized_test_loss is not None for rows in curves.values() for row in rows
    )


def _y_value(row: RoundRow, normalized: bool) -> float:
    """row's y value: its normalized TEST loss if normalized, else its raw
    TEST loss. Inputs: row, normalized (only True when every row has one,
    see _all_rows_normalized). Output: float. Side effects: none.
    Exceptions: none."""
    if normalized and row.normalized_test_loss is not None:
        return row.normalized_test_loss
    return row.test_loss
