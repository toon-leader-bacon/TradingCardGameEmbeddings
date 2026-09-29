"""ProjectionRenderer: draws ProjectionPlot's two static PNG figures in
matplotlib, in a FigureStyle.

Follows the dataviz skill's method: a scatter puts every pair of colors
side by side, and only a few categorical colors stay distinguishable for
color-blind readers under that condition - so the overview colors at most
len(style.series) labels and folds the rest into one neutral "Other", and
the panel figure (small multiples) covers any number of labels with a
single highlight color. Text always uses text colors, never a series color.

Figures are built on matplotlib.figure.Figure directly, never pyplot, so
no global state or display backend is involved.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_OTHER = "Other"
_PANEL_INCHES = 3.0  # width and height of one small-multiples cell


@dataclass(frozen=True)
class FigureStyle:
    """How a projection figure looks. The defaults are the dataviz
    reference palette, light mode (these are files for reports).

    series: the highlight colors, in slot order; its length is how many
        labels the overview may color. The default three (blue, orange,
        aqua) pass the skill's validator with every pair side by side, as
        a scatter needs; a replacement must be re-validated
        (validate_palette.js --pairs all) and should not exceed three -
        documented, deliberately not enforced, since validity depends on the
        colors, not just their count. Aqua
        is below 3:1 contrast on the surface: the legend and the
        projection.csv table view are its required relief.
    other: the "Other" fold - a neutral, never a hue. The default is the
        palette's baseline gray, validated against the default series (the
        darker muted gray #898781 fails against aqua).
    receded: every other card in a panel.
    surface, text_primary, text_secondary: background and text colors.
    marker_area: scatter marker area in points^2 (small: dense scatter).
    dpi: PNG resolution.
    max_panel_columns: the widest the small-multiples grid gets.

    Exceptions: ValueError on construction if series is empty, or
        marker_area, dpi or max_panel_columns is not positive.
    """

    series: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
    other: str = "#c3c2b7"
    receded: str = "#e1e0d9"
    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    marker_area: float = 6.0
    dpi: int = 150
    max_panel_columns: int = 4

    def __post_init__(self) -> None:
        """Validate the fields. Inputs: none. Output: None. Side effects:
        none. Exceptions: ValueError if series is empty,
        or marker_area, dpi or max_panel_columns is not positive."""
        if not self.series:
            raise ValueError("a FigureStyle needs at least one series color")
        if self.marker_area <= 0 or self.dpi < 1 or self.max_panel_columns < 1:
            raise ValueError(
                "marker_area, dpi and max_panel_columns must be positive, got "
                f"{self.marker_area}, {self.dpi}, {self.max_panel_columns}"
            )

    @property
    def max_highlighted(self) -> int:
        """The most labels the overview colors: one per series color.
        Inputs: none. Output: int. Side effects: none. Exceptions: none."""
        return len(self.series)


class ProjectionRenderer:
    """Draws a projection's overview and per-label panels in one style.

    Stateless apart from its read-only style, so one instance (e.g. a
    default argument) can be shared safely. The overview's fold is labeled
    "Other"; a real label of that name would share the legend text (the
    fold is still told apart from it by position, not by name).

    Inputs (constructor): style (FigureStyle; the reference palette by
        default).
    """

    def __init__(self, style: FigureStyle = FigureStyle()) -> None:
        """Inputs: style. Output: none (constructor). Side effects: none.
        Exceptions: none."""
        self._style = style

    @property
    def style(self) -> FigureStyle:
        """Read-only: the style fixed at construction. Inputs: none.
        Output: FigureStyle. Side effects: none. Exceptions: none."""
        return self._style

    def draw_overview(
        self,
        coordinates: np.ndarray,
        labels: Sequence[str],
        highlighted: Sequence[str],
        path: Path,
    ) -> Path:
        """Every card, with highlighted labels colored and the rest as "Other".

        Inputs: coordinates (n, 2); labels (one per card); highlighted (1 to
            style.max_highlighted labels, most important first - slot i
            gets style.series[i], so the same list colors the same label
            identically on every plot; a label with no cards still gets its
            legend entry, at a count of 0); path.
        Output: path.
        Side effects: writes a PNG to path.
        Exceptions: ValueError as require_highlightable; OSError on a
            write failure.

        Example:
            >>> renderer.draw_overview(xy, labels, ["mtg", "gwent"], out / "p.png")
        """
        self.require_highlightable(highlighted)
        figure, axes_grid = self._blank_figure(1, 1)
        axes = axes_grid[0, 0]
        label_array = np.asarray(labels)
        others = ~np.isin(label_array, list(highlighted))

        # "Other" first, beneath every highlighted label
        if others.any():
            count = int(others.sum())
            self._scatter(
                axes, coordinates[others], self.style.other, f"{_OTHER} ({count})"
            )
        for slot, label in enumerate(highlighted):
            members = label_array == label
            legend = f"{label} ({int(members.sum())})"
            self._scatter(axes, coordinates[members], self.style.series[slot], legend)

        # A legend is required for two or more series; text in text colors
        self._add_legend(axes, fold_drawn=bool(others.any()))
        return self._save(figure, path)

    def draw_label_panels(
        self,
        coordinates: np.ndarray,
        labels: Sequence[str],
        panel_labels: Sequence[str],
        path: Path,
    ) -> Path:
        """Small multiples: one panel per label, that label highlighted in
        the first series color over every card in the receded gray, shared
        axes.

        Inputs: coordinates (n, 2); labels (one per card); panel_labels (the
            labels to give a panel, in panel order); path.
        Output: path.
        Side effects: writes a PNG to path.
        Exceptions: ValueError (from _grid_shape) if panel_labels is empty -
            unreachable from ProjectionPlot, whose samples always have at
            least two labels; OSError on a write failure.

        Example:
            >>> renderer.draw_label_panels(xy, labels, ["common", "rare"], out / "p.png")
        """
        rows, columns = self._grid_shape(len(panel_labels))
        figure, axes_grid = self._blank_figure(rows, columns)
        label_array = np.asarray(labels)

        # One panel per label: all cards receded, this label on top
        for axes, label in zip(axes_grid.flat, panel_labels):
            members = label_array == label
            self._scatter(axes, coordinates, self.style.receded, None)
            self._scatter(axes, coordinates[members], self.style.series[0], None)
            axes.set_title(
                f"{label} ({int(members.sum())})", color=self.style.text_primary
            )

        # Unused grid cells stay blank
        for axes in list(axes_grid.flat)[len(panel_labels) :]:
            axes.set_visible(False)
        return self._save(figure, path)

    def require_highlightable(self, highlighted: Sequence[str]) -> None:
        """The one rule for a list of labels to color: public so callers
        (ProjectionPlot) can refuse a bad list before writing anything.

        Inputs: the labels to color. Output: None. Side effects: none.
        Exceptions: ValueError if empty, longer than style.max_highlighted,
            or repeating a label.

        Example:
            >>> ProjectionRenderer().require_highlightable(("mtg", "gwent"))
        """
        limit = self._style.max_highlighted
        if not 1 <= len(highlighted) <= limit:
            raise ValueError(
                f"can color 1 to {limit} labels, got {len(highlighted)}: "
                f"{list(highlighted)}"
            )
        if len(set(highlighted)) != len(highlighted):
            raise ValueError(f"highlighted labels repeat: {list(highlighted)}")

    def _blank_figure(self, rows: int, columns: int) -> tuple[Figure, np.ndarray]:
        """Inputs: grid shape. Output: a Figure on the style's surface and
        its Axes as a 2-D (rows, columns) array - always 2-D, a 1x1 grid
        included (squeeze=False). Axes are recessive: no ticks (t-SNE axes
        carry no units), hairline spines, shared x/y limits across panels.
        Side effects: none. Exceptions: none."""
        style = self._style
        # A single plot gets room for its outside legend; panels grow per cell
        size = (max(6.5, _PANEL_INCHES * columns), max(5.0, _PANEL_INCHES * rows))
        figure = Figure(figsize=size, facecolor=style.surface, layout="constrained")
        axes_grid = figure.subplots(
            rows, columns, squeeze=False, sharex=True, sharey=True
        )
        # Recessive axes: t-SNE coordinates carry no units, so no ticks
        for axes in axes_grid.flat:
            axes.set_facecolor(style.surface)
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_color(style.receded)
                spine.set_linewidth(0.5)
        return figure, axes_grid

    def _scatter(
        self, axes: Axes, points: np.ndarray, color: str, legend_label: str | None
    ) -> None:
        """Inputs: axes, points (m, 2), the mark color, a legend entry (None
        for none). Output: None. Side effects: draws the points (area
        style.marker_area, no edge). Exceptions: none."""
        axes.scatter(
            points[:, 0],
            points[:, 1],
            s=self._style.marker_area,
            c=color,
            linewidths=0,
            label=legend_label,
        )

    def _add_legend(self, axes: Axes, fold_drawn: bool) -> None:
        """Inputs: axes with labeled series; fold_drawn (whether the first
        series drawn is the "Other" fold). Output: None. Side effects: adds
        a frameless legend outside the plot area, text in
        style.text_secondary, markers enlarged so the swatch reads. The
        fold is drawn first (beneath) but listed last, after the
        highlighted labels in slot order - found by its draw position,
        never by its text. Exceptions: none."""
        handles, texts = axes.get_legend_handles_labels()
        # The fold, if drawn, is the first series: move it to the end
        if fold_drawn:
            handles = handles[1:] + handles[:1]
            texts = texts[1:] + texts[:1]
        axes.legend(
            handles,
            texts,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            markerscale=3.0,
            labelcolor=self._style.text_secondary,
        )

    def _grid_shape(self, panel_count: int) -> tuple[int, int]:
        """Inputs: panel_count (>= 1). Output: (rows, columns), at most
        style.max_panel_columns columns, as square as that allows. Side
        effects: none. Exceptions: ValueError if panel_count < 1."""
        if panel_count < 1:
            raise ValueError(f"need at least one panel, got {panel_count}")
        columns = min(self._style.max_panel_columns, math.ceil(math.sqrt(panel_count)))
        return math.ceil(panel_count / columns), columns

    def _save(self, figure: Figure, path: Path) -> Path:
        """Inputs: a finished figure, its path. Output: path. Side effects:
        writes the PNG at style.dpi on the style's surface. Exceptions:
        OSError."""
        figure.savefig(path, dpi=self._style.dpi, facecolor=self._style.surface)
        return path
