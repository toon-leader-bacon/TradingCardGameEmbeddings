"""ChartCanvas: the rendering scaffolding every evaluation figure shares on
top of matplotlib.

ProjectionRenderer (scatter) and CurveRenderer (lines) each build on this:
a blank Figure sized and colored from the chart's theme, a legend fixed
just outside the plot's right edge, and the PNG save. Everything that
differs by chart form - what counts as a "blank" axes (ticks, gridlines,
spines), how a legend's handles are chosen, what gets drawn - stays in the
subclass; this holds only what both renderers already did identically.

Figures are built on matplotlib.figure.Figure directly, never pyplot, so
no global state or display backend is involved.
"""

from pathlib import Path
from typing import Any, Generic, Protocol, Sequence, TypeVar

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.evaluation.chart_theme import ChartTheme


class HasTheme(Protocol):
    """What a ChartCanvas needs from a chart's style: a ChartTheme to draw
    on. FigureStyle and CurveStyle both satisfy this by composing a
    ChartTheme, as the module docstrings of their renderers describe."""

    @property
    def theme(self) -> ChartTheme: ...  # noqa: E704 (Protocol stub)


_StyleT = TypeVar("_StyleT", bound=HasTheme)


class ChartCanvas(Generic[_StyleT]):
    """Shared matplotlib scaffolding for one evaluation renderer.

    Stateless apart from its read-only style, so one instance (e.g. a
    default argument) can be shared safely. A renderer subclasses
    `ChartCanvas[ItsStyle]`, keeping its own drawing semantics (scatter,
    lines, ...) and blank-axes setup, and calls `_new_figure`,
    `_outside_legend` and `_save` for the parts that are identical across
    every chart form.

    Inputs (constructor): style (anything exposing a .theme ChartTheme -
        a FigureStyle, a CurveStyle, or another chart's style).
    """

    def __init__(self, style: _StyleT) -> None:
        """Inputs: style. Output: none (constructor). Side effects: none.
        Exceptions: none."""
        self._style = style

    @property
    def style(self) -> _StyleT:
        """Read-only: the style fixed at construction. Inputs: none.
        Output: the style passed to the constructor. Side effects: none.
        Exceptions: none."""
        return self._style

    def _new_figure(self, figsize: tuple[float, float]) -> Figure:
        """Inputs: figsize (width, height) in inches. Output: a blank
        Figure on the style's theme surface, with constrained layout (so
        an outside legend gets room without manual tuning) - no Axes yet,
        since how many and how they look is form-specific. Side effects:
        none. Exceptions: none."""
        return Figure(
            figsize=figsize,
            facecolor=self._style.theme.surface,
            layout="constrained",
        )

    def _outside_legend(
        self,
        axes: Axes,
        handles: Sequence[Artist] | None = None,
        texts: Sequence[str] | None = None,
        **extra: Any,
    ) -> None:
        """Inputs: axes (with labeled series, when handles/texts are not
        given - then axes.get_legend_handles_labels() supplies them);
        handles, texts (an explicit, possibly reordered, pair - for a
        caller such as ProjectionRenderer that moves its "Other" fold to
        the end of the list); extra (further matplotlib Legend keyword
        arguments, e.g. markerscale - matplotlib validates these, not this
        method). Output: None. Side effects: adds a frameless legend just
        outside the plot's right edge (loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False),
        text in style.theme.text_secondary. Exceptions: none.

        Example:
            >>> canvas._outside_legend(axes, markerscale=3.0)
        """
        if handles is None or texts is None:
            handles, texts = axes.get_legend_handles_labels()
        axes.legend(
            handles,
            texts,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            labelcolor=self._style.theme.text_secondary,
            **extra,
        )

    def _save(self, figure: Figure, path: Path) -> Path:
        """Inputs: a finished figure, its path. Output: path. Side
        effects: writes the PNG at the theme's dpi on its surface.
        Exceptions: OSError.

        Example:
            >>> canvas._save(figure, out_dir / "chart.png")
        """
        theme = self._style.theme
        figure.savefig(path, dpi=theme.dpi, facecolor=theme.surface)
        return path
