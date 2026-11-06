"""ChartTheme: the colors and resolution every evaluation figure shares.

One source for the dataviz skill's reference palette (light mode: these
figures are files for reports). Chart-specific styles (FigureStyle for the
projection scatter, CurveStyle for learning curves) compose a ChartTheme
and add only what their chart form needs.
"""

from dataclasses import dataclass

# The reference palette's eight categorical slots, in their validated
# order (dataviz references/palette.md): blue, orange, aqua, yellow,
# magenta, green, violet, red. The order is the color-blind-safety
# mechanism for adjacent-pair forms (lines, bars): never reorder or cycle.
REFERENCE_CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)


@dataclass(frozen=True)
class ChartTheme:
    """The shared look of every evaluation figure.

    categorical: series colors in slot order (the full reference eight by
        default). A replacement must be re-validated with the dataviz
        validator for the chart forms that use it.
    surface: figure and plot background.
    text_primary, text_secondary: titles; legends and labels. Text never
        takes a series color.
    muted: axis tick labels.
    gridline: hairline gridlines and receded marks.
    axis: baselines / axis lines.
    dpi: PNG resolution.

    Exceptions: ValueError on construction if categorical is empty or dpi
        is below 1.
    """

    categorical: tuple[str, ...] = REFERENCE_CATEGORICAL
    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    muted: str = "#898781"
    gridline: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    dpi: int = 150

    def __post_init__(self) -> None:
        """Validate the fields. Inputs: none. Output: None. Side effects:
        none. Exceptions: ValueError if categorical is empty or dpi < 1."""
        if not self.categorical:
            raise ValueError("a ChartTheme needs at least one categorical color")
        if self.dpi < 1:
            raise ValueError(f"dpi must be >= 1, got {self.dpi}")
