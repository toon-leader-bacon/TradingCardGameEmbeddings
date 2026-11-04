# evaluation/charts

The figure scaffolding both `intrinsic/` and `extrinsic/` draw on. Imports
neither.

## Files

- [chart_theme.py](chart_theme.py): `ChartTheme`, the colors and
  resolution every figure shares (the dataviz reference palette, light
  mode: eight categorical colors in their validated order, surface, text,
  gridline and axis grays). Each chart's style composes it: `FigureStyle`
  (projection scatter) and `CurveStyle` (learning curves).
- [chart_canvas.py](chart_canvas.py): `ChartCanvas`, the matplotlib
  scaffolding every renderer shares on top of a `ChartTheme` - a blank
  Figure sized and colored from the style, an outside legend in the
  project's fixed position, and the PNG save. `ProjectionRenderer` and
  `CurveRenderer` each subclass it, keeping their own drawing semantics
  (scatter, lines) and blank-axes setup, which differ enough (ticks,
  gridlines, spines) to stay apart.
