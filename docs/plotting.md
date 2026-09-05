# Plotting methods

KiwiCalc provides two compatible plotting styles. Standalone helpers are useful
for a quick figure, while `Graph2D` and `Graph3D` retain items, decorations,
themes, and axis configuration for composition and export. All plotting uses
Matplotlib and accepts `show=False` for notebooks, tests, and larger layouts.

## Standalone methods

The function, curve, surface, scatter, and vector helpers accept `fig=` and
`ax=`. Either may be supplied alone; when both are supplied they must refer to
the same figure. Methods which draw one object return its primary Matplotlib
artist. Multi-object helpers return a list of artists. `plot_multiple()` returns
the `(figure, axes)` pair, with `axes` consistently represented as a two-
dimensional NumPy array.

`plot_multiple()` leaves ordinary Python callables and lambdas untitled by
default instead of exposing their implementation representation. Formula
strings and KiwiCalc expressions use their readable formula text. Pass
`subplot_titles=(...)` to provide one explicit title per function.

```python
import matplotlib.pyplot as plt
import kiwicalc as kw

fig, ax = plt.subplots()
line = kw.plot_function("x^2 - 1", fig=fig, ax=ax, show=False)
points = kw.scatter_dots([-1, 0, 1], [0, -1, 0], fig=fig, ax=ax, show=False)
```

Supplying mismatched figures and axes raises `ValueError`. Three-dimensional
methods also reject non-positive steps, mismatched mesh shapes, and coordinate
arrays of unequal length.

## Expanded standalone API

The method-oriented API is organized in three layers. These functions use the
same `fig=`, `ax=`, and `show=` lifecycle as the original plotting helpers and
return their primary Matplotlib artist, or a small result object when several
artists are equally important.

### Phase 1: scientific fields

`plot_vector_field()`, `plot_slope_field()`, `plot_gradient_field()`,
`plot_streamlines()`, and `plot_contour()` are standalone counterparts to the
corresponding `Graph2D` methods. Formula strings, expressions, constants, and
ordinary callables are supported.

```python
fig, ax = plt.subplots()
field = kw.plot_vector_field(
    lambda x, y: -y,
    lambda x, y: x,
    x_range=(-3, 3), y_range=(-3, 3), density=15,
    color="magnitude", colorbar=True,
    fig=fig, ax=ax, show=False,
)
print(field.kiwicalc_colorbar)
```

Field artists expose `kiwicalc_artists` for related components, such as the
line and arrow collections of a stream plot, and `kiwicalc_colorbar` when one
was requested. `plot_streamplot` and `plot_contour_map` are aliases for users
familiar with the corresponding `Graph2D` names.

### Phase 2: common mathematical constructions

`plot_piecewise()` renders interval pieces with independently open or closed
endpoint markers. Each specification has the form
`(interval, function[, closure])`; closures accept names such as
`"closed-open"` or compact notation such as `"[)"`.

```python
result = kw.plot_piecewise([
    ((-2, 0), lambda x: -x, "[)"),
    ((0, 2), lambda x: x**2, "[]"),
], sampling="adaptive", show=False)

result.lines
result.endpoint_markers
result.samples
```

Additional functions in this phase are:

- `plot_region(lower, upper, interval=...)` for areas between functions.
- `plot_inequality("x^2 + y^2 <= 4", ...)` for two-variable solution sets.
- `plot_parametric(x, y, t_range=...)` and `plot_polar(radius, ...)` as direct
  alternatives to constructing curve objects.
- `plot_sequence(sequence, ...)` for KiwiCalc sequences, callables, or explicit
  values. Its `stop` index is inclusive.
- `plot_error_band(function, error, ...)` for symmetric scalar or callable
  uncertainty bands.

Piecewise, region, and error-band sampling stays fixed by default. Their
`sampling="adaptive"` option uses the same bounded sampling controls as
`plot_function()`.

### Phase 3: numerical and dynamical visualizations

`plot_phase_portrait()` combines streamlines with bounded fixed-step RK4
trajectories and marks equilibria found on its sampling grid. Supply either two
field components or one callable returning a pair.

`plot_complex_function()` supports domain coloring and `magnitude`, `phase`,
`real`, or `imaginary` scalar modes. `plot_convergence()` accepts a numeric
history or a traced `NumericalExplanation`. `plot_transform()` compares a
`Curve2D` or an `(n, 2)` point array before and after a callable or homogeneous
3-by-3 transformation. `plot_bifurcation()` renders the retained long-run
states of a one-dimensional iterated map.

```python
portrait = kw.plot_phase_portrait(
    lambda x, y: -y, lambda x, y: x,
    initial_conditions=[(0.5, 0), (1, 0)], show=False,
)

diagram = kw.plot_bifurcation(
    lambda state, r: r*state*(1-state),
    parameter_range=(2.8, 4),
    parameter_samples=500, burn_in=300, keep=80,
    show=False,
)
```

Expensive APIs expose explicit bounds. Phase portraits use `trajectory_steps`,
`max_field_points`, `max_trajectory_points`, and `escape`; bifurcation diagrams
enforce both `max_points` and `max_iterations`; complex and inequality plots
cap their `samples × samples` grids. These visualizations are numerical
evidence, not proofs of completeness or stability.

## Adaptive sampling

Function plots continue to use fixed sampling by default, preserving their
existing coordinate counts and performance. Opt in with `sampling="adaptive"`
when a fixed grid may miss narrow peaks, alias oscillations, or draw across a
pole:

```python
import numpy as np
import kiwicalc as kw

wave = lambda x: np.sin(100 * np.pi * x)
line = kw.plot_function(
    wave, start=0, stop=0.2, step=0.01,
    sampling="adaptive", tolerance=1e-3, show=False,
)

sample = line.kiwicalc_sample
print(sample.point_count, sample.evaluations, sample.truncated)
```

The seed grid still comes from `start`, `stop`, and `step`, or from explicit
strictly increasing `values`. Adaptive sampling probes interval midpoints and
subdivides intervals whose normalized interpolation error exceeds `tolerance`.
`max_points` and `max_depth` are hard work limits; `PlotSample.truncated` is
true when either limit prevents further requested refinement. Undefined values
are represented by `NaN`, which gives Matplotlib a line break.

The same options are available on `plot_function()`, `scatter_function()`,
their multi-function variants, `plot_multiple()`, and `Graph2D.plot()` or
`Graph2D.scatter()`. Each adaptively sampled artist exposes its diagnostic
`PlotSample` as `artist.kiwicalc_sample`; a graph collects the latest results in
`graph.sampling_results`. `kw.sample_for_plot()` provides the sampler directly
when coordinates are needed without rendering.

Adaptive work depends on the function: smooth nearly linear functions often
need few additional points, while oscillatory or singular functions may reach
the configured limit. No finite sampler can guarantee detection of every
feature hidden between its probes. This sampler currently applies only to
sampled two-dimensional functions. Parametric and polar curves retain their
separate curve-sampling API (including its existing adaptive mode), while 3D
plots and educational explanation layers retain their own fixed strategies.

## Reusable graphs

Repeated calls to `plot()` replace artists produced by the preceding graph
render. They do not remove unrelated artists that a caller added directly to
the axes. `clear()` removes the graph's rendered artists, decorations,
colorbars, secondary axes, legends, and motion controllers while retaining the
underlying Matplotlib axes for reuse.

`Graph2D.scatter()` follows the same lifecycle and renders sampled functions,
expressions, and ordinary curves as points. Geometry, vectors, implicit
contours, and explanatory layers keep their natural renderers.

```python
graph = kw.Graph2D().add("x^2", label="quadratic")
graph.plot(show=False, title="First view", legend=True)
graph.plot(show=False, title="", legend=False)  # replaces the earlier render
graph.clear()
```

`Graph3D` creates its figure lazily, so configuring or serializing an unrendered
graph does not open an empty notebook figure.

Themes and axis configuration are serialized even before the first render.
Portable JSON has a schema `version` field; raw Python callables and secondary-
axis conversion functions remain intentionally non-portable.

## Exporting figures

`save()` and its `export()` alias write PNG, SVG, or PDF files and return the
resulting `Path`. Graphs which have not been plotted are rendered automatically;
use `plot_options={...}` to control that first render. An already-rendered graph
is exported unchanged unless `render=True` is requested.

```python
graph = kw.Graph2D(["x^2"])
graph.save("quadratic.svg", plot_options={"title": "Quadratic"})
png_bytes = graph.to_bytes(format="png", render=False)
svg_text = graph.to_svg(render=False)
```

`to_bytes()` avoids temporary files in web services and notebook integrations.
`to_svg()` provides SVG as text for direct embedding. Export options include
`dpi`, `transparent`, `tight`, and any additional Matplotlib `savefig` keyword.

## Educational approximations

Roots, intersections, extrema, inflections, derivatives, integrals, and
asymptotes are numerical visual explanations. They are not symbolic proofs.
Root markers require a verified zero rather than merely a small local minimum.
Derivative, monotonicity, and cumulative-integral layers are split at undefined
function values so they do not visually bridge a discontinuity.

Increase `samples=` or provide a narrower `domain=` when a feature is small
relative to the displayed range. Use KiwiCalc's numerical APIs when a numeric
result and convergence information are required independently of a plot.

## Themes

Themes are local to a graph or supplied axes and never change Matplotlib's
global `rcParams`. Custom themes validate positive sizes, opacity values,
colors, and non-empty color cycles when they are constructed.

```python
theme = kw.PlotTheme(
    name="school",
    line_width=2.5,
    color_cycle=("#0072B2", "#D55E00"),
)
kw.Graph2D(["sin(x)"]).theme(theme).plot(show=False)
```

## Parameter interaction

Animations estimate limits from every requested frame. Slider interactions
sample the parameter interval and expand their limits if a later value exceeds
the initial envelope. Passing an explicit `ylim=` keeps fixed limits. Slider
values must remain inside `parameter_range`, and animation intervals must be
positive.
