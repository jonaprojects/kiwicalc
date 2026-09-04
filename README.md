# KiwiCalc

**Write mathematics naturally in Python.**

[![CI](https://github.com/jonaprojects/kiwicalc/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jonaprojects/kiwicalc/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-944%20passed-brightgreen.svg)](#development)
[![Line coverage](https://img.shields.io/badge/line%20coverage-94.19%25-brightgreen.svg)](#development)
[![Branch coverage](https://img.shields.io/badge/branch%20coverage-91.31%25-brightgreen.svg)](#development)
[![PyPI version](https://img.shields.io/pypi/v/kiwicalc.svg)](https://pypi.org/project/kiwicalc/)
[![PyPI downloads](https://img.shields.io/pypi/dm/kiwicalc.svg)](https://pypi.org/project/kiwicalc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/github/license/jonaprojects/kiwicalc.svg)](LICENSE)

![KiwiCalc function examples](kiwicalc_functions16x9.gif)

KiwiCalc is a Python mathematics library built around readable, math-like expressions. It combines symbolic expressions, equation solving, numerical methods, geometry, linear algebra, plotting, probability, sequences, and printable worksheets behind one approachable API.

## Why KiwiCalc?

Python is excellent for numerical work, but sophisticated mathematical expressions can become difficult to read and manipulate. KiwiCalc lets you construct expressions with familiar notation, substitute values, simplify them, convert them to callable functions, solve equations, and visualize results.

```python
import kiwicalc as kw

x = kw.Var("x")
expression = x**2 + 6*x + 8

print(expression)                         # x^2+6x+8
print(expression.when(x=2).try_evaluate())  # 24
print(kw.solve_quadratic(1, 6, 8))       # (-2, -4)
```

## Installation

Install the latest published version from PyPI:

```bash
python -m pip install kiwicalc
```

KiwiCalc supports Python 3.8 and newer.

## Quick tour

### Expressions and functions

```python
import math
import kiwicalc as kw

x = kw.Var("x")
wave = kw.Sin(x) + kw.Cos(x)

print(wave.when(x=math.pi).try_evaluate())

parabola = kw.Function("f(x) = x^2 + 2x + 1")
print(parabola(3))  # 16
```

### Equations

```python
import kiwicalc as kw

print(kw.solve_linear("3x + 5 = 8"))
print(kw.solve_quadratic(1, 6, 8))

system = kw.LinearSystem((
    "x + y = 5",
    "2x - y = 1",
))
print(system.get_solutions())
```

### Linear algebra and geometry

```python
import kiwicalc as kw

matrix = kw.Matrix([[1, 2], [3, 4]])
column = kw.Matrix.column_vector([5, 6])
vector = kw.Vector([3, 4])
point = kw.Point2D(2, 5)

print(matrix.determinant())
print(matrix @ column)
print(matrix.rref())       # returns a new matrix
print(matrix.rank())       # tolerance-aware for numeric matrices
print(vector.length())
print(point)
```

Convenient constructors and NumPy conversion are also available:

```python
import numpy as np

identity = kw.Matrix.identity(3)
zeroes = kw.Matrix.zeros(2, 3)
matrix = kw.Matrix.from_numpy(np.array([[1, 2], [3, 4]]))
numpy_array = matrix.to_numpy()

# Element-wise multiplication is explicit and does not mutate either operand.
product = matrix.hadamard([[2, 2], [2, 2]])
```

Solving systems is deliberately concise. A flat right-hand side is treated as
a column vector:

```python
A = kw.Matrix([[3, 1], [1, 2]])
x = A.solve([9, 8])                    # [[2], [3]]

fit = A.least_squares([9, 8])
pinv = A.pseudoinverse()
condition = A.condition_number()

details = A.solve([9, 8], return_info=True)
print(details.solution, details.residual_norm, details.rank)
```

Common matrix decompositions return named results and can also be unpacked:

```python
P, L, U = A.lu()             # P @ A == L @ U
Q, R = A.qr()
lower = A.cholesky()

svd = A.svd()
U, singular_values, Vt = svd
original = svd.reconstruct()

eigenvalues, eigenvectors = A.eigen()
hermitian_values, hermitian_vectors = A.eigh()
```

For complex matrices, `A.H` returns the conjugate transpose; `A.T` remains the
ordinary transpose.

Vector-space tools return a small `VectorSpaceBasis` object. Its vectors are
column matrices, while `basis.matrix` combines them as columns:

```python
A = kw.Matrix([[1, 2, 3], [2, 4, 6]])

columns = A.column_space()
rows = A.row_space()
kernel = A.null_space()                 # stable SVD basis
simple_kernel = A.null_space(method="rref")

print(kernel.dimension)
print(kernel.matrix)
print(A.is_independent(axis="rows"))

orthonormal = A.orthonormalize(axis="rows")
projection = A.T.project_onto([1, 2, 3])

steps = A.orthonormalize(axis="rows", return_steps=True)
print(steps.basis, steps.steps)
```

The trivial space is represented explicitly: its basis has dimension zero,
`basis.matrix` is `None`, and `basis.to_numpy()` has shape `(ambient, 0)`.

### Plotting

```python
import kiwicalc as kw

x = kw.Var("x")
(kw.Sin(x) + 0.25*x).plot(start=-10, stop=10)
```

Curves and graphs use the same simple plotting style:

```python
graph = kw.Graph2D()
graph.add(kw.Ellipse(3, 2), label="ellipse", color="royalblue")
graph.add(kw.ArchimedeanSpiral(), label="spiral", color="orange")
graph.plot(legend=True, equal_aspect=True)
```

Graphs also provide scoped visualization themes and math-aware axes without
changing Matplotlib's global settings:

```python
import numpy as np

graph = (
    kw.Graph2D([lambda x: np.sin(x)])
    .theme("classroom")
    .secondary_xaxis(np.degrees, np.radians, label="Angle", unit="deg")
)
graph.plot(
    title="Sine wave",
    xlabel="Angle", ylabel="Amplitude", units=("rad", None),
    x_ticks="pi", xlim=(0, 2*np.pi), minor_ticks=True, minor_grid=True,
)
graph.save("sine-wave.svg")
```

Built-in themes include `classroom`, `projector`, `publication`, `engineering`,
and `colorblind`. Tick modes include `pi`, `degrees`, `scientific`, and
`engineering`; graphs can be exported directly as PNG, SVG, or PDF.

Parametric, polar, implicit, Bézier, and spline curves are available in 2D, along
with named curves such as cardioids, rose curves, cycloids, superellipses,
catenaries, and involutes. Curves have chainable transformations and numerical
analysis helpers:

```python
curve = kw.Cardioid().scale(2).rotate(0.4).translate(1, 2)
print(curve.point_at(0.25), curve.tangent_at(0.25), curve.arc_length())

graph = kw.Graph2D([curve])
graph.mark(curve.point_at(0.25), label="sample point")
graph.vertical_line(1, linestyle="--")
graph.plot(legend=True, equal_aspect=True)
```

Graphs also include math-aware explanation helpers for teaching and engineering
visuals. They accept formulas, `Function` objects, expressions, or callables:

```python
f = "x^3 - 3*x"
graph = (
    kw.Graph2D()
    .add(f, label="f(x)")
    .show_roots(f)
    .show_extrema(f)
    .tangent(f, at=1)
    .slope_triangle(f, at=1, run=0.6)
)
graph.plot(title="Understanding a cubic", xlim=(-3, 3), ylim=(-5, 5), legend=True)
```

Additional helpers cover intersections, inflection points, normal and secant
lines, asymptotes, monotonic regions, inequality shading, Riemann sums, and
derivative or integral overlays.

Scientific fields use the same graph and theme API, with no additional plotting
dependency:

```python
flow = (
    kw.Graph2D()
    .theme("engineering")
    .streamlines(lambda x, y: -y, lambda x, y: x, colorbar=True)
    .contour_map(lambda x, y: x*x + y*y, levels=8, colors="gray")
)
flow.plot(title="Rotational flow", xlim=(-3, 3), ylim=(-3, 3), equal_aspect=True)
```

`vector_field`, `gradient_field`, and `slope_field` add quiver-style diagrams;
`streamlines` displays continuous flow; and `contour_map` supports line or
filled contours, numeric labels, and optional themed colorbars.

Parameterized animation and interaction are also built in:

```python
animation = kw.Graph2D().theme("classroom").animate(
    "f(x,a)=a*sin(x)",
    frames=np.linspace(0.5, 3, 40),
    parameter="a",
    title="Amplitude = {a:.2f}",
)

control = kw.Graph2D().interact(
    "f(x,k)=sin(k*x)",
    parameter_range=(0.5, 5),
    parameter="k", initial=1, step=0.1,
)
```

Animation controllers support pause, resume, GIF/video saving, and embeddable
HTML. Interaction controllers expose their slider value and can be updated
programmatically with `set_value`. Live notebook sliders use Matplotlib's
interactive backend when available; no widget package is required by KiwiCalc.

Matrices include teaching-oriented explanations and visualizations using the
same Matplotlib installation as the graph system. Row reduction is inspectable
as text, data, or a figure:

```python
A = kw.Matrix([[1, 2, 1], [2, 4, 0]])
explanation = A.explain_rref()
print(explanation.as_text())
explanation.plot(theme="classroom")
```

Transformation, eigenvector, and SVD geometry are one method call away:

```python
A = kw.Matrix([[2, 1], [0, 1]])
A.visualize_transformation(vectors=[(1, 1)], theme="engineering")
A.visualize_eigenvectors()
A.visualize_svd()
```

For regression lessons, a design matrix can display its observations, fitted
values, and residuals together. KiwiCalc automatically finds the one varying
predictor column; pass `x=...` for polynomial or multi-predictor designs.

```python
design = kw.Matrix([[1, 0], [1, 1], [1, 2], [1, 3]])
design.visualize_least_squares([1, 2.8, 5.2, 6.9])
```

Linear algebra and geometry meet through immutable affine transformations.
They work with KiwiCalc's existing points, vectors, point collections, and
curves—there is no second vector type to learn:

```python
transform = (
    kw.AffineTransformation.rotation(30, degrees=True)
    .scale(2)
    .translate(4, 1)
)

point = transform(kw.Point2D(1, 0))
vector = transform(kw.Vector2D(1, 0))
curve = transform(kw.Ellipse(3, 2))
```

Constructors cover 2D and 3D translation, scaling around a center, rotation
around a point or axis, shearing, and reflection. Transformations compose in
reading order with `.then(...)`, support inversion, and expose their linear
part, translation, determinant, orientation, and rigidity. Existing matrices
join the same workflow with `matrix.as_affine(translation=...)`.

Set `sampling="adaptive"` on a parametric or polar curve when you want its
sampling density to follow its shape. Space curves and surfaces—including
trefoil and figure-eight knots, paraboloids, hyperbolic paraboloids, and
hyperboloids—compose naturally in 3D:

```python
graph = kw.Graph3D()
graph.add(kw.Helix(turns=5), label="helix", color="purple")
graph.add(kw.Sphere(2), alpha=0.2, color="skyblue")
graph.plot(legend=True)
```

Supported curves, surfaces, and composed graphs can also be saved and restored:

```python
graph.export_json("graph.json")
restored = kw.Graph.from_json("graph.json")
restored.plot()
```

## What is included?

For ordinary single-variable functions, use the unified scalar numerical API:

```python
kw.differentiate(lambda x: x*x, at=3)                 # approximately 6
kw.integrate(lambda x: x*x, 0, 1)                    # approximately 1/3
kw.find_root(lambda x: x*x - 2, bracket=(0, 2))       # approximately sqrt(2)
kw.find_root(lambda x: x*x - 2, x0=1)                # Newton with numerical derivative

result = kw.find_root(lambda x: x*x - 2, x0=1, return_info=True)
print(result.value, result.converged, result.residual, result.function_calls)
```

These functions also accept existing single-variable `Function` objects and
expressions. Integration and root finding require scalar real outputs.
Differentiation accepts scalar points or arrays of independent single-variable
points; arrays are not implicitly interpreted as multivariable coordinates.

- `differentiate`: `central`, `forward`, `backward`, or `richardson`; optional `step`.
- `integrate`: `simpson`, `trapezoid`, or `midpoint`; `intervals` always counts
  subintervals. Simpson rounds odd interval counts up to the next even number.
- `find_root`: `auto` chooses Brent's method with `bracket`, secant with `x0` and
  `x1`, or Newton with `x0`. Explicit `halley` (both derivatives required) and
  `steffensen` are also available. A supplied `derivative` is used by Newton.

`return_info=True` returns a `NumericalResult`. Fixed-resolution integration
and differentiation report no convergence status or estimated error. Root
solvers require `abs(f(root)) <= tolerance`: failure raises `RuntimeError`, or
returns `converged=False` when diagnostics are requested. Callback exceptions
and invalid inputs always propagate; underlying solver warnings are retained.
The existing algorithm-specific functions remain available and unchanged.

Array differentiation and Richardson extrapolation:

```python
import numpy as np

x = np.linspace(0, 2, 100)
dy = kw.differentiate(np.sin, at=x, method="richardson", vectorized=True)
# Alternatively, omit vectorized=True for scalar-only callbacks such as math.sin.
```

Array results preserve the shape of `at`. `step` can be scalar or broadcast to
that shape. Richardson combines central differences at `step` and `step/2`;
its diagnostic `estimated_error` is the correction magnitude, not an error bound.
`vectorized=True` explicitly requests one array callback per stencil sample;
the callback must return exactly the input shape. No exception-based retries occur.

Adaptive integration and sampled data:

```python
area = kw.integrate(np.sin, 0, np.pi, method="adaptive_simpson", tolerance=1e-9)
root = kw.find_root(lambda x: np.cos(x) - x, bracket=(0, 1), method="brent")

y = x**2
slopes = kw.differentiate_samples(y, x)
running_area = kw.cumulative_integrate(y, x)
```

Adaptive Simpson reuses samples and refines the panel with the largest estimated
error. `tolerance` is an absolute estimated-error target; `max_evaluations` and
`max_depth` bound work. With `return_info=True`, inspect `estimated_error` and
`converged`; otherwise exhausted limits raise `RuntimeError`. Error estimation
assumes a sufficiently smooth, resolved function: narrow peaks and discontinuities
can be missed. Fixed-grid Simpson remains the default; its `intervals` option is
not used by the adaptive method.

Sampled-data helpers accept nonuniform strictly increasing or decreasing `x`,
or uniform `spacing`. For multidimensional signals, choose `axis` (default -1).
Both preserve input shape. Differentiation uses three-point differences with
`edge_order=1` or `2` (default 2); two samples fall back to their secant slope.
Cumulative integration uses trapezoids and starts at `initial=0`, optionally an
additive integration constant. These helpers do not smooth noisy measurements.

Brent combines bracketing, bisection, and inverse quadratic interpolation
([algorithm overview](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)).
KiwiCalc retains its absolute-residual success criterion, even if the bracket
has become tiny; select `method="bisection"` to retain the earlier dispatch.
These additions use the existing NumPy dependency, not a new runtime dependency.

For multivariable functions, use separate, explicit entry points:

```python
kw.gradient(lambda x, y: x*x + 3*y*y, at=(2, 1))     # [4, 6]
kw.jacobian(lambda x, y: [x*x + y, x*y], at=(2, 3))  # [[4, 1], [3, 2]]
kw.hessian(lambda x, y: x*x + 3*x*y + y*y, at=(1, 2))  # [[2, 3], [3, 2]]

kw.solve_system(
    lambda x, y: [x*x + y*y - 2, x - y], initial=(0.8, 1.2)
)  # approximately [1, 1]
kw.integrate_nd(lambda x, y: x + y, bounds=[(0, 1), (0, 2)])  # approximately 3
```

Callbacks receive unpacked coordinates by default. For array-based functions,
pass `argument_style="vector"`, for example
`kw.gradient(lambda p: p @ p, at=[1, 2], argument_style="vector")`.
No new vector class is required; derivative and system-solution results are NumPy arrays.

- `gradient`, `jacobian`, and `hessian` accept one point `(n,)` or batches
  `(..., n)`. Their results have shapes `(..., n)`, `(..., m, n)`, and
  `(..., n, n)` respectively. Each callback receives one point, not a batch.
- `gradient` and `jacobian` support central, forward, and backward differences;
  `hessian` uses central differences. Optional `step` is scalar or one positive
  value per coordinate; automatic steps scale with coordinate magnitude.
- Existing expressions and `Function` objects are supported. Use
  `variables=("x", "y")` to specify coordinate order explicitly. Otherwise
  expressions use sorted names and `Function` uses declaration order; component
  lists combine those orders in first-encounter order.
- `jacobian` and `solve_system` also accept lists of scalar component functions.
  `solve_system` handles square nonlinear systems using damped Newton iterations,
  with an optional analytic `jacobian` callback. This is a local solver: the
  initial guess matters, and convergence is not guaranteed.
- `integrate_nd` integrates finite rectangular domains using `midpoint` or
  `trapezoid` grids. `intervals` can be one count or a count per axis.
  `max_evaluations` (default 100,000) guards against exponential grid growth.

All five methods accept `return_info=True`. System diagnostics include
`converged`, the residual infinity norm, and `iterations`; non-convergence raises
`RuntimeError` unless diagnostics were requested. Finite differences and fixed-grid
integration do not claim a convergence test or error estimate. The scalar API
remains separate; its differentiation method now also handles arrays of points.

Numerical integration keeps the legacy function names: `reinman(f, a, b, N)`
now uses midpoint samples across `N-1` intervals, and `simpson(f, a, b, N)`
increases an even sample count to the next odd count before constructing its
grid. Both cover the complete interval, including reversed bounds.

`aberth_method(f, derivative, coefficients)` uses distinct initial guesses,
relative step checks, and coefficient-scaled residuals. Returned roots are no
longer rounded or merged by a fixed distance; the historical set return type
remains. Invalid input and non-convergence now raise explicit errors instead
of silently returning an empty or incomplete set. Repeated and tightly
clustered roots remain numerically difficult.

Bairstow's method finds real and complex polynomial roots directly from
coefficients ordered from highest power to constant:

```python
roots = kw.bairstow_method([1, 0, 0, 0, -1])  # x^4 - 1: ±1 and ±i
```

It returns a list preserving repeated roots. Optional `r` and `s` guesses
describe the quadratic factor `x² - r*x - s`; `epsilon` controls the relative
coefficient remainder, and `nmax` limits iterations per factor. Non-convergence
raises `RuntimeError`. Repeated or ill-conditioned roots may have substantially
less accuracy than the remainder tolerance suggests.

- Symbolic monomials, polynomials, fractions, roots, logarithms, trigonometry, factorials, and composite expressions
- Linear, quadratic, cubic, quartic, polynomial, and system solving
- Numerical root-finding, integration, differentiation, and optimization methods
- Callable functions, function collections, and function chains
- Matrices, vectors, points, lines, conic sections, curves, surfaces, and point collections
- Two- and three-dimensional plotting with Matplotlib
- Scoped visualization themes, intelligent mathematical axes, secondary axes, and figure export
- Arithmetic, geometric, and recursive sequences
- Probability trees
- PDF exercise and worksheet generation
- JSON serialization for supported expressions, curves, surfaces, and composed graphs

## Numerical explanations for teaching

Opt in with `explain=True` to retain actual numerical steps. Normal calls keep
their existing solver loops: the explanation dispatch is outside those loops,
with no per-iteration trace checks, trace objects, or plotting work.

```python
lesson = kw.find_root(
    lambda x: x*x - 2, bracket=(0, 2), method="bisection", explain=True,
)
lesson.plot_steps(0)          # zero-based step; omitted index shows the last record
lesson.plot_convergence()    # measured residual, not unknown true error

area = kw.integrate(lambda x: x*x, 0, 2, intervals=6, method="simpson", explain=True)
area.plot_steps(1)

# In a Jupyter notebook:
from IPython.display import HTML, display
player = lesson.animate()
display(HTML(player.to_jshtml()))
```

Supported root explanations are **bisection, Newton, and secant**; supported
quadrature explanations are **midpoint, trapezoid, and Simpson**. Select bisection
explicitly: bracketed `method="auto"` still chooses Brent, whose explanation is
not implemented. Unsupported teaching methods raise rather than silently switch.

`NumericalExplanation.result` contains the usual diagnostics; `.steps` is a tuple
of frozen `NumericalStep` records containing samples, formulas, decisions, and
estimates. `explain=True` returns an explanation even on solver non-convergence,
so failures can be taught; callback errors and invalid inputs still raise.
`return_info=True` is redundant with explanations. Legacy solver warnings remain.

`trace_limit=1000` caps saved records without changing the computation or result.
`.truncated` and `.total_steps` expose omitted records; plots label truncation.
For quadrature, records are geometric panels built from actual evaluations,
**not the order of floating-point summation**. Simpson retains the existing
odd/even summation order, while each plotted parabola spans two subintervals.

Plotting lazily samples a background curve and may evaluate the callback again;
use pure callbacks. Rendering never changes saved records or diagnostic call
counts. Animation samples the background once, then replays the saved records.
Its HTML player includes play/pause, previous/next, timeline, and speed controls
without widget extensions. Native GUI use also offers `next()`, `previous()`,
`play()`, `pause()`, and `set_speed()`. HTML export closes the source figure to
avoid an extra static notebook plot. Keep traces short for lightweight HTML.

See the local, Git-ignored `examples/numerical_methods_for_educators.ipynb` for all
six methods, convergence comparisons, animation, stalled solves, and trace limits.
Run `python -m scripts.benchmark_tracing` for local median normal/traced timings;
plotting is excluded and timings are not CI assertions. Regression tests compare
results, callback order/counts, and warnings between traced and normal execution.

## Initial-value differential equations

`solve_ivp` solves non-stiff, real-valued equations with the same callback style
for scalar states and systems: `f(t, state)`.

```python
solution = kw.solve_ivp(lambda t, y: -0.5*y, t_span=(0, 10), initial=2)
print(solution.t, solution.y, solution.success)
solution.plot(title="Exponential decay")

def oscillator(t, state):
    position, velocity = state
    return [velocity, -position]

motion = kw.solve_ivp(
    oscillator, t_span=(0, 20), initial=[1, 0],
    t_eval=np.linspace(0, 20, 201), rtol=1e-8, atol=1e-10,
)
motion.plot(labels=["position", "velocity"], title="Harmonic oscillator")
```

- The default `method="rk45"` uses the Dormand–Prince 5(4) pair
  ([method reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html)).
  `rtol` and `atol` control componentwise estimated local error; `atol` may be
  scalar or one positive tolerance per state component. These are not global
  error guarantees. No SciPy dependency is added.
- For fixed steps, use `method="rk4", step=0.01`. RK4 does not use tolerances
  for error control. Without `step`, both methods start at interval length/100;
  only RK45 subsequently adapts it. `max_step` caps step magnitude and
  `max_steps` limits attempts, including rejected steps.
- Scalar initial values receive scalar callbacks and return `y.shape == (times,)`.
  A vector initial value receives a copied one-dimensional array and returns
  `y.shape == (times, components)`. Callback outputs must match the initial shape.
- Backward integration is supported. Optional `t_eval` must be strictly ordered
  in the integration direction and lie inside `t_span`. The solver lands directly
  on requested times, so adding output times can change its step sequence.
  Without `t_eval`, results include the initial state and every accepted endpoint.
- `ODESolution` includes `status`, `message`, `steps`, `rejected_steps`,
  `function_calls`, and `event_calls`. Numerical failure raises `RuntimeError`;
  `raise_on_failure=False` returns a partial result with `success=False`.
  Invalid inputs and callback exceptions still propagate.

Events detect scalar zero crossings:

```python
halfway = kw.ODEEvent(lambda t, y: y - 0.5, terminal=True, direction=-1)
stopped = kw.solve_ivp(lambda t, y: -y, (0, 10), 1, events=halfway)
print(stopped.t_events[0])  # approximately [log(2)]
```

Pass one callback/event or a list. Plain callbacks are nonterminal and detect
both directions. `direction` is -1, 0, or +1 along integration order, including
backward solving. Exact initial zeros are recorded regardless of direction.
`t_events` and `y_events` contain one array per event. A terminal event stops
successfully and is appended to the output even if absent from `t_eval`.
Crossings are localized with RK substeps and time bisection to `event_tolerance`
(default 1e-10), subject to roundoff and ODE solution accuracy. Multiple crossings
inside one step and tangencies can be missed; reduce `max_step` to resolve them.

Solving creates no figures. `solution.to_graph()` returns an unrendered `Graph2D`
for further annotations; `solution.plot(show=False)` renders and returns that
graph without showing it. Select series with `components=[0]` and provide `labels`.
Stiff solvers, complex states, dense output, and boundary-value problems are not
part of this initial ODE API.

## Worksheet reliability notes

`PDFWorksheet.end_page()` creates or refreshes the current exercise page's
answer page; repeated calls do not duplicate answers. Adding more exercises
afterward still edits the exercise page. `create()` renders current page objects
and refreshes enabled answer pages, so repeated exports do not use stale text.
Deleting the last page restores the most recent surviving exercise page;
adding an exercise to a completely emptied worksheet starts a fresh page.

Two-point line exercises avoid vertical-line inputs and retain exact fractional
coefficients. Point-and-slope answers use the line's actual x-intercept.
Trigonometric, logarithmic, and linear-intersection worksheets are available via
`worksheet(dtype='trigo' | 'log' | 'intersection', seed=42, ...)`, or individually
as `PDFTrigonometricEquation`, `PDFLogarithmicEquation`, and
`PDFLinearIntersection`. These English-only generators use local random seeds;
legacy families retain their existing random behavior. Trigonometry covers
sine/cosine special angles on `0 <= x < 360` degrees; logarithmic answers include
domain checks. They are bounded exercise generators, not general equation solvers.

Core algebra exercises use one consistent, seeded API:

```python
exercise = kw.algebra_exercise('factor', difficulty='hard', seed=42)
sheet = kw.PDFWorksheet('Algebra', theme='classroom')
sheet.add_exercise(exercise).end_page()
sheet.create('algebra.pdf')

kw.worksheet('inequalities.pdf', dtype='linear_inequality',
             difficulty='medium', seed=42, equations_per_page=12)
```

The catalog covers simplifying, expanding, factoring, completing the square,
substitution, linear inequalities, absolute-value equations, exponent laws,
rational equations, radical equations, and rearranging formulas. Every exercise
provides exact solution metadata for verification and future interactive answer
checking. See [the core algebra exercise guide](docs/algebra_exercises.md).

Calculus and numerical-method exercises use the same friendly API:

```python
exercise = kw.calculus_exercise('newton', difficulty='hard', seed=42)
sheet = kw.PDFWorksheet('Calculus', theme='academic')
sheet.add_exercise(exercise).end_page()
```

The catalog covers derivatives from first principles, differentiation, tangent
lines, critical points, monotonicity, concavity, optimization, definite
integrals, area between curves, numerical differentiation, trapezoidal and
Simpson rules, Newton iteration, Euler's method, and classical RK4. See the
[calculus exercise guide](docs/calculus_exercises.md).

`PDFWorksheet.add_math(expression)` renders Matplotlib Mathtext without an external
LaTeX installation. `add_plot(figure_or_draw_callback)` embeds a Matplotlib Figure
or a callback receiving an axes object. Formulas and plots are raster images;
existing figures remain open. Function-analysis and intersection answer keys now
include sketches. Paragraphs wrap and paginate automatically; logical pages may
therefore span several physical PDF pages. `create(..., page_size='A4', margin=50,
font_size=12)` accepts A4, Letter, or dimensions in points. General localization
and full LaTeX support remain outside this API.

### Mixed text and mathematics

For reusable typography, colors, spacing, margins, headings, headers/footers,
captions, and writing areas, use `PDFStyle`. It works with both `PDFWorksheet`
and batch `worksheet()` generation. See [the PDF styling guide](docs/pdf_styling.md)
for every option, inheritance rules, examples, and current rendering limitations.

For the friendliest path, choose a coordinated PDF theme:

```python
sheet = kw.PDFWorksheet('Algebra practice', theme='classroom')
sheet.create('algebra.pdf')

school = kw.PDFTheme.get('classroom').with_options(
    primary='#24543D', heading='#24543D', body_size=13,
)
```

Available presets are `academic`, `classroom`, `assessment`, `engineering`,
`accessible`, and `ink_saver`. Themes use semantic color, typography, and
spacing tokens and validate readable text contrast. Existing `PDFStyle` APIs
remain compatible.

Control line height with `sheet.create('worksheet.pdf', line_height=1.25)`.
The default is `1.5` times each text style's font size, across A4, Letter, and
custom page sizes, for questions, answers, and headings. The same option works
with `create_pdf()` and `create_pages()`. It is independent of paragraph spacing;
tall inline formulas can expand a line to prevent clipping. Values must be
positive and finite.

Compose worksheets, individual pages, and polynomial reports before export when
they belong to one document. `PDFDocument` renders the content in one pass, so
its shared `PDFFooter` numbers every physical page continuously—including answer
keys and overflow pages—without manual offsets:

```python
footer = kw.PDFFooter("Practice | Page {page}", alignment="center")
document = kw.PDFDocument(style=kw.PDFStyle(footer=footer))
document.add(algebra_sheet).add(geometry_page).add_report(polynomial)
document.create("course_pack.pdf")
```

Each independent export starts at page 1 by default. Use `page_start` only when
an intentionally separate export needs a different starting number.

Use `PDFText('Solve ', PDFMath(r'\frac{x}{2}=3', font_size=12), ' for x.')`
as an exercise or solution. Prose is escaped literally; only explicit math
segments are interpreted as Mathtext. Plain strings keep their existing behavior.
`format_math(Fraction(1, 3))` preserves exact fractions, and
`format_polynomial([1, 0, -1])` formats descending-power coefficients as
`x^{2} -1`, omitting zero and unit coefficients. `format_math()` and `PDFMath()`
also accept `Mono` and `Poly` objects directly, without modifying them. Single-letter
variables, multivariate terms, negative powers, and numeric fractional exponents
are supported. Other expression types still require explicit Mathtext.
Equation classes, linear systems, function formulas and derivatives, and the
trigonometric, logarithmic, and intersection generators use this formatting
automatically while preserving their legacy plain-text representations.
Generated equation sides retain their unsimplified terms, preserving the exercise.
Legacy linear/quadratic/polynomial batch worksheets also render formatted equations.
Inline formulas wrap as indivisible units; oversized formulas should be placed
in `add_math()` blocks. Unsupported Mathtext raises a descriptive error.

## Documentation and learning resources

- [Full documentation](https://jona-projects.gitbook.io/kiwicalc)
- [Official website](https://jonaprojects.github.io/kiwicalc_landing_page/)
- [Google Colab examples](https://colab.research.google.com/drive/1x411iW1nczAp67YBfp55Erd-72Nd7k7Z?usp=sharing)
- [YouTube channel](https://www.youtube.com/channel/UCLjhA3oBWFVVUyC5c30hsag)

## Development

The current suite contains 1,216 passing tests with 94.05% line coverage and 90.23% branch coverage. CI requires both coverage metrics to remain at or above 90%.

Create an isolated environment and install the project with its development tools:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --editable ".[dev]"
python -m pytest --cov=kiwicalc --cov-report=term-missing --cov-report=json:coverage.json
python scripts/check_line_coverage.py coverage.json 90
python scripts/check_branch_coverage.py coverage.json 90
```

Pull requests and pushes to `main` run the test suite on Linux and Windows. Published GitHub Releases are built, validated, and uploaded to PyPI through Trusted Publishing when the release tag matches the version in `pyproject.toml`.

## Security note

KiwiCalc currently compiles some math-like expression strings into Python callables. Only evaluate expression strings from sources you trust; do not pass untrusted user input directly into the parser.

## License

KiwiCalc is available under the [MIT License](LICENSE).
