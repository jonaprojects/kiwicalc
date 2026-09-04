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
