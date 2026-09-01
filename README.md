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
vector = kw.Vector([3, 4])
point = kw.Point2D(2, 5)

print(matrix.determinant())
print(vector.length())
print(point)
```

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

The current suite contains 1,135 passing tests with 94.14% line coverage and 90.52% branch coverage. CI requires both coverage metrics to remain at or above 90%.

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
