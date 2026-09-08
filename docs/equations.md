# Equations

KiwiCalc supports direct equation-solving functions, equation objects, linear
systems, and local Newton solving for polynomial systems. This guide documents
the compatibility contract of the existing API. Solver result containers are
intentionally not unified in the 1.x API.

## Accepted syntax

Equation strings contain exactly one separator and two non-empty sides. The
polynomial parser accepts:

- integers, decimals, and scientific notation;
- `^` and `**` powers with non-negative integer exponents;
- unary `+` and `-`;
- explicit and implicit multiplication, including `2*x`, `2x`, `2(x+1)`, and
  `(x-1)(x+1)`;
- nested parentheses.

For compatibility, automatically inferred adjacent letters are individual
variables: `xy` means `x*y`. Multi-character variables are supported when they
are supplied explicitly, for example:

```python
equation = kw.QuadraticEquation("theta^2 = 1", variables=("theta",))
```

The legacy coefficient-dictionary parser cannot represent mixed monomials such
as `xy`; it raises `UnsupportedExpressionError` instead of discarding part of
the term. `Poly` and `poly_from_str()` do represent such terms.

Parsing failures derive from `ValueError`:

- `EquationParseError` — invalid structure or tokens;
- `UnsupportedExpressionError` — valid-looking mathematics outside the
  supported polynomial grammar;
- `AmbiguousVariableError` — an explicit variable list cannot resolve a name.

## Scalar solvers

### Linear equations

```python
kw.solve_linear("3x + 5 = 8")            # 1.0
kw.solve_linear("x = x")                 # numpy.inf
kw.solve_linear("1 = 2")                 # None
kw.solve_linear("2x = 8", get_dict=True) # {'x': 4.0}
```

`solve_linear()` returns a scalar for one solution, `np.inf` for an identity,
and `None` for a contradiction. `get_dict=True` and `get_json=True` retain their
legacy formats. More than one non-zero variable is rejected.

`solve_linear_inequality()` handles one-variable linear inequalities and returns
a string such as `"x<=2"`. The comparison direction is reversed when division
by a negative coefficient is required.

### Quadratic equations

`solve_quadratic(a, b, c)` returns a tuple in historical `+sqrt`, `-sqrt`
order. A repeated root appears twice. Complex roots are supported.

Degenerate numeric inputs retain these results:

| Coefficients | Result |
|---|---|
| `a != 0` | two-item tuple |
| `a == 0, b != 0` | one-item tuple |
| `a == b == 0, c != 0` | empty tuple |
| `a == b == c == 0` | `None` |

`solve_quadratic_real()` returns a two-item tuple, a scalar repeated root, or
`None`. `solve_quadratic_from_str()` and the string overload of
`solve_quadratic()` normalize both equation sides before dispatch.

Coefficients are scaled before evaluating the discriminant. This prevents
avoidable overflow and underflow and uses Vieta's product to recover a small
root when the direct formula would suffer cancellation.

### Cubic, quartic, and general polynomials

`solve_cubic()` and `solve_quartic()` return lists of unique complex roots,
sorted by real and then imaginary component. Repeated numerical root clouds are
consolidated only after residual validation. A zero leading coefficient
delegates to the next lower-degree solver, so the delegated return container is
preserved.

`solve_polynomial(coefficients, epsilon=1e-6, nmax=10000)` removes leading
zeros and dispatches by effective degree. Constants and the zero polynomial
return `None`. Degrees two through four use the solvers above; higher degrees
use Aberth–Ehrlich and retain its legacy `set` result. Set iteration order is
not a contract. Aberth raises `RuntimeError` on numerical breakdown or exhausted
iterations rather than returning an incomplete root collection.

`solve_poly_by_factoring()` is an opportunistic rational-root method. It may
return only the roots it can recover and should not replace `solve_polynomial()`
when completeness is required.

## Numerical accuracy

For coefficients `a[0] ... a[n]` and candidate root `r`, KiwiCalc uses the
normalized residual

```text
|p(r)| / max(1, sum(|a[i]| * max(1, |r|) ** (n-i)))
```

This is scale-aware: multiplying every coefficient by a non-zero constant does
not change the quality measure. Fixed-degree roots are polished when the root
is demonstrably simple. A `RuntimeWarning` is emitted if a cubic or quartic
candidate retains a normalized residual above `1e-8`.

For high-degree solving, `epsilon` controls both relative step size and the
coefficient-scaled residual. Smaller values can improve accuracy but require
more iterations. Clustered and repeated roots are inherently ill-conditioned.

All numerical solvers reject non-finite coefficients. Iterative controls require
a positive finite `epsilon` and a positive integer `nmax`.

## Equation objects

The existing classes remain available:

```python
kw.LinearEquation("2x + 1 = 5")
kw.QuadraticEquation("t^2 = 1")
kw.CubicEquation("y^3 - 1 = 0")
kw.QuarticEquation("z^4 = z^2")
kw.PolyEquation("(x - 1)(x + 1) = 0")
```

Fixed-degree objects normalize both sides and delegate to the corresponding
function solver. Copying and `reversed()` preserve the concrete class. Mutating
methods such as `LinearEquation.simplify()` invalidate cached solutions.
Fixed-degree classes can be inspected with several variables but solving them
remains a one-variable operation.

## Systems

`solve_linear_system(equations, variables=None)` accepts square or rectangular
systems and returns a dictionary in variable order. It uses scale-derived SVD
rank tolerances and distinguishes:

- a unique consistent solution;
- an inconsistent system (`ValueError`);
- an underdetermined/non-unique system (`ValueError`).

`LinearSystem` preserves its list-based matrix methods and delegates solution
calculation to `solve_linear_system()`.

`solve_poly_system()` performs damped Newton iteration and returns one local
real solution near `initial_vals`; it does **not** enumerate every solution.
Backtracking rejects steps that increase the residual, and a least-squares step
is attempted for a near-singular Jacobian. Irrecoverable singularity,
stagnation, or non-finite evaluation raises `ValueError`. Exhausting `nmax`
retains the legacy behavior: return the last approximation and emit a warning.
With `show_steps=True`, assignments and residuals are printed for each iteration.

```python
kw.solve_poly_system(
    ("(x+y)-3=0", "2(x-y)=2"),
    initial_vals={"x": 0.0, "y": 0.0},
)
```

## Random equation generators

Random generators validate ranges, precision, degree, and variable names before
sampling. Retry loops are bounded, generated linear systems are full rank, and
advertised solutions satisfy the generated equations within their displayed
rounding precision. Corrective resampling means exact strings for a particular
global random seed are not a compatibility guarantee.

## Unified native symbolic solver

`solve_equation()` is the additive structured API. Existing solver functions
and their return containers remain unchanged.

```python
exact = kw.solve_equation("(x - 2)^2 = 0", steps=True)
exact.solution_set.values          # (ExactNumber(2),)
exact.solution_set.multiplicities  # (2,)

periodic = kw.solve_equation("sin(x) = 0")
print(periodic.solution_set)        # x = n*pi, n in Z

approximate = kw.solve_equation("cos(x) = x", interval=(0, 1))
approximate.solutions               # approximately (0.739085...,)
```

The signature is:

```python
solve_equation(
    equation, variable=None, *, domain="real", interval=None,
    method="auto", numeric_fallback=True, tolerance=1e-10,
    max_iterations=1000, steps=False,
)
```

Inputs may be equation strings, existing `Equation` objects, or `(left, right)`
expression pairs. The default domain is real. Use `domain="complex"` for exact
polynomial complex roots. If an equation has several variables, pass the target
with `variable=`; other variables are treated as parameters.

The native symbolic engine supports exact constant, linear, rational,
quadratic, factorable polynomial, algebraic `RootOf`, isolated square-root,
absolute-value, common exponential/logarithmic, and canonical affine
trigonometric equations. It uses exact rational arithmetic and checks finite
candidates against the original equation to reject invalid denominator and
squaring candidates.

`EquationSolution` reports the `solution_set`, `status`, `method`, `exact`,
`complete`, domain conditions, normalized residuals, optional structured steps,
and numerical evaluation count. Solution sets can be empty, universal, finite,
interval, conditional, unions, or integer-parameterized families. All result
and symbolic expression types support deterministic `to_dict()` round trips.

With `method="symbolic"`, an unsupported family returns a structured unresolved
result. With `method="numeric"`, or automatic fallback after symbolic rules are
exhausted, a finite `interval=(lower, upper)` is required. Numerical isolation
is discontinuity-aware and reports approximate answers with `complete=False`:
for arbitrary functions, a sampled finite search cannot prove that every root
was found.

`solve_equation_system()` performs exact rational row reduction for square or
rectangular linear systems, including parametric underdetermined results.
Nonlinear systems remain explicitly local and approximate: set
`numeric_fallback=True` and provide `initial` values to use damped Newton.
