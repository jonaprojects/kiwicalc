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
quadratic, factorable polynomial, algebraic `RootOf`, repeated square-root,
absolute-value, `exp`, common constant-base exponential/logarithmic, and affine
trigonometric equations. Parameter-aware quadratic and biquadratic formulas
retain leading-coefficient, degree-reduction, discriminant, and identity
branches. Exact special angles are preferred; other real constant right sides
use `asin`, `acos`, or `atan` in complete periodic families. It uses exact
rational arithmetic and checks finite candidates against the original equation
to reject invalid denominator and repeated-squaring candidates.

### Symbolic-coefficient low-degree polynomials

The symbolic solver handles a general parameterized quadratic without assuming
that its leading coefficient or discriminant is nonzero:

```python
general = kw.solve_equation(
    "a*x^2 + b*x + c = 0", variable="x", steps=True,
)

positive = kw.solve_equation_assuming(
    "x^2 + p*x + 1 = 0", "p^2 - 4 > 0", variable="x",
)
```

Over the reals, the result separates positive, zero, and negative discriminant
cases. The `a = 0` branch reduces to the complete symbolic linear contract,
including its identity and contradiction cases. Over the complex domain, the
result distinguishes zero and nonzero discriminants so multiplicity remains
exact. Factored products such as `(a*x+b)*(x-1)*(x+2)=0` use the zero-product
rule, which also retains parameter-only identity branches.

Sparse symbolic quartics of the form `a*x^4+b*x^2+c=0` are solved by the exact
substitution `u=x^2`; every conditional outer root is passed back through the
same solver. General symbolic cubic and quartic formulas remain deliberately
unimplemented.

### Exact polynomial and rational inequalities

`solve_inequality()` returns the same immutable structured result and solution
set types as `solve_equation()`:

```python
outside = kw.solve_inequality("x^2 - 5*x + 6 >= 0", steps=True)
between = kw.solve_inequality("x^2 - 2 < 0")
rational = kw.solve_inequality("(x^2 - 1)/(x - 1) > 0")
```

The API supports `<`, `<=`, `>`, `>=`, and `!=` for exact real univariate
polynomial or rational expressions. It cancels polynomial GCDs for sign
analysis while retaining all original denominator holes. Exact roots and poles
partition the real line; signs propagate by root/pole multiplicity, so repeated
roots do not spuriously flip a sign. Unbounded intervals use `None` for their
infinite endpoint and serialize deterministically.

Symbolic coefficient values may be supplied through `assumptions={...}` before
normalization. General parameter-dependent inequalities, multivariable
inequalities, chained relations, and complex ordering remain unsupported.

### Repeated and nested square roots

The guarded solver repeatedly isolates additive principal square roots and
expands only the bounded square needed for the next step:

```python
two = kw.solve_equation("sqrt(x+1) + sqrt(x-1) = 3", steps=True)
nested = kw.solve_equation("sqrt(x + sqrt(x)) = 2", steps=True)
```

Every radicand and isolated right side contributes a non-negativity condition.
Expansion is capped at 64 additive terms and 512 expression nodes. All final
candidates are checked in the original equation, so roots introduced by any
of the repeated squaring stages are rejected. This phase targets additive and
nested square roots; nonadditive products of distinct radicals and general
rational-power systems remain unresolved.

### Trigonometric reductions

Before algebraic substitution, a bounded real-only rule can apply the
Pythagorean identity and double- or triple-angle formulas when doing so reduces
the equation to a polynomial in an already present base-angle function:

```python
kw.solve_equation("sin(x)^2 + cos(x) = 1")
kw.solve_equation("sin(2*x) = cos(x)")
kw.solve_equation("cos(3*x) = cos(x)")
```

Common factors created by a double-angle reduction use the exact zero-product
rule, and the resulting one-function equations return complete periodic
families. The rule strictly lowers the angle multiplier and is bounded to 512
nodes to prevent rewrite cycles or expansion growth. General sum-to-product,
phase-shift combination, higher multiple-angle, and complex trigonometric
reduction remain unsupported.

Restrictions belong to the original expression, not merely its simplified
form. For example, `0/x = 0` returns the universal real solution with `x != 0`,
and `ln(x) = ln(x)` retains `x > 0`. Intervals are intersected with every
solution-set shape, including identities and conditional parameter branches.
A real interval cannot be combined with `domain="complex"`.

`EquationSolution` reports the `solution_set`, `status`, `method`, `exact`,
`complete`, domain conditions, normalized residuals, optional structured steps,
and numerical evaluation count. Solution sets can be empty, universal, finite,
interval, conditional, unions, or integer-parameterized families. All result
and symbolic expression types support deterministic `to_dict()` round trips
and strict JSON encoding. Real irreducible polynomial roots use exact Sturm
isolation intervals; numerical values obtained from `RootOf.evaluate()` are
approximations of those certified algebraic roots.

### Basic canonical-form contract

`simplify_symbolic()` produces a deterministic, immutable **basic canonical
form**. It reduces exact numeric constants, flattens nested sums and products,
collects like additive terms, removes additive zero and multiplicative one,
uses deterministic ordering for commutative operands, reduces trivial powers,
and evaluates the documented exact elementary-function values.

The canonicalization contract is:

- **Idempotent:** simplifying an already simplified expression makes no change.
- **Deterministic:** construction order and repeated runs do not alter the result.
- **Exact:** numeric folding uses rational arithmetic rather than binary floating
  approximation.
- **Domain-safe:** cancellation and multiplication by zero retain guarded
  subexpressions when the original expression can be undefined.
- **Non-mutating:** neither the input tree nor a previously returned tree changes.
- **Serialization-stable:** a serialized round trip returns the same canonical
  tree.

Use `is_canonical_symbolic(expression)` to check the fixed-point invariant.

```python
canonical = kw.simplify_symbolic("y + 2 + x + y + 1")
str(canonical)                              # "3 + 2*y + x"
kw.is_canonical_symbolic(canonical)         # True
kw.simplify_symbolic(canonical) == canonical  # True
```

Basic simplification intentionally does not promise expansion, factorization,
rational-function cancellation, or general function identities. For example,
`(x + 1)^2` and `x^2 + 2*x + 1` need not have the same basic canonical tree.
Expansion and rational-function cancellation are available through the
explicit normalization operations below. Factorization and broader identities
remain separate operations; keeping them out of basic simplification prevents
rewrite cycles and uncontrolled expression growth.

Domain preservation takes precedence over cosmetic reduction. For example,
`(1/x)^0` retains enough structure for equation solving to preserve `x != 0`,
rather than becoming an unconditionally defined constant.

### Polynomial and rational normalization

`normalize_polynomial_symbolic()` expands a univariate polynomial into an
exact, deterministic sum of powers. Coefficients use rational arithmetic, so
the result does not acquire floating-point roundoff:

```python
expanded = kw.normalize_polynomial_symbolic("(x + 1)^3")
str(expanded)  # "1 + 3*x + 3*x^2 + x^3"
```

`normalize_rational_symbolic()` puts a univariate rational function over a
primitive polynomial denominator, makes the denominator's leading coefficient
positive, and cancels the exact polynomial greatest common divisor. It returns
a `RewriteResult` because cancellation can change the visible expression's
domain. Every excluded point from the source is retained as a structural
condition, including removable holes:

```python
reduced = kw.normalize_rational_symbolic("(x^2 - 1)/(x - 1)")
str(reduced.expression)  # "1 + x"
reduced.conditions       # includes "x - 1 != 0"
reduced.applications[0].rule  # "normalize-rational"
```

The same operation is available as the opt-in `normalize-rational` rewrite:

```python
combined = kw.rewrite_symbolic(
    "1/x + 1/(x + 1)", rules="normalize-rational"
)
str(combined.expression)  # "(x + x^2)^-1*(1 + 2*x)"
combined.conditions       # ("x + 1 != 0", "x != 0")
```

It is deliberately not part of the default rewrite set. Expansion can enlarge
an expression, and users relying on the compact factored tree should not see a
silent representation change. Both normalization functions are currently
limited to one variable with exact numeric coefficients. Pass `variable=` for
an explicit variable name and use `max_degree=` to bound polynomial expansion;
multivariate polynomials, symbolic coefficients, and non-rational functions are
rejected rather than partially transformed. Applying either normalizer twice
is a fixed point.

### Guarded rewrite rules

`rewrite_symbolic()` performs optional algebraic transformations separately
from basic canonicalization. Every accepted application is checked against an
`AssumptionSet`, preserves restrictions from the original expression, and is
recorded with its rule identifier, local tree path, before/after expressions,
required conditions, newly introduced conditions, and explanation.
The engine also derives restrictions from the complete candidate tree, so a
custom replacement cannot silently introduce a reciprocal, logarithm, radical,
or other partial expression on a larger domain.

```python
cancelled = kw.rewrite_symbolic("x/x")
str(cancelled.expression)       # "1"
cancelled.conditions            # ("x != 0",)
cancelled.applications[0].rule  # "cancel-reciprocal"

inverse = kw.rewrite_symbolic("exp(ln(x))")
str(inverse.expression)         # "x"
inverse.conditions              # ("x > 0",)

principal = kw.rewrite_symbolic("sqrt(x^2)")
str(principal.expression)       # "abs(x)"
```

The initial built-in rules are:

| Identifier | Rewrite | Safety condition |
|---|---|---|
| `cancel-reciprocal` | `u * u^-1 -> 1` | Retains `u != 0` and the domain of `u` |
| `exp-log-inverse` | `exp(ln(u)) -> u` | `u > 0` over the reals; `u != 0` over the complexes |
| `sqrt-square` | `sqrt(u^2) -> abs(u)` | Real domain only |
| `normalize-rational` | Exact univariate rational normalization | Opt-in; retains the complete source domain and every cancelled-factor exclusion |

Use `available_rewrite_rules()` to inspect their stable identifiers. Passing
`rules=()` disables them; passing identifiers or custom `RewriteRule` objects
selects an ordered rule set. `normalize-rational` appears in the registry but
is excluded from the default set. Rule order and tree traversal are
deterministic.

A custom rule has a transform and, optionally, a guard. The transform returns
`None` when it does not match. The guard returns a boolean, a `Condition`, an
`AssumptionSet`, or an iterable of conditions:

```python
def positive_square_root(expression, context):
    if (isinstance(expression, kw.SymbolicFunction)
            and expression.name == "sqrt"):
        argument = expression.arguments[0]
        if isinstance(argument, kw.Power) and argument.exponent == kw.ExactNumber(2):
            return argument.base

def nonnegative(before, after, context):
    return kw.RelationCondition(after, ">=", kw.ExactNumber(0))

rule = kw.RewriteRule("positive-square-root", positive_square_root, nonnegative)
safe = kw.rewrite_symbolic("sqrt(x^2)", rule, assumptions="x >= 0")
```

Guards that are refuted are never applied. Unknown guards are skipped by
default. `introduce_conditions=True` explicitly permits the rewrite and adds
the undecided predicates to the result, making conditional narrowing visible.
Rewriting stops at a fixed point and reports `step_limit`, `node_limit`, or
`cycle` when a safety boundary intervenes. `RewriteResult` and
`RewriteApplication` both support structural, JSON-safe serialization.

#### Solver transformation pipeline

The unified scalar solver now applies its symbolic transformations through an
ordered guarded-rule registry rather than a monolithic dispatch block. The
registry covers denominator clearing, bounded trigonometric reduction,
algebraic substitution, exact and symbolic-coefficient polynomial solving,
zero products, compatible logarithm combination, exponential and logarithm
inversion, absolute-value splitting, repeated radical isolation/squaring, and
periodic trigonometric inversion.

Each rule receives the same immutable variable, domain, interval, and
`AssumptionSet` context. A rule either declines the equation, produces another
structural `EquationState`, or terminates with a `SolutionSet`. Guards are
proved, refuted, or retained as structural conditions before the next rule is
allowed to run. Equation-to-equation applications and terminal applications
are both emitted as `SolutionStep` records using the established rule names:

```python
result = kw.solve_equation("sqrt(x + 1) = x - 1", steps=True)
[(step.rule, type(step.after).__name__) for step in result.steps]
# [('normalize', 'EquationState'),
#  ('isolate_radical', 'EquationState'),
#  ('solve_polynomial', 'FiniteSolutionSet')]
```

The rule order is deterministic and bounded to prevent a future collection of
transformations from looping indefinitely. Real-only inversion rules are
explicitly marked as such, so the complex solver continues to return
`unresolved` instead of presenting a principal transcendental branch as a
complete answer. Candidate verification against the original equation remains
the final authority, including after denominator clearing or squaring.

This registry is currently an internal solver extension point. The public
`RewriteRule` API remains expression-oriented, and the existing
`solve_equation()` signature and solution containers are unchanged.

#### Algebraic substitutions

The `algebraic_substitution` solver rule recognizes an equation that is an
exact polynomial or rational function of a repeated inner expression. It
introduces a fresh internal variable, solves the outer polynomial exactly, and
then sends every equation of the form `inner = outer_root` back through the
guarded solver rules. All outer branches must be solved completely; otherwise
the substitution is declined and the result remains `unresolved`.

Supported substitution families include:

- sparse powers whose positive exponents have a common divisor, such as
  `x^4 - 5*x^2 + 4 = 0` or `x^6 - 5*x^3 + 6 = 0`;
- repeated nonlinear algebraic expressions, including `x^2 + 1` and
  rational expressions such as `x + 1/x`;
- repeated `abs`, `sqrt`, `exp`, `ln`/`log`, and trigonometric calls;
- compatible exponential families such as `exp(2*x)` with `exp(x)`,
  `2^(2*x)` with `2^x`, and exact related bases such as `4^x` with `2^x`.

```python
polynomial = kw.solve_equation("x^4 - 5*x^2 + 4 = 0", steps=True)
polynomial.solutions  # (-2, -1, 1, 2)
polynomial.steps[-1].rule  # "algebraic_substitution"

exponential = kw.solve_equation("exp(2*x) - 5*exp(x) + 6 = 0")
# x = ln(2), ln(3)

periodic = kw.solve_equation("sin(x)^2 - sin(x) = 0")
# sin(x)=0 and sin(x)=1, returned as complete integer families
```

Exact multiplicities are multiplied through nested finite branches. Original
domain restrictions remain active, so a substitution involving `x + 1/x`
retains `x != 0`, invalid negative roots for `exp`/`sqrt` are discarded, and
trigonometric outer roots outside `[-1, 1]` produce no real branch. Intervals
are applied to the final target-variable solutions, not to the temporary outer
variable.

Substitution expansion is bounded to outer degree 32, recursive inner solving
is limited to 16 transformations, and at most 128 resulting branches are
accepted. Symbolic outer coefficients, incompatible exponential bases or
affine arguments, general functional decomposition, and complex
transcendental branch inversion remain unsupported. Sparse polynomial-power
substitution is available in real and complex domains; transcendental
substitution is intentionally real-only until complete complex branch families
are represented.

### Assumptions and conditions

Conditions are represented structurally rather than inferred from display
strings. The public predicate model includes `RelationCondition`,
`BetweenCondition`, `DefinedCondition`, `CompoundCondition`, and
`TruthCondition`; `AssumptionSet` is an immutable conjunction of predicates.
It supports substitution, three-valued evaluation (`True`, `False`, or `None`
when parameters remain unknown), basic bound implication, contradiction
detection, deterministic serialization, and legacy string rendering.

The frozen `solve_equation()` signature is unchanged. Use the additive
`solve_equation_assuming()` entry point when caller knowledge should participate
in branch selection and candidate validation:

```python
positive = kw.solve_equation_assuming("x^2 = 1", "x > 0")
positive.solutions                 # (ExactNumber(1),)
positive.conditions                # ("x > 0",) -- compatibility view
positive.assumptions.evaluate({"x": 2})  # True -- structural view

parameterized = kw.solve_equation_assuming(
    "a*x + b = 0", ("a != 0",), variable="x"
)
# The nonzero-coefficient branch is selected directly.

fixed_parameter = kw.solve_equation_assuming(
    "a*x = 4", {"a": 2}, variable="x"
)
fixed_parameter.solutions          # (ExactNumber(2),)
```

Atomic strings accepted by `parse_condition()` use `=`, `!=`, `<`, `<=`, `>`,
or `>=`; chained intervals such as `0 <= x <= 1` and explicit definedness
predicates are also supported. Compose boolean predicates explicitly with
`CompoundCondition` so precedence is unambiguous. Older serialized results
containing only condition strings remain readable and are upgraded to
structural predicates when possible.

#### Sign and domain inference

`AssumptionSet.infer_sign(expression)` performs conservative structural sign
propagation through exact constants, sums, products, integer and rational
powers, absolute values, square roots, exponentials, logarithms, intervals,
and equality substitutions. It returns one of `positive`, `negative`, `zero`,
`nonnegative`, `nonpositive`, `nonzero`, `unknown`, or `undefined`:

```python
facts = kw.AssumptionSet(("x > 0", "y <= 0", "y != 0"))
facts.infer_sign("x*y")       # "negative"
facts.infer_sign("x^2")      # "positive"
facts.infer_sign("x + 2")    # "positive"

interval = kw.AssumptionSet((kw.parse_condition("0 <= t <= 1"),))
interval.infer_sign("t")     # "nonnegative"
```

Signs describe values on the expression's valid domain. Consequently,
`infer_sign("x^-2")` is `positive`, but the assumption set only entails
`x^-2 > 0` after it can also prove `x != 0`. Composite sign results strengthen
`entails()` and `refutes()`, allowing facts such as `x > 0` and `y < 0` to
prove `x*y < 0` without numerical sampling.

`is_defined(expression, domain)` returns `True`, `False`, or `None`, where
`None` means the current facts cannot decide. `infer_domain(expression)` uses
that evidence to return `real`, `complex`, `undefined`, or `unknown`:

```python
positive = kw.AssumptionSet(("x > 0",))
positive.is_defined("ln(x)", "real")  # True
positive.infer_domain("sqrt(x)")      # "real"

negative = kw.AssumptionSet(("x < 0",))
negative.is_defined("sqrt(x)", "real")  # False
negative.infer_domain("sqrt(x)")         # "complex"

kw.AssumptionSet().is_defined("1/x", "real")  # None
```

Inference is deliberately incomplete. It does not attempt arbitrary
inequality solving, interval arithmetic with correlated expressions, or proof
of transcendental identities. An inconclusive result stays `unknown`/`None`
instead of being guessed.

Equation-to-equation derivation stages store immutable `EquationState` objects,
not presentation-only strings. `step.equivalent_at(values)` can replay a stage
at an admissible assignment; terminal solution-set and system-summary stages
remain explicitly typed or textual as appropriate.

With `method="symbolic"`, an unsupported family returns a structured unresolved
result. With `method="numeric"`, or automatic fallback after symbolic rules are
exhausted, a finite `interval=(lower, upper)` is required. Numerical isolation
is adaptive and discontinuity-aware. It reports approximate answers with
`complete=False`: for arbitrary functions, a finite sampled search cannot prove
that every root was found.

#### Adaptive numerical isolation

The bounded scalar fallback starts with a small deterministic seed mesh and
caches every function evaluation. It recursively subdivides intervals showing
curvature, a local residual valley, a change in slope direction, oscillation,
or a transition between defined and undefined values. Smooth regions are not
uniformly resampled, so ordinary crossing equations generally use fewer
evaluations than the former fixed grid:

```python
root = kw.solve_equation(
    "cos(x) = x", method="numeric", interval=(0, 1), steps=True
)
root.solutions     # approximately (0.7390851332,)
root.evaluations   # deterministic; typically well below 257
root.complete      # False
```

Two independent evidence paths are used:

- a finite sign-change bracket followed by safeguarded secant/bisection
  refinement for crossing roots;
- a sampled local minimum of `abs(left - right)` followed by bounded
  golden-section refinement for tangent and even-multiplicity roots.

Root candidates retain their evidence intervals. Deduplication merges
overlapping evidence for the same refined root, but does not merge nearby roots
solely because their numeric values are close. A sign change across a pole is
not sufficient: refinement must reach a small finite residual, and every final
candidate is still checked against the original equation and its structural
domain assumptions. One-sided secant evidence handles floating-point endpoint
roots without treating a merely small asymptotic tail as a zero.

`max_iterations` bounds each refinement and also derives a hard overall
evaluation budget; recursion depth and the number of evaluations are capped
independently. `EquationSolution.evaluations` reports the actual calls, while
residuals are normalized during final candidate verification. Adaptive search
can still miss arbitrarily narrow, highly oscillatory, clustered, or
pathological roots that do not leave evidence in the sampled intervals. This is
why numerical results remain explicitly approximate and incomplete.

`solve_equation_system()` performs exact rational row reduction for square or
rectangular linear systems, including parametric underdetermined results. It
also has an initial exact nonlinear-system engine. The engine first performs
finite triangular and affine substitution. This includes equations such as
`x - y = 1`, equations affine in one variable such as `x + y^2 = 3`, and safe
rational eliminations such as `x*y = 2`, where the nonzero right-hand side
proves that division cannot lose a zero-coefficient branch.

When direct substitution cannot triangularize a system, two-variable exact
polynomial systems can use a Sylvester resultant. The retained univariate
polynomial is solved by the scalar exact engine, the eliminated variable is
recovered on every branch, and every candidate is checked against every
original equation. Empty complete branches are reported as `inconsistent`,
not `unresolved`.

```python
circle_line = kw.solve_equation_system((
    "x^2 + y^2 = 5",
    "x - y = 1",
), steps=True)

coupled = kw.solve_equation_system((
    "x^2 + y = 3",
    "y^2 + x = 3",
), steps=True)

circle_line.complete       # True
coupled.steps[-1].rule      # "polynomial_resultant"
```

This first exact phase deliberately targets finite, zero-dimensional systems.
Positive-dimensional nonlinear sets, general parametric case splitting,
systems with more than two coupled variables, transcendental elimination, and
general Groebner-basis solving remain `unresolved`. To keep exact expansion
bounded, multivariate recognition is limited to total degree 8 and 256 terms;
the resultant's Sylvester matrix is limited to 8 rows. A declined exact path
never returns a partial solution list.

For unsupported nonlinear systems, set `numeric_fallback=True` and provide
`initial` values to use damped Newton. The fallback is real-only, local, and
approximate. It rejects non-real or non-finite residual evaluations instead of
silently discarding imaginary components. `steps=True` records normalization,
exact row reduction, triangular/affine substitution, polynomial resultant, or
local Newton as applicable.

The native expression grammar uses multi-character identifiers (`xy` is one
symbol). This intentionally differs from the legacy polynomial parser, where
concatenated one-letter names historically mean multiplication. Conversion back
to a legacy expression is therefore limited to trees the legacy model can
preserve; functions, constants such as `pi`, negative powers, and other unsafe
trees raise `UnsupportedExpressionError`.
