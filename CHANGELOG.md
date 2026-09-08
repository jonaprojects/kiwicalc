# Changelog

## Unreleased

- Added an immutable structural assumptions engine for native equation solving,
  including relational, interval, definedness, truth, and compound predicates;
  contradiction detection; substitution; three-valued evaluation; basic bound
  implication; and deterministic serialization.
- Added `solve_equation_assuming()` for explicit parameter/domain knowledge while
  preserving the frozen `solve_equation()` signature and legacy rendered
  `EquationSolution.conditions` values.
- Native solver restrictions, conditional branches, interval guards, and
  derivation-step conditions now retain machine-readable predicates and use
  them to reject invalid candidates or resolve known branches.

### Equations and parsing

- Added the native `solve_equation()` structured API with exact rational
  arithmetic, solution-set objects, conditions, multiplicities, structured
  derivations, algebraic `RootOf` values, and bounded numerical fallback.
- Added native symbolic support for rational, radical, absolute-value,
  exponential, logarithmic, and canonical trigonometric equations.
- Added `solve_equation_system()` for exact rational linear systems and
  explicit local numerical fallback for nonlinear systems.
- Preserved denominator, logarithm, radical, trigonometric, and power-domain
  restrictions through cancellation and identity simplification.
- Added `exp` inversion, general affine inverse-trigonometric families, and
  exact radical special angles; impossible real trigonometric values now return
  an empty solution set.
- Applied finite intervals consistently to universal, finite, periodic, union,
  and conditional solution sets, and rejected real intervals in complex mode.
- Replaced tolerance-based real `RootOf` classification with exact Sturm
  isolation and made algebraic solution dictionaries strict-JSON compatible.
- Made native-to-legacy conversion fail safely when the legacy expression model
  cannot preserve semantics.
- Enforced parser depth on unary and power recursion, strengthened immutable
  result validation, and prevented real nonlinear-system fallback from
  discarding imaginary residual components.

- Preserved the 1.x equation APIs while stabilizing quadratic, cubic, quartic,
  polynomial, linear-system, and polynomial-system solving.
- Added coefficient scaling, residual verification, guarded root polishing,
  damped Newton steps, and scale-aware linear-system rank checks.
- Added a recursive polynomial parser supporting parentheses, implicit
  multiplication, scientific notation, unary signs, and explicit
  multi-character variables.
- Added `EquationParseError`, `UnsupportedExpressionError`, and
  `AmbiguousVariableError`, all compatible with existing `ValueError` handlers.
- Malformed equations, fractional or negative polynomial powers, unresolved
  variable names, and mixed terms unsupported by legacy coefficient dictionaries
  now fail explicitly instead of being partially parsed.
- Hardened random equation generators and invalidated cached solutions after
  equation mutation.
