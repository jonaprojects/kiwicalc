# Changelog

## Unreleased

### Equations and parsing

- Added the native `solve_equation()` structured API with exact rational
  arithmetic, solution-set objects, conditions, multiplicities, structured
  derivations, algebraic `RootOf` values, and bounded numerical fallback.
- Added native symbolic support for rational, radical, absolute-value,
  exponential, logarithmic, and canonical trigonometric equations.
- Added `solve_equation_system()` for exact rational linear systems and
  explicit local numerical fallback for nonlinear systems.

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
