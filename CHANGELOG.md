# Changelog

## Unreleased

- Conditional solution sets now have native human-readable formatting, including
  condition conjunctions, one-branch-per-line unions, and parenthesized nested
  unions.
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
- Formalized the native basic canonical-form contract and added an executable
  `is_canonical_symbolic()` fixed-point check. Canonicalization is deterministic,
  exact, immutable, idempotent, and domain-safe, while expansion and
  factorization remain explicit non-goals.
- Fixed domain-sensitive zero-power canonicalization and nested guarded-zero
  products found by randomized idempotence testing.
- Added deterministic guarded rewrite infrastructure with auditable rule
  applications, three-valued assumption guards, automatic domain preservation,
  cycle/node/step limits, structural serialization, and initial reciprocal,
  exponential/logarithm, and real square-root rules.
- Simplified real-domain conditions for positive even integer powers so
  universally nonnegative squares do not create spurious assumptions.
- Added conservative sign and domain inference to `AssumptionSet`, including
  structural propagation, interval and substitution knowledge, three-valued
  definedness, composite-relation entailment, and domain-aware guard proofs.
- Added exact `normalize_polynomial_symbolic()` and guarded
  `normalize_rational_symbolic()` APIs for bounded univariate normalization,
  primitive coefficient scaling, polynomial-GCD cancellation, and deterministic
  fixed points.
- Added the opt-in `normalize-rational` rewrite rule. It preserves all source
  domain restrictions and explicitly records exclusions introduced by removed
  factors; exact polynomial nonzero assumptions can now prove nonzero factors.
- Refactored unified-solver transformations onto a deterministic guarded-rule
  pipeline. Denominator clearing, polynomial and symbolic-linear solving,
  logarithmic/exponential/radical inversion, absolute-value branching, and
  periodic trigonometric solving now share structural contexts, guard handling,
  resource bounds, and auditable `SolutionStep` generation.
- Marked real-only solver rules explicitly and retained final verification
  against the original equation, preserving complex completeness guarantees and
  rejection of roots introduced by denominator clearing or squaring.
- Replaced fixed-grid scalar numerical fallback with cached adaptive isolation.
  The isolator refines curvature, residual valleys, oscillation, and domain
  boundaries; uses safeguarded sign-change refinement for crossing roots and
  bounded local minimization for tangent/even roots; and enforces depth and
  evaluation budgets.
- Root deduplication now requires overlapping isolation evidence rather than
  distance alone. Pole crossings and merely small asymptotic tails are rejected,
  while nearby roots and floating-point roots at interval endpoints are retained.
- Added guarded algebraic substitution for sparse polynomial powers, repeated
  nonlinear and rational expressions, repeated elementary functions, compatible
  exponential arguments, and exact related bases such as `4^x` and `2^x`.
- Substitution now solves every outer root through the existing rule pipeline,
  propagates finite-root multiplicities and domain restrictions, applies
  intervals only to final target-variable branches, and declines incomplete
  inner solves. Degree, transformation, and branch limits bound expansion.
- Began exact nonlinear-system solving with complete triangular and affine
  substitution, safe rational elimination, and bounded bivariate polynomial
  resultants. Every resultant candidate is verified in the original system;
  unsupported or resource-limited systems remain explicitly unresolved.
- Added parameter-aware symbolic quadratic solving with complete leading-
  coefficient and discriminant cases, exact degree reduction, zero-product
  solving for factored low-degree forms, and symbolic biquadratic substitution.
- Added `solve_inequality()` for exact real univariate polynomial and rational
  inequalities. Its multiplicity-aware sign chart preserves poles and cancelled
  denominator holes and returns serializable bounded or unbounded intervals.
- Extended `solve_inequality()` with guarded exact reductions for affine-wrapped
  absolute values, principal square roots, and positive or negative rational
  powers. The solver preserves radicand and zero exclusions, applies parity and
  monotonicity rules, supports compatible same-family comparisons, and fails
  closed for unsupported compound forms.
- Added complete parameterized linear and quadratic inequalities, including
  coefficient-sign, discriminant, lower-degree, identity, and contradiction
  branches. Parameter-scaled rational inequalities now reverse their relation
  by scalar sign and preserve denominator holes when the scalar is zero.
- Extended guarded radical solving to repeated additive and nested square roots
  through bounded expansion and original-equation candidate verification.
- Added real Pythagorean, double-angle, and triple-angle reductions that feed
  exact zero-product, algebraic-substitution, and periodic-family solving.

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
