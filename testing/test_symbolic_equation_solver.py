import inspect
import math

import pytest

import kiwicalc as kw


def numeric(value):
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value)


def numeric_values(result):
    return sorted((numeric(value) for value in result.solution_set.values), key=lambda value: (value.real, value.imag))


def test_unified_solver_signature_and_exports():
    assert tuple(inspect.signature(kw.solve_equation).parameters) == (
        "equation", "variable", "domain", "interval", "method",
        "numeric_fallback", "tolerance", "max_iterations", "steps",
    )
    for name in ("solve_equation", "EquationSolution", "FiniteSolutionSet", "RootOf", "solve_equation_system"):
        assert name in kw.__all__


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1+2*3", 7),
        ("2(x+1)", 8),
        ("-x^2", -9),
        ("2^-2", 0.25),
        ("|x-5|", 2),
        ("sin(pi/6)", 0.5),
        ("cos(pi)", -1),
    ],
)
def test_symbolic_parser_precedence_functions_and_exact_values(source, expected):
    expression = kw.parse_symbolic(source)
    assert complex(expression.evaluate({"x": 3})).real == pytest.approx(expected)


def test_native_simplification_differentiation_and_legacy_adapters():
    expression = kw.to_symbolic(kw.poly_from_str("x^2+2x+1"))
    derivative = kw.differentiate_symbolic(expression, "x")
    assert derivative.evaluate({"x": 3}) == pytest.approx(8)
    assert kw.structurally_equal(kw.simplify_symbolic("2+1+x"), "3+x")
    assert str(kw.to_legacy_expression(expression)) == str(kw.poly_from_str("x^2+2x+1"))


@pytest.mark.parametrize("source", ("", "x+@", "foo(x)", "sin()", "(x+1"))
def test_symbolic_parser_rejects_invalid_or_unsupported_syntax(source):
    with pytest.raises((kw.EquationParseError, kw.UnsupportedExpressionError)):
        kw.parse_symbolic(source)


def test_parser_resource_limit_is_bounded():
    with pytest.raises(kw.UnsupportedExpressionError, match="limit"):
        kw.parse_symbolic("x+" * 5000 + "x")


def test_exact_linear_and_quadratic_results():
    linear = kw.solve_equation("2x+1=5")
    assert linear.exact and linear.complete and numeric_values(linear) == [2]
    quadratic = kw.solve_equation("x^2-2=0")
    assert numeric_values(quadratic) == pytest.approx([-math.sqrt(2), math.sqrt(2)])


def test_repeated_root_multiplicity_is_preserved():
    result = kw.solve_equation("(x-2)^2=0")
    assert numeric_values(result) == [2]
    assert result.solution_set.multiplicities == (2,)
    complex_repeated = kw.solve_equation("(x^2+1)^2=0", domain="complex")
    assert complex_repeated.solution_set.multiplicities == (2, 2)


def test_factorable_and_rootof_polynomials():
    factorable = kw.solve_equation("x^3-6x^2+11x-6=0")
    assert numeric_values(factorable) == [1, 2, 3]
    irreducible = kw.solve_equation("x^5-x+1=0", domain="complex")
    assert len(irreducible.solution_set.values) == 5
    assert all(isinstance(value, kw.RootOf) for value in irreducible.solution_set.values)


def test_complex_domain_and_real_domain_polynomial_behavior():
    assert isinstance(kw.solve_equation("x^2+1=0").solution_set, kw.EmptySolutionSet)
    roots = numeric_values(kw.solve_equation("x^2+1=0", domain="complex"))
    assert roots == pytest.approx([-1j, 1j])


def test_constant_equations_use_universal_and_empty_sets():
    assert isinstance(kw.solve_equation("1=1", variable="x").solution_set, kw.UniversalSolutionSet)
    assert isinstance(kw.solve_equation("1=2", variable="x").solution_set, kw.EmptySolutionSet)
    assert isinstance(kw.solve_equation("a=a", variable="x").solution_set, kw.UniversalSolutionSet)


def test_known_symbolic_coefficient_does_not_create_spurious_parameter_branch():
    result = kw.solve_equation("pi*x=1")
    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    assert result.conditions == ()
    assert numeric_values(result) == pytest.approx([1 / math.pi])


def test_rational_equation_retains_exclusions():
    result = kw.solve_equation("(x^2-1)/(x-1)=0")
    assert numeric_values(result) == [-1]
    assert result.conditions == ("x - 1 != 0",)


def test_radical_candidates_are_checked_in_original_equation():
    result = kw.solve_equation("sqrt(x+1)=x-1", steps=True)
    assert numeric_values(result) == [3]
    assert any(step.rule == "isolate_radical" for step in result.steps)


def test_absolute_value_branches_and_impossible_value():
    assert numeric_values(kw.solve_equation("abs(2x-1)=3")) == [-1, 2]
    assert isinstance(kw.solve_equation("abs(x)=-1").solution_set, kw.EmptySolutionSet)


def test_exponential_and_logarithmic_inversion():
    assert numeric_values(kw.solve_equation("2^x=8")) == [3]
    assert numeric_values(kw.solve_equation("2^(x+1)=2^4")) == [3]
    logarithmic = kw.solve_equation("ln(x)=2")
    assert numeric_values(logarithmic) == pytest.approx([math.e ** 2])
    assert logarithmic.conditions == ("x > 0",)
    combined = kw.solve_equation("ln(x)+ln(x-1)=ln(2)")
    assert numeric_values(combined) == [2]
    assert len(combined.conditions) == 3


def test_periodic_trigonometric_families_and_interval_filtering():
    family = kw.solve_equation("sin(x)=0").solution_set
    assert isinstance(family, kw.ParametricSolutionSet)
    assert family.parameter_domain == "integers"
    bounded = kw.solve_equation("sin(x)=0", interval=(-math.pi, math.pi))
    assert numeric_values(bounded) == pytest.approx([-math.pi, 0, math.pi])
    cosine = kw.solve_equation("cos(2x)=1/2")
    assert isinstance(cosine.solution_set, kw.UnionSolutionSet)


def test_symbolic_parameter_produces_conditional_branches():
    result = kw.solve_equation("a*x+b=0", variable="x")
    assert result.complete and isinstance(result.solution_set, kw.UnionSolutionSet)
    assert all(isinstance(branch, kw.ConditionalSolutionSet) for branch in result.solution_set.sets)


def test_unsupported_symbolic_equation_is_explicitly_unresolved():
    result = kw.solve_equation("cos(x)=x", method="symbolic")
    assert result.status == "unresolved" and not result.complete


def test_bounded_numeric_fallback_and_residual_reporting():
    result = kw.solve_equation("cos(x)=x", interval=(0, 1), tolerance=1e-11)
    assert result.method == "hybrid" and not result.exact and not result.complete
    assert result.solutions == pytest.approx((0.7390851332151607,))
    assert result.residuals[0] <= 1e-11 and result.evaluations > 0
    tangent = kw.solve_equation("(x-0.12345)^2=0", method="numeric", interval=(0, 1))
    assert tangent.solutions == pytest.approx((0.12345,), abs=1e-7)


def test_numeric_mode_requires_real_finite_interval():
    with pytest.raises(ValueError, match="finite interval"):
        kw.solve_equation("cos(x)=x", method="numeric")
    with pytest.raises(ValueError, match="real domain"):
        kw.solve_equation("cos(x)=x", method="numeric", domain="complex", interval=(0, 1))


def test_numerical_isolation_does_not_treat_a_pole_as_a_root():
    result = kw.solve_equation("1/x=0", method="numeric", interval=(-1, 1))
    assert isinstance(result.solution_set, kw.EmptySolutionSet)


def test_string_equation_object_and_expression_pair_agree():
    expected = numeric_values(kw.solve_equation("x^2=1"))
    assert numeric_values(kw.solve_equation(kw.PolyEquation("x^2=1"))) == expected
    assert numeric_values(kw.solve_equation((kw.Var("x") ** 2, 1))) == expected


def test_solution_serialization_round_trip_is_deterministic():
    result = kw.solve_equation("cos(2x)=1/2", steps=True)
    restored = kw.EquationSolution.from_dict(result.to_dict())
    assert restored == result
    assert restored.to_dict() == result.to_dict()
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(result)) == result
    expression = kw.parse_symbolic("sqrt(2)+x")
    assert object_from_dict(object_to_dict(expression)) == expression


def test_exact_linear_system_unique_underdetermined_and_inconsistent():
    unique = kw.solve_equation_system(("x+y=3", "x-y=1"))
    assert {name: numeric(value).real for name, value in unique.solutions[0].items()} == {"x": 2, "y": 1}
    underdetermined = kw.solve_equation_system(("x+y=2",))
    assert underdetermined.parameters == ("t0",) and underdetermined.complete
    inconsistent = kw.solve_equation_system(("x+y=1", "x+y=2"))
    assert inconsistent.status == "inconsistent" and inconsistent.solutions == ()


def test_nonlinear_system_is_unresolved_or_uses_explicit_local_fallback():
    exact = kw.solve_equation_system(("x^2=2", "y=x"))
    assert exact.method == "symbolic" and exact.complete and len(exact.solutions) == 2
    pairs = sorted((numeric(item["x"]).real, numeric(item["y"]).real) for item in exact.solutions)
    assert pairs == pytest.approx([(-math.sqrt(2), -math.sqrt(2)), (math.sqrt(2), math.sqrt(2))])
    inconsistent = kw.solve_equation_system(("x^2+y^2=1", "x*y=1"))
    assert inconsistent.status == "inconsistent" and inconsistent.complete
    result = kw.solve_equation_system(
        ("sin(x)+y=1", "2*x+y=1"), numeric_fallback=True,
        initial={"x": 0.2, "y": 0.6},
    )
    assert result.method == "numeric"
    assert result.solutions[0] == pytest.approx({"x": 0, "y": 1}, abs=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"domain": "integer"}, {"method": "guess"}, {"interval": (1, 1)},
        {"tolerance": 0}, {"max_iterations": 0}, {"steps": "yes"},
    ),
)
def test_unified_solver_validates_controls(kwargs):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_equation("x=1", **kwargs)
