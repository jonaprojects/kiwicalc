import json

import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ("equation", "condition"),
    (
        ("0/x=0", "x != 0"),
        ("1/(x-1)=1/(x-1)", "x - 1 != 0"),
        ("sqrt(x)=sqrt(x)", "x >= 0"),
        ("ln(x)=ln(x)", "x > 0"),
        ("0*ln(x)=0", "x > 0"),
    ),
)
def test_identity_preserves_original_expression_domain(equation, condition):
    result = kw.solve_equation(equation, variable="x")
    assert isinstance(result.solution_set, kw.UniversalSolutionSet)
    assert condition in result.conditions
    assert result.complete


@pytest.mark.parametrize("equation", ("x-x=0", "0*x=0"))
def test_variable_inference_precedes_destructive_simplification(equation):
    result = kw.solve_equation(equation)
    assert result.variable == "x"
    assert isinstance(result.solution_set, kw.UniversalSolutionSet)


def test_exp_and_general_exact_trigonometric_values_are_supported():
    exponential = kw.solve_equation("exp(x)=2")
    assert exponential.complete and str(exponential.solution_set) == "{ln(2)}"
    assert isinstance(kw.solve_equation("sin(x)=2").solution_set, kw.EmptySolutionSet)
    trigonometric = kw.solve_equation("sin(x)=sqrt(2)/2")
    assert trigonometric.complete
    assert isinstance(trigonometric.solution_set, kw.UnionSolutionSet)


def test_complex_transcendental_never_claims_principal_branch_is_complete():
    result = kw.solve_equation("2^x=-1", variable="x", domain="complex")
    assert result.status == "unresolved"
    assert not result.complete
    with pytest.raises(ValueError, match="real domain"):
        kw.solve_equation("x^2+1=0", domain="complex", interval=(-2, 2))


def test_logarithm_base_conditions_are_retained():
    result = kw.solve_equation("log(a,x)=2", variable="x")
    assert {"a > 0", "a != 1", "x > 0"}.issubset(result.conditions)


@pytest.mark.parametrize("source", ("2 3", "x 2"))
def test_adjacent_numbers_require_an_operator(source):
    with pytest.raises(kw.EquationParseError):
        kw.parse_symbolic(source)


@pytest.mark.parametrize("source", ("-" * 2000 + "x", "^".join(["x"] * 1500)))
def test_every_recursive_parser_path_honors_depth_limit(source):
    with pytest.raises(kw.UnsupportedExpressionError, match="nesting limit"):
        kw.parse_symbolic(source)


@pytest.mark.parametrize("source", ("sqrt(x)", "sin(x)", "ln(x)", "pi*x"))
def test_legacy_adapter_rejects_semantically_unsupported_trees(source):
    with pytest.raises(kw.UnsupportedExpressionError):
        kw.to_legacy_expression(kw.parse_symbolic(source))


def test_intervals_apply_to_universal_and_conditional_solutions():
    identity = kw.solve_equation("x=x", interval=(0, 1))
    assert isinstance(identity.solution_set, kw.IntervalSolutionSet)
    assert (identity.solution_set.lower, identity.solution_set.upper) == (0, 1)
    conditional = kw.solve_equation("a*x+b=0", variable="x", interval=(0, 1))
    assert "<= -1*a^-1*b <=" in str(conditional.solution_set)


def test_periodic_parameter_never_shadows_target_variable():
    family = kw.solve_equation("sin(n)=0").solution_set
    assert isinstance(family, kw.ParametricSolutionSet)
    assert family.variable == "n"
    assert family.parameter != family.variable


def test_rootof_result_is_strict_json_serializable():
    result = kw.solve_equation("x^5-x+1=0", domain="complex")
    encoded = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)
    assert json.loads(encoded)["solution_set"]["type"] == "finite"
    legacy = result.to_dict()
    legacy["residuals"] = [float("nan")] * len(result.solution_set.values)
    assert all(value is None for value in kw.EquationSolution.from_dict(legacy).residuals)


def test_real_rootof_values_use_certified_disjoint_intervals():
    result = kw.solve_equation("x^3-3*x+1=0")
    assert result.complete and len(result.solution_set.values) == 3
    intervals = [value.interval for value in result.solution_set.values]
    assert all(interval is not None for interval in intervals)
    assert all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:]))
    for value in result.solution_set.values:
        approximation = value.evaluate({})
        assert abs(approximation**3 - 3 * approximation + 1) < 1e-12


def test_real_rootof_classification_does_not_use_imaginary_thresholds():
    real = kw.solve_equation("x^5-x+1=0")
    complex_result = kw.solve_equation("x^5-x+1=0", domain="complex")
    assert len(real.solution_set.values) == 1
    assert len(complex_result.solution_set.values) == 5


def test_system_results_are_deeply_immutable():
    result = kw.solve_equation_system(("x+y=2",))
    with pytest.raises(TypeError):
        result.solutions[0]["x"] = 99


@pytest.mark.parametrize(
    "kwargs",
    (
        {"tolerance": 0}, {"max_iterations": 0}, {"steps": "yes"},
        {"numeric_fallback": "yes"},
    ),
)
def test_system_solver_validates_controls_on_exact_paths(kwargs):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_equation_system(("x=1",), **kwargs)


def test_system_solver_rejects_string_as_equation_collection():
    with pytest.raises(TypeError, match="sequence"):
        kw.solve_equation_system("x=1")


def test_system_steps_are_recorded_for_symbolic_paths():
    result = kw.solve_equation_system(("x+y=2", "x-y=0"), steps=True)
    assert [step.rule for step in result.steps] == [
        "normalize_system", "rational_row_reduction",
    ]


def test_scalar_equation_steps_are_structural_and_replayable():
    result = kw.solve_equation("2*x+3=7", steps=True)
    normalize = result.steps[0]
    assert isinstance(normalize.before, kw.EquationState)
    assert isinstance(normalize.after, kw.EquationState)
    for value in (-10, 2, 9):
        assert normalize.equivalent_at({"x": value})
    restored = kw.EquationSolution.from_dict(result.to_dict())
    assert restored == result


def test_real_system_fallback_never_discards_imaginary_residuals():
    with pytest.raises(ValueError, match="not real-valued"):
        kw.solve_equation_system(
            ("sqrt(x)+x=-1",), variables=("x",),
            numeric_fallback=True, initial={"x": -1},
        )
