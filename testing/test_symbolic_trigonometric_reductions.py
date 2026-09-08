import math

import pytest

import kiwicalc as kw


def numeric(value):
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value).real


def bounded_values(equation, lower=-2 * math.pi, upper=2 * math.pi):
    result = kw.solve_equation(
        equation, method="symbolic", interval=(lower, upper), steps=True,
    )
    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    return result, sorted(numeric(value) for value in result.solutions)


def test_pythagorean_reduction_solves_mixed_sine_cosine_polynomial():
    result, values = bounded_values("sin(x)^2+cos(x)=1")

    assert result.exact and result.complete
    assert len(values) == 7
    assert values == pytest.approx([
        -2 * math.pi, -3 * math.pi / 2, -math.pi / 2, 0,
        math.pi / 2, 3 * math.pi / 2, 2 * math.pi,
    ])
    assert result.steps[1].rule == "trigonometric_reduction"


def test_pythagorean_identity_and_contradiction_are_recognized_exactly():
    identity = kw.solve_equation(
        "sin(x)^2+cos(x)^2=1", method="symbolic", steps=True,
    )
    contradiction = kw.solve_equation(
        "sin(x)^2+cos(x)^2=2", method="symbolic",
    )

    assert isinstance(identity.solution_set, kw.UniversalSolutionSet)
    assert isinstance(contradiction.solution_set, kw.EmptySolutionSet)
    assert identity.exact and identity.complete
    assert identity.steps[1].rule == "trigonometric_reduction"


def test_double_angle_sine_reduction_factors_complete_branches():
    result, values = bounded_values("sin(2*x)=cos(x)")

    assert len(values) == 8
    assert all(abs(math.sin(2 * value) - math.cos(value)) < 1e-10 for value in values)
    assert [step.rule for step in result.steps] == [
        "normalize", "trigonometric_reduction", "solve_zero_product",
    ]


def test_double_angle_cosine_reduces_to_one_trig_kernel():
    result, values = bounded_values("cos(2*x)=sin(x)")

    assert len(values) == 6
    assert all(abs(math.cos(2 * value) - math.sin(value)) < 1e-10 for value in values)
    assert result.steps[-1].rule == "algebraic_substitution"


@pytest.mark.parametrize(
    "equation, count",
    (("sin(3*x)=sin(x)", 13), ("cos(3*x)=cos(x)", 9)),
)
def test_triple_angle_reductions_return_every_bounded_solution(equation, count):
    result, values = bounded_values(equation)

    assert len(values) == count
    assert result.exact and result.complete
    function = math.sin if equation.startswith("sin") else math.cos
    assert all(abs(function(3 * value) - function(value)) < 1e-10 for value in values)


def test_exact_radical_special_angle_stays_compact():
    result, values = bounded_values("sin(x)=sqrt(2)/2")

    assert values == pytest.approx([
        -7 * math.pi / 4, -5 * math.pi / 4,
        math.pi / 4, 3 * math.pi / 4,
    ])
    assert "asin" not in str(result.solution_set)


def test_unsupported_higher_multiple_angle_remains_unresolved():
    result = kw.solve_equation("cos(4*x)=cos(x)", method="symbolic")

    assert result.status == "unresolved"
    assert not result.complete


def test_trigonometric_reduction_is_real_only():
    result = kw.solve_equation(
        "sin(x)^2+cos(x)^2=1", method="symbolic", domain="complex",
    )

    assert result.status == "unresolved"


def test_trigonometric_reduction_is_deterministic_and_serializable():
    equation = "sin(2*x)=cos(x)"
    first = kw.solve_equation(equation, method="symbolic", steps=True)
    second = kw.solve_equation(equation, method="symbolic", steps=True)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert kw.EquationSolution.from_dict(first.to_dict()) == first
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(first)) == first
