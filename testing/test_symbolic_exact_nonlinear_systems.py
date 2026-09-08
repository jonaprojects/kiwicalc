import math

import pytest

import kiwicalc as kw


def numeric(value):
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value)


def numeric_solutions(result):
    return sorted(
        tuple(round(numeric(solution[name]).real, 10) for name in result.variables)
        for solution in result.solutions
    )


def test_affine_elimination_solves_line_circle_exactly_and_closes_assignments():
    result = kw.solve_equation_system(
        ("x^2+y^2=5", "x-y=1"), steps=True,
    )

    assert result.status == "solved"
    assert result.method == "symbolic"
    assert result.exact and result.complete
    assert numeric_solutions(result) == [(-1.0, -2.0), (2.0, 1.0)]
    assert all(not value.variables for solution in result.solutions for value in solution.values())
    assert result.steps[-1].rule == "triangular_substitution"


def test_safe_rational_affine_elimination_handles_nonconstant_coefficient():
    result = kw.solve_equation_system(("x^2+y^2=5", "x*y=2"))

    assert result.exact and result.complete
    assert numeric_solutions(result) == [
        (-2.0, -1.0), (-1.0, -2.0), (1.0, 2.0), (2.0, 1.0),
    ]


def test_coupled_quadratics_use_exact_bivariate_resultant():
    result = kw.solve_equation_system(
        ("x^2+y=3", "y^2+x=3"), steps=True,
    )

    assert result.exact and result.complete
    assert len(result.solutions) == 4
    assert (-1.0, 2.0) in numeric_solutions(result)
    assert (2.0, -1.0) in numeric_solutions(result)
    assert result.steps[-1].rule == "polynomial_resultant"
    assert "zero-dimensional" in result.message


def test_resultant_solves_system_without_an_affine_equation():
    result = kw.solve_equation_system(
        ("x^2+y^2=5", "x^2-y^2=3"), variables=("x", "y"),
    )

    assert result.exact and result.complete
    assert numeric_solutions(result) == [
        (-2.0, -1.0), (-2.0, 1.0), (2.0, -1.0), (2.0, 1.0),
    ]


@pytest.mark.parametrize(
    "equations, expected_rule",
    (
        (("x^2+y^2=1", "x*y=1"), "triangular_substitution"),
        (("x^2+y^2=0", "x^2+y^2=1"), "polynomial_resultant"),
    ),
)
def test_exact_nonlinear_inconsistency_is_distinguished_from_unresolved(
    equations, expected_rule,
):
    result = kw.solve_equation_system(equations, steps=True)

    assert result.status == "inconsistent"
    assert result.solutions == ()
    assert result.exact and result.complete
    assert result.steps[-1].rule == expected_rule


def test_resultant_candidates_are_checked_against_additional_equations():
    result = kw.solve_equation_system(
        ("x^2+y^2=5", "x^2-y^2=3", "x=2"),
    )

    assert result.exact and result.complete
    assert numeric_solutions(result) == [(2.0, -1.0), (2.0, 1.0)]


def test_complex_domain_keeps_complete_polynomial_branches():
    result = kw.solve_equation_system(
        ("x^2+y^2=0", "x^2-y^2=2"), domain="complex",
    )

    assert result.exact and result.complete
    assert len(result.solutions) == 4
    for solution in result.solutions:
        x, y = numeric(solution["x"]), numeric(solution["y"])
        assert abs(x * x + y * y) < 1e-8
        assert abs(x * x - y * y - 2) < 1e-8


def test_positive_dimensional_and_unsupported_systems_remain_unresolved():
    curve = kw.solve_equation_system(("x*y=0",), variables=("x", "y"))
    transcendental = kw.solve_equation_system(("sin(x)+y=0", "x^2+y^2=1"))

    assert curve.status == "unresolved" and not curve.complete
    assert transcendental.status == "unresolved" and not transcendental.complete
    assert "No supported complete exact" in curve.message


def test_resultant_result_is_deterministic_and_serializable():
    equations = ("x^2+y^2=5", "x^2-y^2=3")
    first = kw.solve_equation_system(equations, steps=True)
    second = kw.solve_equation_system(equations, steps=True)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert kw.EquationSystemSolution.from_dict(first.to_dict()) == first
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(first)) == first


def test_local_numeric_fallback_remains_available_when_exact_elimination_declines():
    result = kw.solve_equation_system(
        ("sin(x)+y=1", "2*x+y=1"), numeric_fallback=True,
        initial={"x": 0.2, "y": 0.6},
    )

    assert result.method == "numeric"
    assert result.solutions[0] == pytest.approx({"x": 0.0, "y": 1.0}, abs=1e-7)


def test_resultant_resource_limits_fail_closed_without_partial_answers():
    result = kw.solve_equation_system(("x^8+y^2=1", "x^8-y^2=0"))

    assert result.status == "unresolved"
    assert not result.exact and not result.complete
    assert result.solutions == ()
