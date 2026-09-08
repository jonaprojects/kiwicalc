import random

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


def _relative_residual(coefficients, root):
    degree = len(coefficients) - 1
    scale = sum(
        abs(coefficient) * max(1.0, abs(root)) ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    return abs(np.polyval(coefficients, root)) / max(1.0, scale)


@pytest.mark.parametrize(
    ("solver", "coefficients", "expected"),
    [
        (kw.solve_cubic, (1, -3, 3, -1), [1]),
        (kw.solve_cubic, (1, 0, 0, 0), [0]),
        (kw.solve_quartic, (1, -4, 6, -4, 1), [1]),
        (kw.solve_quartic, (1, 0, 0, 0, 0), [0]),
    ],
)
def test_repeated_roots_are_correct_and_consolidated(solver, coefficients, expected):
    roots = solver(*coefficients)
    assert roots == pytest.approx(expected)
    assert all(_relative_residual(coefficients, root) < 1e-12 for root in roots)


def test_fixed_degree_solvers_randomized_residual_oracle():
    rng = random.Random(20260906)
    for degree, solver in ((3, kw.solve_cubic), (4, kw.solve_quartic)):
        for _ in range(250):
            coefficients = [rng.randint(-8, 8) for _ in range(degree + 1)]
            coefficients[0] = coefficients[0] or 1
            roots = solver(*coefficients)
            assert roots
            assert all(_relative_residual(coefficients, root) < 1e-10 for root in roots)


def test_quadratic_avoids_catastrophic_cancellation():
    coefficients = (1.0, 1e16, 1.0)
    small, large = kw.solve_quadratic(*coefficients)
    assert small == pytest.approx(-1e-16)
    assert large == pytest.approx(-1e16)
    assert _relative_residual(coefficients, small) < 1e-15


@pytest.mark.parametrize(
    ("equation_type", "source", "expected_coefficients"),
    [
        (kw.QuadraticEquation, "t^2=1", [1, 0, -1]),
        (kw.CubicEquation, "y^3-1=0", [1, 0, 0, -1]),
        (kw.QuarticEquation, "z^4=z^2", [1, 0, -1, 0, 0]),
    ],
)
def test_fixed_degree_objects_normalize_both_sides_and_variable_names(
    equation_type, source, expected_coefficients
):
    equation = equation_type(source)
    assert equation.coefficients() == expected_coefficients
    assert all(_relative_residual(expected_coefficients, root) < 1e-10 for root in equation.solve())
    assert isinstance(reversed(equation), equation_type)


def test_fixed_degree_objects_accept_matching_explicit_variables():
    equation = kw.QuadraticEquation("x^2=1", variables=("x",))
    assert equation.coefficients() == [1, 0, -1]
    assert equation.solve() == pytest.approx((1, -1))
    with pytest.raises(ValueError, match="match"):
        kw.QuadraticEquation("x^2=1", variables=("y",))


def test_linear_identities_and_negative_inequality_direction():
    assert kw.solve_linear("2=2") == np.inf
    assert kw.solve_linear("2=3") is None
    assert kw.solve_linear_inequality("-2x<4") == "x>-2"
    assert kw.solve_linear_inequality("-2x>=4") == "x<=-2"


def test_polynomial_factoring_has_no_debug_output(capsys):
    kw.solve_poly_by_factoring([1, -6, 11, -6])
    assert capsys.readouterr().out == ""


def test_random_polynomial2_has_every_power_and_valid_python_syntax():
    random.seed(10)
    expression = kw.random_polynomial2(5, values=(1, 3))
    assert all(token in expression for token in ("x^5", "x^4", "x^3", "x^2", "x"))
    python_expression = kw.random_polynomial2(5, values=(1, 3), python_syntax=True)
    compile(python_expression, "<random polynomial>", "eval")


def test_random_polynomial_rejects_impossible_leading_coefficient():
    with pytest.raises(ValueError, match="non-zero"):
        kw.random_polynomial(3, solutions_range=(0, 0))
    with pytest.raises(ValueError, match="non-zero"):
        kw.random_polynomial2(3, values=(0, 0))


def test_random_linear_systems_are_always_solvable():
    for seed in range(40):
        random.seed(seed)
        equations, values = kw.random_linear_system(("x", "y", "z"), get_solutions=True)
        result = kw.solve_linear_system(equations, variables=("x", "y", "z"))
        assert result == pytest.approx(dict(zip(("x", "y", "z"), values)), abs=1e-8)


def test_linear_system_matrix_split_and_simplification():
    system = kw.LinearSystem(("x+y=3", "x-y=1"), variables=("x", "y"))
    matrix, vector = system.to_matrix_and_vector()
    assert np.linalg.solve(matrix, vector) == pytest.approx([2, 1])
    assert system.simplify() is None
    assert system.get_solutions() == pytest.approx({"x": 2, "y": 1})


def test_polynomial_system_accepts_generator_and_can_show_steps(capsys):
    equations = (equation for equation in ("x+y=3", "x-y=1"))
    result = kw.solve_poly_system(equations, initial_vals={"x": 0, "y": 0}, show_steps=True)
    assert result == pytest.approx({"x": 2, "y": 1})
    assert "residual=" in capsys.readouterr().out


def test_polynomial_equation_can_plot_a_constant_side():
    equation = kw.PolyEquation("x^2=1")
    assert equation.plot_solutions(show=False, step=0.5) is None
    plt.close("all")


@pytest.mark.parametrize("source", ("x+1", "=1", "x=", "x=1=2"))
def test_equation_objects_reject_malformed_equations(source):
    with pytest.raises((ValueError, TypeError)):
        kw.LinearEquation(source)


def test_public_star_export_contains_legacy_equation_helpers():
    assert "solve_quadratic_from_str" in kw.__all__
    assert "solve_linear_inequality" in kw.__all__
