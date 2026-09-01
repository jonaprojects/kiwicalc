import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


def test_linear_equation_object_protocol():
    equation = kw.LinearEquation("2x+1=7")
    assert equation.equation == "2x+1=7"
    assert equation.first_side == "2x+1"
    assert equation.second_side == "7"
    assert equation.num_of_variables == 1
    assert equation.solution == pytest.approx(3)
    assert str(equation) == "2x+1=7"
    assert "Equation" in repr(equation)
    assert reversed(equation).equation == "7=2x+1"
    assert equation.__copy__().solution == pytest.approx(3)
    assert "Final step" in equation.show_steps()


def test_linear_equation_simplify():
    equation = kw.LinearEquation("2x+3=x+7")
    equation.simplify()
    assert equation.equation.replace(" ", "") == "x=4"


def test_linear_equation_plot_solution_contract():
    assert kw.LinearEquation("2x+3=x+7").plot_solution(show=False, step=1) == pytest.approx((4, 11))


@pytest.mark.parametrize(
    ("equation_type", "source", "expected"),
    [
        (kw.QuadraticEquation, "x^2-5x+6=0", [2, 3]),
        (kw.CubicEquation, "x^3-6x^2+11x-6=0", [1, 2, 3]),
    ],
)
def test_polynomial_equation_objects(equation_type, source, expected):
    equation = equation_type(source)
    roots = equation.solve()
    assert sorted(round(complex(root).real, 6) for root in roots) == expected
    assert equation_type.__name__.replace("Equation", "") in repr(equation)


def test_quartic_equation_object_contract():
    equation = kw.QuarticEquation("x^4-5x^2+4=0")
    assert sorted(round(complex(root).real, 6) for root in equation.solve()) == [-2, -1, 1, 2]


def test_quadratic_simplified_string():
    equation = kw.QuadraticEquation("x^2-5x+6=0")
    assert equation.simplified_str() == "x^2-5x+6"
    with pytest.warns(UserWarning):
        assert kw.QuadraticEquation("x^2+y^2=0").solve() is None


def test_quadratic_real_mode_contract():
    assert kw.QuadraticEquation("x^2-5x+6=0").solve("real") == (3, 2)
    assert kw.solve_quadratic("x^2-5x+6=0") == pytest.approx((3, 2))


def test_poly_equation_object_and_copy():
    equation = kw.PolyEquation(kw.Poly("x^3-6x^2+11x-6"), 0)
    roots = equation.solve()
    assert sorted(round(complex(root).real, 6) for root in roots) == [1, 2, 3]
    assert equation.first_poly == kw.Poly("x^3-6x^2+11x-6")
    assert equation.second_poly.coefficients() == [0]
    assert equation.to_PolyExpr() == kw.Poly("x^3-6x^2+11x-6")
    assert equation.__copy__().to_PolyExpr() == equation.to_PolyExpr()


def test_random_equation_generators_return_solvable_contracts():
    linear, linear_solution = kw.random_linear(get_solution=True)
    assert kw.solve_linear(f"{linear}=0") == pytest.approx(linear_solution)

    for degree in (2, 3, 4):
        expression, solutions = kw.random_polynomial(degree=degree, get_solutions=True)
        function = kw.Function(f"f(x)={expression}")
        for solution in solutions:
            assert function(solution) == pytest.approx(0, abs=1e-4)

    expression = kw.random_polynomial2(3)
    assert isinstance(expression, str)


def test_class_random_generators_smoke():
    expression = kw.LinearEquation.random_expression(values=(1, 2), items_range=(2, 2), variable="x")
    assert isinstance(expression, str)
    equation, solution, variable = kw.LinearEquation.random_equation(
        values=(1, 3), items_per_side=(2, 2), digits_after=2,
        get_solution=True, get_variable=True, variable="x"
    )
    assert variable == "x"
    assert kw.solve_linear(equation) == pytest.approx(solution)

    assert isinstance(kw.QuadraticEquation.random(values=(-3, 3), get_solutions=True), tuple)
    assert isinstance(kw.CubicEquation.random(solutions_range=(-3, 3), get_solutions=True), tuple)
    assert isinstance(kw.QuarticEquation.random(solutions_range=(-3, 3), get_solutions=True), tuple)

    assert isinstance(kw.PolyEquation.random_expression(of_order=3), str)
    assert isinstance(kw.PolyEquation.random_quadratic(), str)
    assert isinstance(kw.PolyEquation.random_equation(of_order=3), str)


@pytest.mark.parametrize(
    ("coefficients", "expected_count"),
    [([1, -3, 2], 2), ([1, 0, 1], 2), ([1, -6, 11, -6], 3)],
)
def test_general_polynomial_solver_residuals(coefficients, expected_count):
    roots = kw.solve_polynomial(coefficients)
    assert len(roots) == expected_count
    for root in roots:
        residual = sum(coefficient * root ** power for power, coefficient in enumerate(reversed(coefficients)))
        assert residual == pytest.approx(0, abs=1e-5)


def test_real_solver_variants_and_factoring():
    assert sorted(kw.solve_cubic_real(1, -6, 11, -6)) == pytest.approx([1, 2, 3])
    assert kw.solve_quadratic_real(1, 0, 1) is None
    assert sorted(kw.solve_poly_by_factoring([1, -6, 11, -6])) == pytest.approx([1, 2, 3])
