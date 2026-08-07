import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ('equation', 'expected'),
    [
        ('2x+1=7', 3),
        ('5-x=2x-4', 3),
        ('0.5x=2', 4),
    ],
)
def test_solve_linear_known_equations(equation, expected):
    assert kw.solve_linear(equation) == pytest.approx(expected)


def test_solve_linear_can_return_dict_and_json():
    assert kw.solve_linear('2x=8', get_dict=True) == {'x': 4}
    assert kw.solve_linear('2x=8', get_json=True) == '{"variable": "x", "result": 4.0}'


@pytest.mark.parametrize(
    ('solver', 'coefficients'),
    [
        (kw.solve_quadratic, (1, -5, 6)),
        (kw.solve_cubic, (1, -6, 11, -6)),
        (kw.solve_quartic, (1, 0, -5, 0, 4)),
    ],
)
def test_polynomial_solver_roots_satisfy_equation(solver, coefficients):
    roots = solver(*coefficients)
    for root in roots:
        value = sum(coefficient * root ** power for power, coefficient in enumerate(reversed(coefficients)))
        assert value == pytest.approx(0, abs=1e-6)


def test_quadratic_supports_complex_roots():
    roots = kw.solve_quadratic(1, 0, 1)
    assert {complex(round(root.real, 8), round(root.imag, 8)) for root in roots} == {1j, -1j}


def test_linear_system_solves_and_is_independent_of_variable_order():
    equations = ('x+y=5', '2x-y=1')
    assert kw.solve_linear_system(equations, variables=('x', 'y')) == {'x': 2, 'y': 3}
    assert kw.solve_linear_system(equations, variables=('y', 'x')) == {'y': 3, 'x': 2}


def test_linear_system_object_exposes_same_solution():
    system = kw.LinearSystem(('x+y=5', '2x-y=1'), variables=('x', 'y'))
    assert system.get_solutions() == {'x': 2, 'y': 3}


def test_newton_solver_handles_small_polynomial_system():
    result = kw.solve_poly_system(
        ('x+y=3', 'x-y=1'),
        initial_vals={'x': 0.0, 'y': 0.0},
    )
    assert result == pytest.approx({'x': 2, 'y': 1})
