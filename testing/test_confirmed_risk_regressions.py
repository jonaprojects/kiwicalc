import importlib

import pytest

import kiwicalc as kw
from kiwicalc.numeric import roots

utils = importlib.import_module('kiwicalc.core.utils')


def test_determinant_tracks_row_swap_sign_and_preserves_source():
    matrix = kw.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    assert matrix.determinant() == -1
    assert matrix.matrix == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize(
    ('solver', 'coefficients'),
    [
        (kw.solve_quadratic, (0, 2, -4)),
        (kw.solve_cubic, (0, 0, 2, -4)),
        (kw.solve_quartic, (0, 0, 0, 2, -4)),
    ],
)
def test_fixed_degree_solvers_reduce_leading_zero_coefficients(solver, coefficients):
    assert solver(*coefficients) == pytest.approx((2,))


def test_general_polynomial_solver_strips_all_leading_zero_coefficients():
    assert kw.solve_polynomial([0, 0, 2, -4]) == [2]
    assert kw.solve_polynomial([0, 0, 0]) is None


def test_linear_system_supports_unique_overdetermined_system():
    result = kw.solve_linear_system(('x+y=2', 'x-y=0', '2x=2'))

    assert result == pytest.approx({'x': 1, 'y': 1})


@pytest.mark.parametrize(
    'equations',
    [
        ('x+y=2',),
        ('x+y=2', '2x+2y=4'),
        ('x+y=2', 'x+y=3'),
    ],
)
def test_linear_system_rejects_non_unique_or_inconsistent_systems(equations):
    with pytest.raises(ValueError):
        kw.solve_linear_system(equations)


def test_probability_paths_must_follow_parent_child_edges():
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'root'))
    tree.add(0.5, 'a')
    b = tree.add(0.4, 'b')
    tree.add(0.25, 'c', parent=b)

    assert tree.get_probability('root/b/c') == pytest.approx(0.1)
    with pytest.raises(ValueError):
        tree.get_probability('root/a/c')
    with pytest.raises(ValueError):
        tree.get_probability('missing')
    with pytest.raises(ValueError):
        tree.get_probability('root/c')


def test_matrix_rejects_ragged_rows():
    with pytest.raises(ValueError, match='same length'):
        kw.Matrix([[1, 2], [3]])


def test_gauss_handles_wide_matrix_with_zero_initial_columns():
    matrix = kw.Matrix([[0, 0, 1], [0, 1, 0]])

    matrix.gauss()

    assert matrix.matrix == [[0, 1, 0], [0, 0, 1]]


def test_polynomial_system_reports_singular_jacobian_cleanly():
    with pytest.raises(ValueError, match='Jacobian is singular'):
        kw.solve_poly_system(('x^2=1',), initial_vals={'x': 0}, nmax=3)


def test_skew_vectors_have_no_intersection():
    first = kw.Vector((1, 0, 0), start_coordinate=(0, 0, 0))
    second = kw.Vector((0, 1, 0), start_coordinate=(0, 0, 1))

    assert first.intersection(second) is None


def test_newton_returns_last_safe_estimate_on_zero_derivative():
    with pytest.warns(UserWarning, match='derivative is zero'):
        result = roots.newton_raphson(lambda x: x**2 + 1, lambda x: 2 * x, 1)

    assert result == 0


def test_halley_returns_last_safe_estimate_on_zero_denominator():
    with pytest.warns(UserWarning, match='denominator is zero'):
        result = roots.halleys_method(lambda x: x**3 + 1, lambda x: 3 * x**2, lambda x: 6 * x, 0)

    assert result == 0


def test_fitting_helpers_reject_degenerate_x_values():
    with pytest.raises(ValueError, match='distinct x values'):
        utils.linear_regression([1, 1], [2, 3])
    with pytest.raises(ValueError, match='distinct x values'):
        utils.lagrange_polynomial([1, 1], [2, 3])
