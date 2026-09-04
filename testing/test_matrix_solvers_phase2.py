import math

import numpy as np
import pytest

import kiwicalc as kw


def test_solve_returns_a_column_matrix_and_preserves_inputs():
    coefficients = kw.Matrix([[3, 1], [1, 2]])
    original = coefficients.copy()
    solution = coefficients.solve([9, 8])
    assert solution.shape == (2, 1)
    assert solution == [[2], [3]]
    assert coefficients == original


def test_solve_accepts_vector_and_multiple_right_hand_sides():
    coefficients = kw.Matrix([[3, 1], [1, 2]])
    vector_solution = coefficients.solve(kw.Vector((9, 8)))
    assert vector_solution == [[2], [3]]

    right = kw.Matrix([[9, 1], [8, 0]])
    expected = np.linalg.solve(coefficients.to_numpy(), right.to_numpy())
    assert coefficients.solve(right).to_numpy() == pytest.approx(expected)


def test_solve_diagnostics_are_optional_and_friendly():
    result = kw.Matrix([[3, 1], [1, 2]]).solve([9, 8], return_info=True)
    assert isinstance(result, kw.LinearSolveResult)
    assert result.solution == [[2], [3]]
    assert result.residual_norm == pytest.approx(0)
    assert result.rank == 2
    assert result.condition_number == pytest.approx(np.linalg.cond([[3, 1], [1, 2]]))
    assert result.method == 'solve'
    assert result.is_exact


def test_solve_handles_complex_values():
    coefficients = kw.Matrix([[1 + 1j, 0], [0, 2 - 1j]])
    right = [2 + 2j, 4 - 2j]
    assert coefficients.solve(right) == [[2 + 0j], [2 + 0j]]


def test_solve_rejects_non_unique_and_malformed_systems():
    with pytest.raises(ValueError, match='least_squares'):
        kw.Matrix([[1, 0], [0, 1], [1, 1]]).solve([1, 2, 3])
    with pytest.raises(ValueError, match='singular'):
        kw.Matrix([[1, 2], [2, 4]]).solve([1, 2])
    with pytest.raises(ValueError, match='expected 2'):
        kw.Matrix.identity(2).solve([1, 2, 3])
    with pytest.raises(TypeError, match='numeric values'):
        kw.Matrix.identity(2).solve(['x', 'y'])
    with pytest.raises(ValueError, match='finite'):
        kw.Matrix.identity(2).solve([1, np.inf])


def test_least_squares_matches_numpy_for_overdetermined_system():
    coefficients = kw.Matrix([[1, 1], [1, 2], [1, 3]])
    right = [1, 2, 2]
    expected, _, _, _ = np.linalg.lstsq(coefficients.to_numpy(), np.asarray(right), rcond=None)
    solution = coefficients.least_squares(right)
    assert solution.shape == (2, 1)
    assert solution.to_numpy().ravel() == pytest.approx(expected)

    result = coefficients.least_squares(right, return_info=True)
    assert isinstance(result, kw.LinearSolveResult)
    assert result.method == 'least_squares'
    assert result.rank == 2
    assert result.residual_norm == pytest.approx(
        np.linalg.norm(coefficients.to_numpy() @ expected - np.asarray(right))
    )
    assert not result.is_exact


def test_least_squares_returns_minimum_norm_underdetermined_solution():
    coefficients = kw.Matrix([[1, 1]])
    solution = coefficients.least_squares([2])
    assert solution.to_numpy().ravel() == pytest.approx([1, 1])


def test_pseudoinverse_satisfies_moore_penrose_identities():
    matrix = kw.Matrix([[1, 2], [3, 4], [5, 6]])
    pseudoinverse = matrix.pseudoinverse()
    assert pseudoinverse.shape == (2, 3)
    assert (matrix @ pseudoinverse @ matrix).to_numpy() == pytest.approx(matrix.to_numpy())
    assert (pseudoinverse @ matrix @ pseudoinverse).to_numpy() == pytest.approx(pseudoinverse.to_numpy())
    with pytest.raises(ValueError, match='non-negative'):
        matrix.pseudoinverse(rcond=-1)


def test_condition_number_matches_numpy_and_reports_singularity():
    matrix = kw.Matrix([[4, 2], [1, 3]])
    assert matrix.condition_number() == pytest.approx(np.linalg.cond(matrix.to_numpy()))
    assert kw.Matrix.identity(3).condition_number() == pytest.approx(1)
    assert math.isinf(kw.Matrix([[1, 2], [2, 4]]).condition_number())


def test_numeric_solver_methods_reject_symbolic_matrices():
    symbolic = kw.Matrix([[kw.Var('x')]])
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.solve([1])
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.least_squares([1])
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.pseudoinverse()
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.condition_number()
