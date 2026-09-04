import numpy as np
import pytest

import kiwicalc as kw


def assert_matrix_close(actual, expected, **kwargs):
    assert actual.to_numpy() == pytest.approx(np.asarray(expected), **kwargs)


def test_column_space_uses_independent_original_columns():
    matrix = kw.Matrix([[1, 2, 0], [0, 0, 1], [1, 2, 1]])
    basis = matrix.column_space()
    assert isinstance(basis, kw.VectorSpaceBasis)
    assert basis.space == 'column'
    assert basis.dimension == 2
    assert basis.ambient_dimension == 3
    assert basis[0] == [[1], [0], [1]]
    assert basis[1] == [[0], [1], [1]]
    assert basis.matrix == [[1, 0], [0, 1], [1, 1]]


def test_row_space_returns_rref_basis_vectors():
    matrix = kw.Matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])
    basis = matrix.row_space()
    assert basis.space == 'row'
    assert basis.dimension == matrix.rank() == 2
    assert basis.ambient_dimension == 3
    assert_matrix_close(basis.matrix, np.array([[1, 0], [0, 1], [1, 1]]))


def test_null_space_is_orthonormal_and_obeys_rank_nullity():
    matrix = kw.Matrix([[1, 2, 3], [2, 4, 6]])
    basis = matrix.null_space()
    assert basis.space == 'null'
    assert basis.dimension + matrix.rank() == matrix.num_of_columns
    null_matrix = basis.matrix
    assert_matrix_close(matrix @ null_matrix, np.zeros((2, basis.dimension)), abs=1e-12)
    assert_matrix_close(null_matrix.H @ null_matrix, np.eye(basis.dimension), abs=1e-12)

    algebraic = matrix.null_space(method='rref')
    assert algebraic[0] == [[-2], [1], [0]]
    assert algebraic[1] == [[-3], [0], [1]]
    with pytest.raises(ValueError, match='method'):
        matrix.null_space(method='magic')


def test_trivial_null_space_has_an_explicit_empty_basis():
    basis = kw.Matrix.identity(3).null_space()
    assert basis.is_trivial
    assert basis.dimension == 0
    assert basis.matrix is None
    assert basis.to_numpy().shape == (3, 0)
    assert list(basis) == []


def test_basis_selector_and_independence_checks():
    matrix = kw.Matrix([[1, 2, 3], [0, 1, 1]])
    assert matrix.basis().space == 'column'
    assert matrix.basis('row_space').space == 'row'
    assert matrix.basis('kernel').space == 'null'
    assert matrix.is_independent('rows')
    assert not matrix.is_independent('columns')
    with pytest.raises(ValueError, match='column'):
        matrix.basis('left')
    with pytest.raises(ValueError, match='rows'):
        matrix.is_independent('diagonal')


def test_gram_schmidt_returns_an_orthonormal_basis_without_mutation():
    matrix = kw.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    original = matrix.copy()
    basis = matrix.orthonormalize()
    assert isinstance(basis, kw.VectorSpaceBasis)
    assert basis.dimension == 2
    assert_matrix_close(basis.matrix.H @ basis.matrix, np.eye(2))
    assert matrix == original


def test_gram_schmidt_exposes_explanation_steps_and_dependencies():
    matrix = kw.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    result = matrix.orthonormalize(return_steps=True)
    assert isinstance(result, kw.GramSchmidtResult)
    assert len(result.steps) == 3
    assert isinstance(result.steps[0], kw.GramSchmidtStep)
    assert result.steps[0].projection_coefficients == ()
    assert result.steps[0].normalized is not None
    assert result.steps[2].dependent
    assert result.steps[2].normalized is None
    with pytest.raises(ValueError, match='dependent'):
        matrix.orthonormalize(drop_dependent=False)


def test_gram_schmidt_supports_rows_and_complex_inner_products():
    rows = kw.Matrix([[1, 1, 0], [1, 0, 1]])
    row_basis = rows.orthonormalize(axis='rows')
    assert row_basis.space == 'row'
    assert row_basis.ambient_dimension == 3
    assert_matrix_close(row_basis.matrix.H @ row_basis.matrix, np.eye(2))

    complex_matrix = kw.Matrix([[1, 1j], [1j, 1]])
    complex_basis = complex_matrix.orthonormalize()
    assert_matrix_close(complex_basis.matrix.H @ complex_basis.matrix, np.eye(complex_basis.dimension))
    with pytest.raises(ValueError, match='axis'):
        rows.orthonormalize(axis='depth')


def test_projection_onto_column_space_returns_optional_diagnostics():
    subspace = kw.Matrix([[1, 0], [0, 1], [0, 0]])
    projected = subspace.project_onto([1, 2, 3])
    assert projected == [[1], [2], [0]]

    result = subspace.project_onto([1, 2, 3], return_info=True)
    assert isinstance(result, kw.ProjectionResult)
    assert result.projected == projected
    assert result.residual == [[0], [0], [3]]
    assert result.coefficients == [[1], [2]]
    assert result.residual_norm == pytest.approx(3)
    assert_matrix_close(subspace.H @ result.residual, np.zeros((2, 1)))


def test_projection_supports_multiple_vectors_and_validates_dimensions():
    subspace = kw.Matrix([[1], [0], [0]])
    targets = kw.Matrix([[2, 4], [3, 5], [6, 7]])
    assert subspace.project_onto(targets) == [[2, 4], [0, 0], [0, 0]]
    with pytest.raises(ValueError, match='expected 3'):
        subspace.project_onto([1, 2])


def test_numeric_vector_space_operations_reject_symbolic_data():
    symbolic = kw.Matrix([[kw.Var('x')]])
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.orthonormalize()
    with pytest.raises(TypeError, match='numeric matrix'):
        symbolic.project_onto([1])


def test_vector_space_basis_validates_ambient_dimensions():
    with pytest.raises(ValueError, match='positive'):
        kw.VectorSpaceBasis((), 0)
    with pytest.raises(ValueError, match='column matrices'):
        kw.VectorSpaceBasis((kw.Matrix.row([1, 2]),), 2)
