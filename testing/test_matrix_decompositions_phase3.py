import numpy as np
import pytest

import kiwicalc as kw


def assert_matrix_close(actual, expected, **kwargs):
    assert actual.to_numpy() == pytest.approx(np.asarray(expected), **kwargs)


def test_diagonal_conjugate_and_hermitian_adjoint():
    assert kw.Matrix.diagonal([1, 2, 3]) == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    with pytest.raises(ValueError, match='at least one'):
        kw.Matrix.diagonal([])
    matrix = kw.Matrix([[1 + 2j, 3], [4j, 5 - 1j]])
    assert matrix.conjugate() == [[1 - 2j, 3], [-4j, 5 + 1j]]
    assert matrix.H == [[1 - 2j, -4j], [3, 5 + 1j]]


def test_partial_pivoted_lu_identity_and_reconstruction():
    matrix = kw.Matrix([[0, 2, 1], [1, 1, 0], [2, 1, 1]])
    result = matrix.lu()
    assert isinstance(result, kw.LUDecomposition)
    permutation, lower, upper = result
    assert_matrix_close(permutation @ matrix, lower @ upper)
    assert_matrix_close(result.reconstruct(), matrix.matrix)
    assert_matrix_close(lower, np.tril(lower.to_numpy()))
    assert_matrix_close(upper, np.triu(upper.to_numpy()))
    assert np.diag(lower.to_numpy()) == pytest.approx(np.ones(3))


def test_lu_handles_complex_values_and_rejects_unsupported_inputs():
    matrix = kw.Matrix([[1 + 1j, 2], [3, 4 - 1j]])
    assert_matrix_close(matrix.lu().reconstruct(), matrix.to_numpy())
    with pytest.raises(ValueError, match='square'):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]).lu()
    with pytest.raises(ValueError, match='non-singular'):
        kw.Matrix([[1, 2], [2, 4]]).lu()


@pytest.mark.parametrize('mode', ['reduced', 'complete'])
def test_qr_reconstructs_rectangular_matrix(mode):
    matrix = kw.Matrix([[1, 2], [3, 4], [5, 7]])
    result = matrix.qr(mode=mode)
    assert isinstance(result, kw.QRDecomposition)
    q, r = result
    assert_matrix_close(result.reconstruct(), matrix.to_numpy())
    assert_matrix_close(q.H @ q, np.eye(q.num_of_columns))
    assert_matrix_close(r, np.triu(r.to_numpy()))


def test_qr_validates_mode_and_supports_complex_matrices():
    with pytest.raises(ValueError, match='mode'):
        kw.Matrix.identity(2).qr(mode='raw')
    matrix = kw.Matrix([[1 + 1j, 2], [3j, 4], [5, 6 - 1j]])
    decomposition = matrix.qr()
    assert_matrix_close(decomposition.reconstruct(), matrix.to_numpy())
    assert_matrix_close(decomposition.q.H @ decomposition.q, np.eye(2))


def test_cholesky_real_and_complex_reconstruction():
    real = kw.Matrix([[4, 2], [2, 3]])
    lower = real.cholesky()
    assert_matrix_close(lower @ lower.H, real.to_numpy())
    assert_matrix_close(lower, np.tril(lower.to_numpy()))

    complex_matrix = kw.Matrix([[2, 1j], [-1j, 2]])
    complex_lower = complex_matrix.cholesky()
    assert_matrix_close(complex_lower @ complex_lower.H, complex_matrix.to_numpy())


def test_cholesky_reports_shape_symmetry_and_definiteness_errors():
    with pytest.raises(ValueError, match='square'):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]).cholesky()
    with pytest.raises(ValueError, match='symmetric'):
        kw.Matrix([[1, 2], [3, 4]]).cholesky()
    with pytest.raises(ValueError, match='positive-definite'):
        kw.Matrix([[1, 2], [2, 1]]).cholesky()


@pytest.mark.parametrize('full_matrices', [False, True])
def test_svd_reconstruction_and_named_factors(full_matrices):
    matrix = kw.Matrix([[1, 2], [3, 4], [5, 6]])
    result = matrix.svd(full_matrices=full_matrices)
    assert isinstance(result, kw.SVDDecomposition)
    u, singular_values, vt = result
    assert singular_values == tuple(sorted(singular_values, reverse=True))
    assert_matrix_close(result.reconstruct(), matrix.to_numpy())
    assert_matrix_close(u.H @ u, np.eye(u.num_of_columns))
    assert_matrix_close(vt @ vt.H, np.eye(vt.num_of_rows))


def test_general_eigen_decomposition_supports_real_and_complex_pairs():
    diagonal = kw.Matrix.diagonal([2, 5])
    result = diagonal.eigen()
    assert isinstance(result, kw.EigenDecomposition)
    values, vectors = result
    assert sorted(values) == pytest.approx([2, 5])
    assert_matrix_close(diagonal @ vectors, vectors @ kw.Matrix.diagonal(values))
    assert diagonal.eig().eigenvalues == values

    rotation = kw.Matrix([[0, -1], [1, 0]])
    complex_result = rotation.eigen()
    assert sorted(value.imag for value in complex_result.eigenvalues) == pytest.approx([-1, 1])
    assert_matrix_close(
        rotation @ complex_result.eigenvectors,
        complex_result.eigenvectors @ kw.Matrix.diagonal(complex_result.eigenvalues),
    )


def test_eigh_is_stable_for_symmetric_and_hermitian_matrices():
    matrix = kw.Matrix([[2, 1 - 1j], [1 + 1j, 3]])
    result = matrix.eigh()
    values, vectors = result
    assert tuple(values) == tuple(sorted(values))
    assert_matrix_close(vectors.H @ vectors, np.eye(2))
    assert_matrix_close(matrix @ vectors, vectors @ kw.Matrix.diagonal(values))
    with pytest.raises(ValueError, match='symmetric'):
        kw.Matrix([[1, 2], [3, 4]]).eigh()
    with pytest.raises(ValueError, match='square'):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]).eigen()


def test_decompositions_reject_symbolic_matrices_without_mutation():
    symbolic = kw.Matrix([[kw.Var('x')]])
    for operation in (symbolic.lu, symbolic.qr, symbolic.cholesky, symbolic.svd, symbolic.eigen, symbolic.eigh):
        with pytest.raises(TypeError, match='numeric matrix'):
            operation()

    matrix = kw.Matrix([[4, 2], [2, 3]])
    original = matrix.copy()
    matrix.lu()
    matrix.qr()
    matrix.cholesky()
    matrix.svd()
    matrix.eigen()
    matrix.eigh()
    assert matrix == original
