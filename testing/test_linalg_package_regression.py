"""End-to-end regression coverage for the public linear-algebra package."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kiwicalc as kw


PUBLIC_LINALG_NAMES = (
    "Matrix", "LinearSolveResult", "LUDecomposition", "QRDecomposition",
    "SVDDecomposition", "EigenDecomposition", "VectorSpaceBasis",
    "GramSchmidtStep", "GramSchmidtResult", "ProjectionResult",
    "RowOperation", "RowReductionExplanation", "LinearAlgebraPlot",
    "AffineTransformation",
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def assert_close(actual, expected, tolerance=1e-10):
    if isinstance(actual, kw.Matrix):
        actual = actual.to_numpy()
    if isinstance(expected, kw.Matrix):
        expected = expected.to_numpy()
    assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_all_phase_types_are_exported_from_the_friendly_top_level_api():
    for name in PUBLIC_LINALG_NAMES:
        assert hasattr(kw, name), f"kiwicalc.{name} is missing"


def test_phase_1_foundation_regression_and_non_mutation():
    source = np.asarray(((1.0, 2.0, 3.0), (2.0, 4.0, 7.0)))
    matrix = kw.Matrix.from_numpy(source)
    original = matrix.copy()

    assert matrix.shape == (2, 3)
    assert matrix.T.shape == (3, 2)
    assert matrix.rank() == 2
    assert matrix.pivot_columns() == (0, 2)
    assert_close(matrix.rref(), ((1, 2, 0), (0, 0, 1)))
    assert_close(matrix.hadamard(matrix), source ** 2)
    assert matrix == original

    detached = matrix.to_numpy()
    detached[0, 0] = 999
    assert matrix[0][0] == 1


@pytest.mark.parametrize("seed", range(5))
def test_phase_2_square_solvers_match_numpy_for_well_conditioned_inputs(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    right = rng.normal(size=(4, 2))
    matrix = kw.Matrix(values.tolist())
    original = matrix.copy()

    result = matrix.solve(right, return_info=True)

    assert_close(result.solution, np.linalg.solve(values, right))
    assert result.rank == 4
    assert result.residual_norm < 1e-10
    assert result.condition_number == pytest.approx(np.linalg.cond(values))
    assert matrix == original


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (5, 2)])
def test_phase_2_rectangular_solver_and_pseudoinverse_regression(shape):
    rng = np.random.default_rng(sum(shape))
    values = rng.normal(size=shape)
    right = rng.normal(size=shape[0])
    matrix = kw.Matrix(values.tolist())

    assert_close(matrix.least_squares(right), np.linalg.lstsq(values, right, rcond=None)[0][:, None])
    pseudoinverse = matrix.pseudoinverse()
    assert_close(matrix @ pseudoinverse @ matrix, values)
    assert_close(pseudoinverse @ matrix @ pseudoinverse, pseudoinverse)


def test_phase_3_all_decompositions_reconstruct_the_same_matrix():
    matrix = kw.Matrix([[4.0, 1.0, 1.0], [1.0, 3.0, 0.5], [1.0, 0.5, 2.0]])
    original = matrix.copy()

    assert_close(matrix.lu().reconstruct(), matrix)
    assert_close(matrix.qr().reconstruct(), matrix)
    lower = matrix.cholesky()
    assert_close(lower @ lower.H, matrix)
    assert_close(matrix.svd().reconstruct(), matrix)

    eigenvalues, eigenvectors = matrix.eigh()
    assert_close(matrix @ eigenvectors, eigenvectors @ kw.Matrix.diagonal(eigenvalues))
    assert_close(eigenvectors.H @ eigenvectors, np.eye(3))
    assert matrix == original


def test_phase_4_fundamental_subspaces_and_projection_invariants():
    matrix = kw.Matrix([[1, 2, 3, 4], [2, 4, 6, 8], [0, 1, 1, 0]])
    rank = matrix.rank()
    columns = matrix.column_space()
    rows = matrix.row_space()
    null = matrix.null_space()

    assert columns.dimension == rows.dimension == rank
    assert rank + null.dimension == matrix.num_of_columns
    assert_close(matrix @ null.matrix, np.zeros((matrix.num_of_rows, null.dimension)))
    assert_close(null.matrix.H @ null.matrix, np.eye(null.dimension))

    orthonormal = columns.matrix.orthonormalize()
    assert_close(orthonormal.matrix.H @ orthonormal.matrix, np.eye(rank))
    projection = columns.matrix.project_onto([3, -1, 2], return_info=True)
    assert_close(columns.matrix.H @ projection.residual, np.zeros((rank, 1)))
    assert_close(projection.projected + projection.residual, [[3], [-1], [2]])


def test_phase_5_explanations_and_visualizations_produce_nonempty_figures():
    matrix = kw.Matrix([[2.0, 1.0], [0.0, 3.0]])
    explanation = matrix.explain_rref()
    plots = (
        explanation.plot(show=False),
        matrix.visualize_transformation(vectors=[(1, 1)], show=False),
        matrix.visualize_eigenvectors(show=False),
        matrix.visualize_svd(show=False),
        kw.Matrix([[1, 0], [1, 1], [1, 2]]).visualize_least_squares(
            [1, 2.5, 4.8], show=False,
        ),
    )

    assert explanation.result == matrix.rref()
    assert explanation.steps
    for plot in plots:
        assert isinstance(plot, kw.LinearAlgebraPlot)
        assert plot.axes
        assert plot.artists
        assert all(ax.has_data() or not ax.axison for ax in plot.axes)


def test_phase_6_affine_workflow_preserves_geometry_and_round_trips():
    transform = (
        kw.AffineTransformation.rotation(35, degrees=True)
        .scale(1.5, 0.75)
        .shear(x=0.2)
        .translate(3, -2)
    )
    point = kw.Point2D(2, -1)
    vector = kw.Vector2D(1, 2, start_coordinate=(2, -1))
    points = kw.Point2DCollection([(0, 0), (1, 2)])

    transformed_point = transform(point)
    transformed_vector = transform(vector)
    transformed_points = transform(points)

    assert isinstance(transformed_point, kw.Point2D)
    assert isinstance(transformed_vector, kw.Vector2D)
    assert isinstance(transformed_points, kw.Point2DCollection)
    assert_close(transform.inverse()(transformed_point).coordinates, point.coordinates)
    assert_close(transform.inverse()(transformed_vector).start_coordinate, vector.start_coordinate)
    assert_close(transform.inverse()(transformed_vector).direction, vector.direction)


def test_cross_phase_matrix_to_geometry_to_visualization_workflow():
    matrix = kw.Matrix([[0, -1], [1, 0]])
    affine = matrix.as_affine(translation=(2, 3))
    square = kw.Point2DCollection([(0, 0), (1, 0), (1, 1), (0, 1)])
    transformed = affine(square)
    plot = matrix.visualize_transformation(
        vectors=[point.coordinates for point in square.points[1:]], show=False,
    )

    assert [tuple(np.round(point.coordinates, 10)) for point in transformed.points] == [
        (2.0, 3.0), (2.0, 4.0), (1.0, 4.0), (1.0, 3.0),
    ]
    assert_close(plot.data["matrix"], matrix)
    assert len(plot.axes) == 2
