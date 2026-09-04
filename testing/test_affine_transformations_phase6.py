import numpy as np
import pytest

import kiwicalc as kw


def assert_coordinates(actual, expected):
    assert np.allclose(actual, expected, atol=1e-10)


def test_readable_composition_order_and_call_syntax():
    transform = kw.AffineTransformation.rotation(90, degrees=True).translate(2, 3)

    assert_coordinates(transform((1, 0)), (2, 4))
    assert transform.dimension == 2
    assert transform.offset == (2, 3)
    assert transform.is_rigid
    assert transform.preserves_orientation
    assert transform.determinant == pytest.approx(1)

    exposed_matrix = transform.matrix
    exposed_matrix.matrix[0][2] = 999
    assert_coordinates(transform((1, 0)), (2, 4))


def test_scale_shear_and_reflection_constructors():
    centered = kw.AffineTransformation.scaling(2, 3, center=(1, 1))
    assert_coordinates(centered((1, 1)), (1, 1))
    assert_coordinates(centered((2, 2)), (3, 4))
    assert centered.determinant == pytest.approx(6)
    assert not centered.is_rigid

    shear = kw.AffineTransformation.shearing(x=2)
    assert_coordinates(shear((1, 3)), (7, 3))

    reflected = kw.AffineTransformation.reflection("y=x")
    assert_coordinates(reflected((2, 5)), (5, 2))
    assert not reflected.preserves_orientation
    assert reflected.is_rigid


def test_arbitrary_reflection_and_reflection_around_a_point():
    vertical_line = kw.AffineTransformation.reflection((1, 0), point=(2, 0))
    assert_coordinates(vertical_line((5, 4)), (-1, 4))

    through_origin = kw.AffineTransformation.reflection("origin", dimension=3)
    assert_coordinates(through_origin((1, -2, 3)), (-1, 2, -3))

    with pytest.raises(ValueError, match="cannot be zero"):
        kw.AffineTransformation.reflection((0, 0))


def test_3d_rotation_supports_named_and_arbitrary_axes():
    around_z = kw.AffineTransformation.rotation(90, axis="z", degrees=True)
    assert_coordinates(around_z((1, 0, 2)), (0, 1, 2))

    around_center = kw.AffineTransformation.rotation(
        np.pi, axis=(0, 0, 2), center=(1, 0, 0),
    )
    assert_coordinates(around_center((2, 0, 0)), (0, 0, 0))

    chained = kw.AffineTransformation.identity(3).rotate(90, degrees=True)
    assert_coordinates(chained((1, 0, 0)), (0, 1, 0))


def test_inverse_and_both_composition_styles():
    first = kw.AffineTransformation.translation(3, -1)
    second = kw.AffineTransformation.scaling(2)
    readable = first.then(second)
    matrix_style = second @ first

    assert_coordinates(readable((1, 2)), (8, 2))
    assert_coordinates(matrix_style((1, 2)), (8, 2))
    assert_coordinates(readable.inverse()(readable((4, -3))), (4, -3))

    with pytest.raises(ValueError, match="not invertible"):
        kw.AffineTransformation.scaling(0).inverse()


def test_points_and_collections_keep_their_geometry_types():
    transform = kw.AffineTransformation.translation(2, -1)
    point = transform.apply(kw.Point2D(3, 4))
    collection = transform.apply(kw.Point2DCollection([(0, 0), (1, 2)]))

    assert isinstance(point, kw.Point2D)
    assert_coordinates(point.coordinates, (5, 3))
    assert isinstance(collection, kw.Point2DCollection)
    assert [point.coordinates for point in collection.points] == [[2.0, -1.0], [3.0, 1.0]]


def test_vectors_transform_start_and_end_without_becoming_new_vector_abstractions():
    vector = kw.Vector2D(2, 0, start_coordinate=(1, 1))
    transform = kw.AffineTransformation.rotation(90, degrees=True).translate(4, 0)

    result = transform(vector)

    assert isinstance(result, kw.Vector2D)
    assert_coordinates(result.start_coordinate, (3, 1))
    assert_coordinates(result.direction, (0, 2))
    assert_coordinates(vector.start_coordinate, (1, 1))
    assert_coordinates(vector.direction, (2, 0))


def test_curves_use_the_existing_homogeneous_transformation_pipeline():
    circle = kw.Ellipse(1, 1)
    transform = kw.AffineTransformation.translation(2, 3).scale(2)

    curve = transform(circle)
    x, y = curve.sample(samples=5)

    assert isinstance(curve, kw.TransformedCurve2D)
    assert x[0] == pytest.approx(6)
    assert y[0] == pytest.approx(6)

    with pytest.raises(ValueError, match="2D transformation.*3D curve"):
        transform(kw.Helix())


def test_numpy_and_plain_coordinate_data_are_convenient():
    transform = kw.AffineTransformation.translation(1, 2)
    array = np.asarray(((0, 0), (2, 3)), dtype=float)

    transformed = transform(array)

    assert isinstance(transformed, np.ndarray)
    assert_coordinates(transformed, ((1, 2), (3, 5)))
    assert transform([(0, 0), (2, 3)]) == [(1.0, 2.0), (3.0, 5.0)]


def test_matrix_bridge_accepts_linear_and_homogeneous_matrices():
    linear = kw.Matrix([[2, 0], [0, 3]])
    transform = linear.as_affine(translation=(1, -1))
    assert_coordinates(transform((2, 2)), (5, 5))
    assert transform.linear == linear

    homogeneous = kw.Matrix([[1, 0, 4], [0, 1, 5], [0, 0, 1]])
    assert_coordinates(homogeneous.as_affine(homogeneous=True)((1, 1)), (5, 6))

    with pytest.raises(ValueError, match="already encoded"):
        homogeneous.as_affine((1, 2), homogeneous=True)
    with pytest.raises(ValueError, match="2x2 or 3x3"):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]).as_affine()


def test_friendly_validation_errors_cover_dimensions_and_values():
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        kw.AffineTransformation.identity(4)
    with pytest.raises(ValueError, match="3x3.*4x4"):
        kw.AffineTransformation.from_matrix(kw.Matrix.identity(2))
    with pytest.raises(ValueError, match="final homogeneous row"):
        kw.AffineTransformation.from_matrix([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    with pytest.raises(ValueError, match="different dimensions"):
        kw.AffineTransformation.identity(2).then(kw.AffineTransformation.identity(3))
    with pytest.raises(ValueError, match="Expected a 2D"):
        kw.AffineTransformation.identity(2).translate(1, 2, 3)
    with pytest.raises(ValueError, match="shape"):
        kw.AffineTransformation.identity(2)(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="geometry or numeric"):
        kw.AffineTransformation.identity(2)("not coordinates")
    with pytest.raises(ValueError, match="supports 2D"):
        kw.AffineTransformation.identity(3).shear(x=1)
