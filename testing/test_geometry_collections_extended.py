import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw
from kiwicalc.core.utils import linear_regression
from kiwicalc.linalg.spaces import copy


def test_point_collection_coordinates_mutation_and_iteration():
    collection = kw.PointCollection([(0, 1), (2, 3), (4, 5)])
    assert collection.coords_at(0) == [0, 2, 4]
    assert collection.max_coord_at(1) == 5
    assert collection.min_coord_at(1) == 1
    assert collection.avg_coord_at(0) == pytest.approx(2)
    assert list(collection) == collection.points
    assert str(collection) == "(0,1), (2,3), (4,5)"
    assert "PointCollection" in repr(collection)

    collection.add_point(kw.Point2D(6, 7))
    collection.remove_point(0)
    assert collection.coords_at(0) == [2, 4, 6]
    with pytest.raises(IndexError):
        collection.coords_at(4)
    with pytest.raises(TypeError):
        kw.PointCollection([object()])


def test_dimension_specific_point_collections():
    one = kw.Point1DCollection([(1,), (2,)])
    two = kw.Point2DCollection([(1, 2), (3, 4)])
    three = kw.Point3DCollection([(1, 2, 3), (4, 5, 6)])
    four = kw.Point4DCollection([(1, 2, 3, 4), (5, 6, 7, 8)])

    assert one.coords_at(0) == [1, 2]
    assert two.x_values == [1, 3]
    assert two.y_values == [2, 4]
    assert two.sum() == kw.Point2D(4, 6)
    assert three.sum() == kw.Point3D(5, 7, 9)
    assert four.sum() == kw.Point4D(6, 8, 10, 12)

    with pytest.raises(ValueError):
        kw.Point2DCollection([(1, 2, 3)])
    with pytest.raises(ValueError):
        two.add_point(kw.Point3D(1, 2, 3))
    with pytest.raises(ValueError):
        three.add_point(kw.Point2D(1, 2))


def test_point_collection_regression_contract():
    collection = kw.Point2DCollection([(0, 1), (1, 3), (2, 5)])
    regression = collection.linear_regression()
    assert regression(5) == pytest.approx(11)


def test_point_collection_scatter_paths():
    collection = kw.Point2DCollection([(0, 1), (1, 3), (2, 5)])
    assert linear_regression(collection.x_values, collection.y_values, get_values=True) == pytest.approx((2, 1))
    collection.scatter(show=False)

    kw.Point1DCollection([(1,), (2,)]).scatter(show=False)
    kw.Point3DCollection([(1, 2, 3), (4, 5, 6)]).scatter(show=False)
    kw.Point4DCollection([(1, 2, 3, 4), (4, 5, 6, 7)]).scatter(show=False)
    plt.close("all")


def test_vector_arithmetic_power_and_factories():
    vector = kw.Vector((2, 3), start_coordinate=(1, 1))
    assert vector.end_coordinate == [3, 4]
    assert vector.general_point("t") == [1 + 2 * kw.Var("t"), 1 + 3 * kw.Var("t")]
    assert vector.multiply_all(2) == kw.Vector((4, 6), start_coordinate=(2, 2))
    fresh = kw.Vector((2, 3), start_coordinate=(1, 1))
    assert -fresh == kw.Vector((-2, -3), start_coordinate=(3, 4))
    assert abs(fresh) == kw.Vector((2, 3), start_coordinate=(1, 1))
    assert fresh.length() == pytest.approx(math.sqrt(13))
    assert kw.Vector.fill(3, 2) == kw.Vector((2, 2, 2))
    assert kw.Vector.fill_zeros(2) == kw.Vector((0, 0))


def test_vector_power_contract():
    assert kw.Vector((2, 3)).power_by_vector((2, 3)) == kw.Matrix([[4, 8], [9, 27]])


def test_vector_collection_full_protocol():
    first = kw.Vector((1, 0))
    second = kw.Vector((0, 2))
    collection = kw.VectorCollection(first, second)
    assert collection.num_of_vectors == 2
    assert second in collection
    assert bool(collection)
    assert list(collection) == [first, second]
    assert collection.total_number_of_items() == 3
    assert collection.to_matrix() == kw.Matrix([[1, 0], [0, 2]])

    assert list(collection.map(lambda vector: vector * 2)) == [first * 2, second * 2]
    assert list(collection.filter(lambda vector: vector.length() > 1)) == [second]
    collection.sort_by_length()
    assert collection.vectors == [first, second]

    assert collection[0] == first
    collection[0] = kw.Vector((4, 4))
    assert collection.pop() == second
    assert collection.pop() == kw.Vector((4, 4))
    assert not bool(collection)


def test_vector_collection_find_contract():
    collection = kw.VectorCollection(kw.Vector((1, 0)), kw.Vector((0, 2)))
    assert collection.find(kw.Vector((0, 2))) == 1


def test_vector_collection_copy_contract():
    collection = kw.VectorCollection(kw.Vector((1, 0)), kw.Vector((0, 2)))
    copied = collection.__copy__()
    assert copied == collection
    assert copied is not collection


def test_vector_collection_addition_contract():
    collection = kw.VectorCollection(kw.Vector((1, 0)), kw.Vector((0, 2)))
    assert len(collection + kw.Vector((3, 3))) == 3


def test_vector_specializations_and_limits():
    vector2d = kw.Vector2D(3, 4)
    vector3d = kw.Vector3D(1, 2, 3)
    assert (vector2d.x_step, vector2d.y_step) == (3, 4)
    assert (vector3d.x_step, vector3d.y_step, vector3d.z_step) == (1, 2, 3)
    vector2d.plot(show=False)
    vector3d.plot(show=False)
    plt.close("all")


def test_surface_construction_equality_metrics_and_copy():
    surface = kw.Surface((1, 2, 4, -8))
    assert (surface.a, surface.b, surface.c, surface.d) == (1, 2, 4, -8)
    assert surface == [1, 2, 4, -8]
    assert surface != (1, 2, 4, 7)
    assert surface.to_lambda()(0, 0) == pytest.approx(2)
    assert "Surface" in repr(surface)

    with pytest.raises(ValueError):
        kw.Surface((1, 2))
    with pytest.raises(TypeError):
        surface == "invalid"

    assert kw.mav(lambda x: x, lambda x: x + 1, 0, 2, 1) == pytest.approx(1)
    assert kw.msv(lambda x: x, lambda x: x + 2, 0, 2, 1) == pytest.approx(4)
    assert kw.mrv(lambda x: x, lambda x: x + 2, 0, 2, 1) == pytest.approx(2)

    expression = kw.Poly("x+1")
    with copy(expression) as copied:
        assert copied == expression
        assert copied is not expression


def test_surface_string_parsing_contract():
    assert kw.Surface("x+2y+4z-8=0").to_lambda()(0, 0) == pytest.approx(2)
