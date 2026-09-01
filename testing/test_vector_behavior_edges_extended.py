import math

import pytest

import kiwicalc as kw
from kiwicalc.geometry.vectors import _get_limits_vectors_2d, _get_limits_vectors_3d


def test_vector_constructor_forms_and_dimension_validation():
    assert kw.Vector(direction_vector=(3, 4), start_coordinate=(1, 2)).end_coordinate == [4, 6]
    assert kw.Vector(direction_vector=(3, 4), end_coordinate=(4, 6)).start_coordinate == [1, 2]
    assert kw.Vector(start_coordinate=(1, 2), end_coordinate=(4, 6)).direction == [3, 4]
    with pytest.raises(ValueError):
        kw.Vector()
    with pytest.raises(ValueError):
        kw.Vector((1, 2), start_coordinate=(0, 0, 0))
    with pytest.raises(ValueError):
        kw.Vector((1, 2), end_coordinate=(0, 0, 0))


def test_vector_products_ratios_and_lengths():
    vector = kw.Vector((3, 4))
    assert vector.length() == 5
    assert vector.equal_lengths(kw.Vector((0, 5)))
    assert vector.multiply((2, 3)) == 18
    assert vector.multiply(2) == kw.Vector((6, 8))
    assert kw.Vector((2, 4)).equal_direction_ratio(kw.Vector((1, 2)))
    assert not kw.Vector((2, 4)).equal_direction_ratio(kw.Vector((1, 3)))
    assert not kw.Vector((1, 2)).equal_direction_ratio(kw.Vector((1, 2, 3)))
    assert kw.Vector((0, 0)).equal_direction_ratio(kw.Vector((0, 0)))
    with pytest.raises(ValueError):
        vector.scalar_product((1, 2, 3))
    with pytest.raises(TypeError):
        vector.multiply(object())


def test_vector_power_and_arithmetic_errors():
    vector = kw.Vector((2, 3))
    assert vector**2 == kw.Vector((4, 9))
    assert kw.Vector((2, 3)).power_by(2) == kw.Vector((4, 9))
    assert vector**kw.Vector((2, 1)) == kw.Matrix([[4, 2], [9, 3]])
    with pytest.raises(TypeError):
        vector**object()
    with pytest.raises(TypeError):
        vector + object()
    with pytest.raises(TypeError):
        vector - object()
    with pytest.raises(TypeError):
        vector == object()


def test_vector_random_specializations(monkeypatch):
    values = iter([1, 2, 3, 4])
    monkeypatch.setattr("random.randint", lambda a, b: next(values))
    vector = kw.Vector.random_vector((0, 9), num_of_dimensions=2)
    assert vector.direction == [1, 2]
    assert vector.start_coordinate == [3, 4]

    values = iter([1, 2, 3, 4])
    monkeypatch.setattr("random.randint", lambda a, b: next(values))
    assert isinstance(kw.Vector2D.random_vector((0, 9)), kw.Vector2D)


def test_vector_specialization_validation():
    with pytest.raises(ValueError):
        kw.Vector2D(1, 2, start_coordinate=(0, 0, 0))
    with pytest.raises(ValueError):
        kw.Vector2D(1, 2, end_coordinate=(0, 0, 0))
    with pytest.raises(ValueError):
        kw.Vector3D(1, 2, 3, start_coordinate=(0, 0))
    with pytest.raises(ValueError):
        kw.Vector3D(1, 2, 3, end_coordinate=(0, 0))
    with pytest.raises(ValueError):
        kw.Vector((1,)).plot(show=False)


def test_vector_collection_order_statistics_and_removal():
    short = kw.Vector((1, 0))
    medium = kw.Vector((0, 2))
    long = kw.Vector((3, 4))
    collection = kw.VectorCollection(short, medium, long)
    assert collection.longest() == long
    assert collection.shortest() == short
    assert collection.longest(get_index=True) == (2, long)
    assert collection.shortest(get_index=True) == (0, short)
    assert collection.nlongest(2) == [long, medium]
    assert collection.nshortest(2) == [short, medium]
    assert collection.find(kw.Vector((9, 9))) == -1
    assert collection.longest(remove=True) == long
    assert len(collection) == 2
    assert collection.shortest(remove=True, get_index=True) == (0, short)


def test_vector_collection_append_setter_and_validation():
    collection = kw.VectorCollection()
    collection.append([(1, 2), (3, 4)])
    collection.append(kw.VectorCollection(kw.Vector((5, 6))))
    assert collection == [[1, 2], [3, 4], [5, 6]]
    collection += kw.Vector((7, 8))
    assert collection.num_of_vectors == 4
    collection.vectors = kw.Vector((9, 10))
    assert collection == kw.Vector((9, 10))
    with pytest.raises(TypeError):
        collection.append(object())
    with pytest.raises(TypeError):
        setattr(collection, "vectors", object())


def test_vector_collection_scalar_multiplication_and_division_errors():
    collection = kw.VectorCollection(kw.Vector((1, 2)), kw.Vector((3, 4)))
    assert collection * 3 == [[3, 6], [9, 12]]
    assert collection == [[1, 2], [3, 4]]
    collection *= 2
    assert collection == [[2, 4], [6, 8]]
    with pytest.raises(ValueError):
        collection / 0
    with pytest.raises(TypeError):
        collection / object()


def test_vector_plot_limit_helpers():
    vectors2d = [kw.Vector((1, 2), start_coordinate=(0, 0)), kw.Vector((-2, 1), start_coordinate=(3, -1))]
    assert _get_limits_vectors_2d(vectors2d) == pytest.approx((0, 3.15, -1.05, 2.1))
    vectors3d = [kw.Vector((1, 2, 3)), kw.Vector((-1, -2, -3), start_coordinate=(2, 2, 2))]
    assert _get_limits_vectors_3d(vectors3d) == (0, 2, 0, 2, -1, 3)
