import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.core.ranges import values_in_range


def test_sequence_base_protocol_branches(monkeypatch):
    arithmetic = kw.ArithmeticProg(2, difference=3)
    assert list(arithmetic.range(1, 4)) == [2, 5, 8]
    assert arithmetic.product_in_range(1, 4) == 80
    assert arithmetic.product_first_n(3) == 80
    assert arithmetic.sum_in_range(1, 4) == 15
    assert 8 in arithmetic
    assert 7 not in arithmetic
    assert arithmetic[3] == 8
    with pytest.warns(UserWarning, match="indices start from 1"):
        assert arithmetic[0:3:2] == [2, 8]
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    arithmetic.plot(1, 4, show=True)
    arithmetic.plot(1, 4, show=False)
    assert called == [True]
    plt.close("all")


@pytest.mark.parametrize(
    ("factory", "source", "attribute", "expected"),
    [
        (kw.GeometricSeq, "2,6", "ratio", 3),
        (kw.GeometricSeq, "2 6", "ratio", 3),
        (kw.ArithmeticProg, "2,6", "difference", 4),
        (kw.ArithmeticProg, "2 6", "difference", 4),
    ],
)
def test_sequence_string_parsing_and_inference(factory, source, attribute, expected):
    sequence = factory(source)
    assert getattr(sequence, attribute) == expected
    assert "Sequence" in repr(sequence)
    assert "..." in str(sequence)


def test_sequence_constructor_and_index_edges():
    assert kw.GeometricSeq(3, ratio=2).first == 3
    assert kw.ArithmeticProg(3, difference=2).first == 3
    assert kw.GeometricSeq([2, 6]).index_of(12) == -1
    assert kw.ArithmeticProg([2, 6]).index_of(5) == -1
    with pytest.raises(ValueError, match="Zeroes"):
        kw.GeometricSeq([0, 1])
    with pytest.raises(ValueError, match="specify the ratio"):
        kw.GeometricSeq([2])
    with pytest.raises(ValueError, match="specify the difference"):
        kw.ArithmeticProg([2])
    with pytest.raises(TypeError):
        kw.GeometricSeq(object())
    with pytest.raises(TypeError):
        kw.ArithmeticProg(object())


def test_recursive_sequence_edge_contracts():
    sequence = kw.RecursiveSeq("a_n = a_{n-1} + a_{n-2}", (1, 1))
    assert sequence.first == 1
    assert sequence.at_n(5, accumulate=False) == 5
    assert not sequence.place_already_found(4)
    assert str(sequence) == "a_n = a_{n-1} + a_{n-2}"
    with pytest.raises(NotImplementedError):
        sequence.index_of(5)
    with pytest.raises(NotImplementedError):
        sequence.sum_first_n(5)
    insufficient = kw.RecursiveSeq("a_n = a_{n-1} + a_{n-2}", (1,))
    with pytest.raises(ValueError, match="Not enough initial values"):
        insufficient.at_n(3)


def test_range_constructor_validation_and_infinite_edges():
    with pytest.raises(TypeError, match="expression"):
        kw.Range(object(), (0, 1), (kw.LESS_THAN, kw.LESS_THAN))
    with pytest.raises(TypeError, match="limits"):
        kw.Range(1, None, (kw.LESS_THAN, kw.LESS_THAN))
    with pytest.raises(ValueError, match="length"):
        kw.Range(1, (0,), (kw.LESS_THAN, kw.LESS_THAN))
    with pytest.raises(TypeError, match="Minimum"):
        kw.Range(1, (object(), 2), (kw.LESS_THAN, kw.LESS_THAN))
    with pytest.raises(TypeError, match="Maximum"):
        kw.Range(1, (0, object()), (kw.LESS_THAN, kw.LESS_THAN))
    with pytest.raises(TypeError, match="operators"):
        kw.Range(1, (0, 2), None)
    with pytest.raises(ValueError, match="operators"):
        kw.Range(1, (0, 2), (kw.LESS_THAN,))

    impossible_min = kw.Range(1, (np.inf, np.inf), (kw.LESS_THAN, kw.LESS_THAN))
    impossible_max = kw.Range(1, (-np.inf, -np.inf), (kw.LESS_THAN, kw.LESS_THAN))
    unbounded = kw.Range(1, (None, None), (None, None))
    assert impossible_min.try_evaluate() is False
    assert impossible_max.try_evaluate() is False
    assert unbounded.try_evaluate() is True
    assert str(unbounded) == "-∞None1None∞"


def test_range_collection_all_boolean_outcomes_and_operators():
    true_range = kw.Range(1, (0, 2), (kw.LESS_THAN, kw.LESS_THAN))
    false_range = kw.Range(3, (0, 2), (kw.LESS_THAN, kw.LESS_THAN))
    unknown_range = kw.create_range("x<2")
    assert kw.RangeOR((false_range, false_range)).try_evaluate() is False
    assert kw.RangeOR((false_range, unknown_range)).try_evaluate() is None
    assert kw.RangeAND((true_range, true_range)).try_evaluate() is True
    assert kw.RangeAND((true_range, unknown_range)).try_evaluate() is None
    assert isinstance(kw.RangeCollection((true_range,)) | false_range, kw.RangeOR)
    assert isinstance(kw.RangeCollection((true_range,)) & false_range, kw.RangeAND)
    nested = kw.RangeOR((kw.RangeAND((true_range, false_range)), unknown_range))
    assert "(" in str(nested)
    assert isinstance(nested.__copy__(), kw.RangeOR)
    assert isinstance(kw.RangeAND((true_range,)).__copy__(), kw.RangeAND)
    assert nested.simplify() is None


def test_values_in_range_rounding_branch():
    values, results = values_in_range(lambda x: x / 3, 0, 0.3, 0.1, round_results=True)
    assert values == pytest.approx([0, 0.1, 0.2])
    assert results == pytest.approx([0, 0.0333333333, 0.0666666667], abs=1e-5)


def test_point_collection_distance_and_dimension_edges(monkeypatch):
    collection = kw.PointCollection([(0, 0), (3, 4), (3, 0)])
    assert collection.longest_distance() == 5
    distance, points = collection.longest_distance(get_points=True)
    assert distance == 5 and len(points) == 2
    assert collection.shortest_distance() == 3
    assert collection.shortest_distance(get_points=True)[0] == 3
    assert kw.PointCollection([]).longest_distance() == 0
    assert kw.PointCollection([(1, 2)]).shortest_distance() == 0
    for method in (collection.max_coord_at, collection.min_coord_at, collection.avg_coord_at):
        with pytest.raises(IndexError):
            method(7)
    with pytest.raises(ValueError, match="1 coordinates"):
        kw.Point1DCollection([(1, 2)])
    with pytest.raises(ValueError, match="4 coordinates"):
        kw.Point4DCollection([(1, 2, 3)])
    with pytest.raises(ValueError, match="4D"):
        kw.PointCollection([(1, 2, 3, 4, 5)]).scatter(show=False)
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    kw.PointCollection([(1,), (2,)]).scatter(show=True)
    assert called == [True]
    plt.close("all")


def test_point_collection_regression_show_branches(monkeypatch):
    collection = kw.Point2DCollection([(0, 1), (1, 3), (2, 5)])
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    collection.plot_regression(show=True)
    collection.scatter_with_regression(show=False)
    assert collection.linear_regression(get_tuple=True) == pytest.approx((2, 1))
    assert called == [True]
    plt.close("all")


def test_surface_constructor_equality_and_zero_sample_edges():
    surface = kw.Surface((1, 2, 3))
    assert surface.d == 0
    assert surface == kw.Surface((1, 2, 3, 0))
    assert surface == (1, 2, 3, 0)
    assert surface == {0, 1, 2, 3}
    assert surface is not None and surface != None
    zero_c = kw.Surface((1, 2, 0, 3))
    with pytest.warns(UserWarning, match="c = 0"):
        assert zero_c.to_lambda()(4, 5) == 0
    for metric in (kw.mav, kw.msv, kw.mrv):
        with pytest.raises(ZeroDivisionError, match="0 points"):
            metric(lambda x: x, lambda x: x, 2, 1, 1)


@pytest.mark.parametrize("dtype", ["poly", "log", "ln", "trigo", "root", "factorial"])
def test_expression_factory_dtype_branches(dtype):
    sources = {
        "poly": "x+1", "log": "log(x)", "ln": "ln(x)",
        "trigo": "sin(x)", "root": 4, "factorial": 4,
    }
    assert kw.create(sources[dtype], dtype=dtype) is not None


def test_expression_factory_rejects_unknown_inputs():
    assert kw.create_from_dict(3).try_evaluate() == 3
    with pytest.raises(ValueError, match="Invalid parameter"):
        kw.create("x", dtype="unknown")
    with pytest.raises(ValueError, match="Unknown expression type"):
        kw.create_from_dict({"type": "mystery"})
