import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw
from kiwicalc.geometry.points import process_to_points


def test_dimension_specific_point_properties_and_protocols():
    one = kw.Point1D(1)
    two = kw.Point2D(1, 2)
    three = kw.Point3D(1, 2, 3)
    four = kw.Point4D(1, 2, 3, 4)
    assert one.x == 1
    assert (two.x, two.y) == (1, 2)
    assert (three.x, three.y, three.z) == (1, 2, 3)
    assert (four.x, four.y, four.z, four.c) == (1, 2, 3, 4)
    assert -kw.Point((1, -2)) == kw.Point((-1, 2))
    assert len(two) == 2
    assert repr(two) == "Point([1, 2])"
    assert str(two) == "(1,2)"


def test_point_mutation_and_error_paths():
    point = kw.Point((1, 2))
    point.coordinates = [3, 4]
    point += kw.Point((1, 1))
    assert point == kw.Point((4, 5))
    point -= (2, 3)
    assert point == kw.Point((2, 2))
    with pytest.raises(TypeError):
        point + 1
    with pytest.raises(TypeError):
        point - 1
    with pytest.raises(TypeError):
        kw.Point(object())
    with pytest.raises(TypeError):
        point == (2, 2)


def test_point_scatter_for_supported_dimensions():
    kw.Point(1).scatter(show=False)
    kw.Point((1, 2)).scatter(show=False)
    kw.Point((1, 2, 3)).scatter(show=False)
    plt.close("all")


def test_line_metrics_equation_and_callable():
    line = kw.Line2D((0, 1), kw.Point2D(2, 5))
    assert line.middle() == kw.Point2D(1, 3)
    assert line.length() == pytest.approx(math.sqrt(20))
    assert line.slope == 2
    assert line.free_number == 1
    assert line.equation() == "2.0x+1"
    assert line.to_lambda()(4) == 9
    with pytest.raises(TypeError):
        kw.Line2D(object(), (1, 2))
    with pytest.raises(TypeError):
        kw.Line2D((1, 2), object())


def test_vertical_line_contract():
    line = kw.Line2D((2, 1), (2, 5))
    with pytest.warns(UserWarning):
        assert line.slope is None
    with pytest.warns(UserWarning):
        assert line.free_number is None
    with pytest.warns(UserWarning):
        assert line.equation() is None
    with pytest.warns(UserWarning):
        assert line.to_lambda() is None


def test_circle_properties_equation_and_callable():
    circle = kw.Circle(3, center=(1, -2))
    assert circle.radius == 3
    assert circle.diameter == 6
    assert circle.center == kw.Point((1, -2))
    assert circle.left_edge == kw.Point((-2, -2))
    assert circle.right_edge == kw.Point((4, -2))
    assert circle.top_edge == kw.Point((1, 1))
    assert circle.bottom_edge == kw.Point((1, -5))
    assert circle.area() == pytest.approx(9 * math.pi)
    assert circle.perimeter() == pytest.approx(6 * math.pi)
    assert circle.equation == "(x-1)^2 + (y-(-2))^2 = 9"
    with pytest.warns(UserWarning):
        upper, lower = circle.to_lambda()(1)
    assert (upper, lower) == pytest.approx((1, -5))
    assert "Circle" in repr(circle)
    assert "Circle" in str(circle)


def test_circle_point_containment_uses_radial_distance():
    circle = kw.Circle(2)
    assert circle.point_inside((0, 0))
    assert circle.point_inside(kw.Point2D(2, 0))
    assert not circle.point_inside((1.9, 1.9))
    assert not circle.point_inside((3, 0))
    with pytest.raises(ValueError):
        circle.point_inside((1, 2, 3))
    with pytest.raises(ValueError):
        circle.point_inside(object())


def test_circle_inside_and_y_intersections():
    outer = kw.Circle(5, center=(1, 1))
    assert kw.Circle(2, center=(1, 1)).is_inside(outer)
    assert not kw.Circle(3, center=(5, 1)).is_inside(outer)
    with pytest.raises(TypeError):
        outer.is_inside(object())

    intersections = kw.Circle(2).y_intersection()
    assert intersections == (kw.Point((0, 2)), kw.Point((0, -2)))
    assert kw.Circle(2, center=(2, 3)).y_intersection() == kw.Point((0, 3))
    assert kw.Circle(1, center=(2, 0)).y_intersection() is None
    with pytest.warns(UserWarning):
        complex_points = kw.Circle(1, center=(2, 0)).y_intersection(get_complex=True)
    assert len(complex_points) == 2


def test_circle_symbolic_parameters_assignment_and_validation():
    circle = kw.Circle(kw.Var("r"), center=(kw.Var("a"), kw.Var("b")))
    assert circle.has_parameters()
    with pytest.raises(ValueError):
        circle.point_inside((0, 0))
    assigned = circle.when(r=2, a=1, b=1)
    assert not assigned.has_parameters()
    assert assigned.point_inside((1, 1))
    assert circle.has_parameters()
    with pytest.raises(ValueError):
        circle.is_inside(kw.Circle(4))
    with pytest.raises(TypeError):
        kw.Circle(object())
    with pytest.raises(ValueError):
        kw.Circle(1, center=(1, 2, 3))
    with pytest.raises(TypeError):
        kw.Circle(1, center=(1, object()))


def test_process_to_points_with_values_and_errors():
    assert process_to_points(lambda x: x * x, values=[-1, 0, 2]) == ([-1, 0, 2], [1, 0, 4])

    def guarded(value):
        if value == 0:
            raise ValueError
        return 1 / value

    assert process_to_points(guarded, values=[-1, 0, 1]) == ([-1, 0, 1], [-1, None, 1])
    values, results = process_to_points("f(x)=x+1", start=0, stop=2, step=1)
    assert values == [0, 1, 2]
    assert results == [1, 2, 3]
