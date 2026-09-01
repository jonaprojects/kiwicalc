import importlib

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


points_module = importlib.import_module("kiwicalc.geometry.points")


def test_point_reverse_arithmetic_multiplication_and_equality_edges(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    kw.Point((1, 2)).scatter(show=True)
    assert (kw.Point((1, 2))).__radd__([3, 4]) == kw.Point((4, 6))
    assert (kw.Point((1, 2))).__radd__(kw.Point((3, 4))) == kw.Point((4, 6))
    with pytest.raises(NotImplementedError):
        kw.Point((1, 2)).__radd__(3)
    assert kw.Point((1, 2)).__rsub__([3, 4]) == kw.Point((2, 2))
    assert kw.Point((1, 2)).__rsub__(kw.Point((3, 4))) == kw.Point((2, 2))
    with pytest.raises(NotImplementedError):
        kw.Point((1, 2)).__rsub__(3)

    assert kw.Point((1, 2)) * kw.Mono(2) == kw.Point((2, 4))
    symbolic = kw.Point((1, 2)) * kw.Var("x")
    assert symbolic.coordinates[0] == kw.Mono("x")
    assert kw.Point((1, 2)) * kw.Point((3, 4)) == 11
    with pytest.raises(NotImplementedError):
        kw.Point((1, 2)) * kw.PointCollection([kw.Point((3, 4))])

    assert not (kw.Point((1, 2)) == None)  # noqa: E711
    assert kw.Point((1, 2)) == kw.PointCollection([kw.Point((1, 2))])
    assert kw.Point((1, 2)) != kw.PointCollection([kw.Point((1, 2)), kw.Point((1, 2))])


def test_line_copy_symbolic_length_and_plot_dispatch(monkeypatch):
    first = kw.Point2D(0, 0)
    assert kw.Line2D(first, kw.Point2D(1, 1), gen_copies=False)._point1 is first
    assert kw.Line2D(first, kw.Point2D(1, 1), gen_copies=True)._point1 is not first
    assert kw.Line2D(kw.Point2D(kw.Var("x"), 0), (1, 1)).length() is None

    plots = importlib.import_module("kiwicalc.plotting.plots")
    calls = []
    monkeypatch.setattr(plots, "plot_function", lambda *args, **kwargs: calls.append("plot"))
    monkeypatch.setattr(plots, "scatter_function", lambda *args, **kwargs: calls.append(kwargs["title"]))
    vertical = kw.Line2D((1, 0), (1, 2))
    assert vertical.plot(show=False) is None
    assert vertical.scatter(show=False) is None
    line = kw.Line2D((0, 1), (1, 3))
    line.plot(show=False)
    line.scatter(show=False)
    line.scatter(show=False, title="custom")
    assert calls == ["plot", str(line), "custom"]


def test_circle_copy_symbolic_metrics_and_point_boundaries():
    radius = kw.Var("r")
    center = kw.Point2D(kw.Var("a"), kw.Var("b"))
    circle = kw.Circle(radius, center, gen_copies=True)
    assert circle.radius is not radius and circle.center is not center
    assert circle.area().variables
    assert circle.perimeter().variables
    numeric = kw.Circle(2, (0, 0))
    assert numeric.area() == pytest.approx(4 * kw.pi)
    assert numeric.perimeter() == pytest.approx(4 * kw.pi)
    with pytest.raises(ValueError):
        numeric.point_inside((kw.Var("x"), 0))
    assert not numeric.point_inside((-3, 0))
    assert not numeric.point_inside((0, 3))
    assert not numeric.point_inside((0, -3))


def test_circle_containment_plot_lambda_and_equation_edges(monkeypatch):
    outer = kw.Circle(5, (0, 0))
    assert kw.Circle(1, (0, 4.5)).is_inside(outer) is False
    assert kw.Circle(1, (0, -4.5)).is_inside(outer) is False
    assert kw.Circle(1, (-4.5, 0)).is_inside(outer) is False

    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "show", lambda: None)
    outer.plot(fig=fig, ax=ax)
    plt.close(fig)
    with pytest.raises(ValueError):
        kw.Circle(kw.Var("r"), (0, 0)).plot()

    with pytest.warns(UserWarning):
        numeric_lambda = outer.to_lambda()
    assert numeric_lambda(0) == (5, -5)
    with pytest.warns(UserWarning):
        symbolic_lambda = kw.Circle(kw.Var("r"), (0, 0)).to_lambda()
    assert symbolic_lambda(0)[0].variables

    assert "y^2" in kw.Circle(2, (1, 0)).equation
    assert "(y-(x+1))" in kw.Circle(2, (0, kw.Poly("x+1"))).equation


def test_circle_parameter_detection_intersection_and_y_intersections(monkeypatch):
    assert kw.Circle(kw.Var("r"), (0, 0)).has_parameters()
    assert kw.Circle(1, (kw.Var("a"), 0)).has_parameters()
    assert kw.Circle(1, (0, kw.Var("b"))).has_parameters()
    with pytest.raises(ValueError):
        kw.Circle(kw.Var("r"), (0, 0)).intersection(kw.Circle(1, (0, 0)))
    assert kw.Circle(1, (0, 0)).intersection(object()) is None

    assert kw.Circle(1, (2, 3)).y_intersection() is None
    with pytest.warns(UserWarning):
        complex_points = kw.Circle(1, (2, 3)).y_intersection(get_complex=True)
    assert len(complex_points) == 2
    assert kw.Circle(2, (2, 3)).y_intersection() == kw.Point((0, 3))
    assert len(kw.Circle(2, (0, 3)).y_intersection()) == 2


def test_mono_add_subtract_and_product_merge_edges():
    assert kw.Mono("x") + kw.Mono(0) == kw.Mono("x")
    assert kw.Mono("x") + kw.Abs(kw.Mono(0)) == kw.Mono("x")
    assert kw.Mono("x") - "x" == 0
    assert kw.Mono("x") - kw.Mono(0) == kw.Mono("x")
    assert kw.Mono("2x") + kw.Mono("3x") == kw.Mono("5x")
    assert kw.Mono("5x") - kw.Mono("3x") == kw.Mono("2x")

    unsimplified = kw.Poly(0)
    unsimplified.expressions = [kw.Mono("x"), kw.Mono("-x"), kw.Mono("y")]
    assert kw.Mono("x") * unsimplified == kw.Mono("xy")
    assert kw.Mono("x") * kw.Abs(2) == kw.Mono("2x")


def test_mono_division_sorting_and_calculus_edges():
    assert isinstance(kw.Mono("x") / kw.Poly("x+1"), kw.PolyFraction)
    assert kw.Mono(0) / kw.Mono("x") == 0
    with pytest.raises(ZeroDivisionError):
        kw.Mono("x").__itruediv__(kw.Mono(0))
    assert kw.Mono("x") / kw.Mono(2) == kw.Mono("0.5x")
    assert isinstance(kw.Mono("x") / kw.Mono("x^2"), kw.PolyFraction)

    assert (kw.Mono("x^3") < kw.Mono("x^2")) == 2
    assert (kw.Mono("x") < kw.Mono("xy")) == kw.Mono("x")
    assert (kw.Mono("xy") < kw.Mono("x")) == kw.Mono("x")
    assert isinstance(kw.Mono("x") < 2, kw.Mono)
    assert kw.Mono(1) < kw.Mono("x") == kw.Mono("x")

    with pytest.raises(ValueError):
        kw.Mono("xy").derivative()
    power_zero = kw.Mono(2, {"x": 0})
    assert power_zero.derivative() == 0
    assert kw.Mono("x^-2").derivative() == kw.Mono("-2x^-3")
    with pytest.raises(ValueError):
        kw.Mono("xy").integral()
    assert power_zero.integral() == kw.Mono("2x")
