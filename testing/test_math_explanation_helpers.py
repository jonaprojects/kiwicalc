import json

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting.explanations import extrema, inflections, roots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_numeric_explanation_analysis_finds_roots_extrema_and_inflections():
    values = np.linspace(-3, 3, 1201)
    assert roots(lambda x: x * x - 1, values) == pytest.approx([-1, 1], abs=1e-4)

    turning_points = extrema(lambda x: x * x, values)
    assert turning_points[0][:2] == pytest.approx((0, 0), abs=1e-4)
    assert turning_points[0][2] == "minimum"

    bending_points = inflections(lambda x: x ** 3, values)
    assert bending_points[0] == pytest.approx((0, 0), abs=1e-4)


def test_point_explanations_are_chainable_and_use_offset_labels():
    cubic = lambda x: x ** 3 - 3 * x
    graph = kw.Graph2D([cubic])
    returned = graph.show_roots(cubic).show_extrema(cubic).show_inflections(cubic)
    assert returned is graph

    graph.plot(show=False, xlim=(-3, 3), ylim=(-5, 5))

    assert sum(isinstance(item, PathCollection) for item in graph.artists) == 3
    assert {text.get_text() for text in graph.ax.texts} >= {"maximum", "minimum", "inflection"}
    assert len({text.xyann for text in graph.ax.texts}) > 1


def test_intersections_can_be_unlabeled_or_custom_labeled():
    graph = kw.Graph2D().show_intersections(lambda x: x, lambda x: x * x, domain=(-1, 2), label="meeting")
    graph.plot(show=False)
    assert [text.get_text() for text in graph.ax.texts] == ["meeting", "meeting"]

    quiet = kw.Graph2D().show_intersections(lambda x: x, lambda x: -x, label=False)
    quiet.plot(show=False)
    assert len(quiet.ax.texts) == 0


def test_tangent_normal_secant_and_slope_triangle_render_friendly_layers():
    parabola = lambda x: x * x
    graph = (
        kw.Graph2D([parabola])
        .tangent(parabola, at=1, color="green")
        .normal(parabola, at=1, color="purple")
        .secant(parabola, between=(0, 2), color="gray")
        .slope_triangle(parabola, at=1, run=0.5)
    )
    graph.plot(show=False, xlim=(-1, 3), ylim=(-2, 5), legend=True)

    assert {line.get_label() for line in graph.ax.lines} >= {"tangent", "normal", "secant"}
    assert {text.get_text() for text in graph.ax.texts} == {"run = 0.5", "rise = 1"}


def test_tangent_and_normal_accept_curves_with_normalized_positions():
    circle = kw.Ellipse(2, 1, samples=101)
    graph = kw.Graph2D([circle]).tangent(circle, 0.25).normal(circle, 0.25)
    artists = graph.plot(show=False, equal_aspect=True, return_artists=True)
    assert len(artists) == 3


def test_asymptotes_support_automatic_and_explicit_values():
    automatic = kw.Graph2D([lambda x: 1 / x]).show_asymptotes(lambda x: 1 / x, domain=(-10, 10))
    automatic.plot(show=False, xlim=(-10, 10), ylim=(-5, 5))
    assert any(np.allclose(line.get_xdata(), [0, 0], atol=0.02) for line in automatic.ax.lines[1:])

    explicit = kw.Graph2D().show_asymptotes(lambda x: x, vertical=[-1, 2], horizontal=3)
    explicit.plot(show=False)
    assert len(explicit.ax.lines) == 3


def test_region_explanations_render_monotonicity_and_inequality_shading():
    graph = (
        kw.Graph2D([lambda x: x * x - 1])
        .show_monotonicity(lambda x: x * x - 1, domain=(-2, 2))
        .shade_solution(lambda x: x * x - 1, ">=", domain=(-2, 2), label="solution")
    )
    graph.plot(show=False, xlim=(-2, 2), ylim=(-2, 4), legend=True)
    assert any(isinstance(artist, PolyCollection) for artist in graph.artists)
    assert len(graph.ax.patches) >= 2


@pytest.mark.parametrize("method,patch_type", [("left", Rectangle), ("right", Rectangle), ("midpoint", Rectangle), ("trapezoid", Polygon)])
def test_riemann_sum_methods(method, patch_type):
    graph = kw.Graph2D([lambda x: x * x]).riemann_sum(lambda x: x * x, (0, 2), rectangles=4, method=method)
    graph.plot(show=False, xlim=(0, 2), ylim=(0, 4))
    patches = [artist for artist in graph.artists if isinstance(artist, patch_type)]
    assert len(patches) == 4


def test_derivative_and_integral_overlays_follow_expected_values():
    graph = kw.Graph2D().show_derivative(lambda x: x * x, domain=(-2, 2)).show_integral(lambda x: 2 * x, domain=(-2, 2), constant=4)
    graph.plot(show=False, legend=True)
    derivative, integral = graph.ax.lines
    assert derivative.get_ydata()[len(derivative.get_ydata()) // 2] == pytest.approx(0, abs=1e-4)
    assert integral.get_ydata()[-1] == pytest.approx(4, abs=1e-3)
    assert {line.get_label() for line in graph.ax.lines} == {"derivative", "integral"}


def test_explanation_layers_serialize_when_sources_are_portable():
    graph = (
        kw.Graph2D(["x^2"])
        .show_roots("x^2-1", domain=(-2, 2))
        .show_intersections("x", "x^2", domain=(-1, 2))
        .shade_solution("x", ">=", other="x^2", domain=(-1, 2))
        .riemann_sum("x^2", (0, 2), rectangles=3)
    )
    restored = kw.Graph.from_dict(json.loads(graph.to_json()))
    assert [item["kind"] for item in restored._decorations] == ["roots", "intersections", "solution", "riemann"]
    assert restored.plot(show=False, return_artists=True)


def test_callable_explanations_have_a_clear_serialization_error():
    graph = kw.Graph2D().show_roots(lambda x: x)
    with pytest.raises(TypeError, match="callables cannot be serialized"):
        graph.to_dict()


@pytest.mark.parametrize(
    "action,message",
    [
        (lambda: kw.Graph2D().secant(lambda x: x, (1,)), "two x coordinates"),
        (lambda: kw.Graph2D().riemann_sum(lambda x: x, (0, 1), rectangles=0).plot(show=False), "at least 1"),
        (lambda: kw.Graph2D().riemann_sum(lambda x: x, (0, 1), method="guess").plot(show=False), "left, right"),
        (lambda: kw.Graph2D().shade_solution(lambda x: x, "approximately").plot(show=False), "relation must"),
        (lambda: kw.Graph2D().slope_triangle(lambda x: x, 0, run=0).plot(show=False), "must not be zero"),
        (lambda: kw.Graph2D().show_roots(lambda x: x, domain=(2, -2)).plot(show=False), "smaller"),
    ],
)
def test_explanation_validation_errors_are_helpful(action, message):
    with pytest.raises(ValueError, match=message):
        action()
