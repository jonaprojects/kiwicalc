import json
import math
import inspect
from typing import get_args, get_type_hints

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from matplotlib.quiver import Quiver
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting.fields import evaluate_xy, make_grid


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_two_variable_evaluation_respects_names_and_scalar_fallbacks():
    X, Y = np.meshgrid([1, 2], [3, 4])
    assert evaluate_xy("x + 2*y", X, Y) == pytest.approx(np.array([[7, 8], [9, 10]]))
    assert evaluate_xy("f(x,y)=x-y", X, Y) == pytest.approx(np.array([[-2, -1], [-3, -2]]))
    assert evaluate_xy(3, X, Y) == pytest.approx(np.full_like(X, 3))
    assert evaluate_xy(lambda x, y: math.sin(x) + y, X, Y) == pytest.approx(np.sin(X) + Y)


def test_vector_field_is_chainable_and_can_encode_magnitude():
    graph = kw.Graph2D()
    returned = graph.vector_field(lambda x, y: -y, lambda x, y: x, density=(5, 4), color="magnitude")
    assert returned is graph

    artists = graph.plot(show=False, xlim=(-2, 2), ylim=(-1, 1), return_artists=True)
    quiver = artists[0]
    assert isinstance(quiver, Quiver)
    assert quiver.U.size == 20
    assert np.max(quiver.get_array()) == pytest.approx(np.hypot(2, 1))


def test_vector_field_normalization_and_fixed_color():
    graph = kw.Graph2D().vector_field(3, 4, density=3, normalize=True, color="navy")
    quiver = graph.plot(show=False, return_artists=True)[0]
    assert np.hypot(quiver.U, quiver.V) == pytest.approx(np.ones(9))
    assert quiver.get_array() is None


def test_slope_field_uses_unit_direction_segments_by_default():
    graph = kw.Graph2D().slope_field(lambda x, y: x - y, density=5)
    quiver = graph.plot(show=False, xlim=(-1, 1), ylim=(-1, 1), return_artists=True)[0]
    assert isinstance(quiver, Quiver)
    assert np.hypot(quiver.U, quiver.V) == pytest.approx(np.ones(25))
    center = 12
    assert quiver.V[center] == pytest.approx(0)


def test_gradient_field_computes_a_numerical_gradient():
    graph = kw.Graph2D().gradient_field("f(x,y)=x^2+y^2", x_range=(-2, 2), y_range=(-2, 2), density=9, color="black")
    quiver = graph.plot(show=False, return_artists=True)[0]
    assert quiver.U.reshape(9, 9)[4, 6] == pytest.approx(2, abs=1e-8)
    assert quiver.V.reshape(9, 9)[6, 4] == pytest.approx(2, abs=1e-8)

    normalized = kw.Graph2D().gradient_field("f(x,y)=x^2+y^2", density=9, normalize=True, color="black")
    center = normalized.plot(show=False, return_artists=True)[0]
    assert center.U.reshape(9, 9)[4, 4] == pytest.approx(0)
    assert center.V.reshape(9, 9)[4, 4] == pytest.approx(0)


def test_streamlines_render_lines_and_arrows_with_optional_colorbar():
    graph = kw.Graph2D().streamlines(
        lambda x, y: -y, lambda x, y: x,
        x_range=(-2, 2), y_range=(-2, 2), samples=24,
        density=0.7, colorbar=True,
    )
    artists = graph.plot(show=False, return_artists=True)
    assert isinstance(artists[0], LineCollection)
    assert len(artists) == 2
    assert len(graph._colorbars) == 1
    assert graph._colorbars[0].ax.get_ylabel() == "Magnitude"


def test_line_and_filled_contours_support_labels_and_colorbars():
    lines = kw.Graph2D().contour_map("f(x,y)=x^2+y^2", levels=[1, 2, 3], labels=True, colors="navy")
    artists = lines.plot(show=False, xlim=(-2, 2), ylim=(-2, 2), return_artists=True)
    assert artists and len(lines.ax.texts) > 0

    filled = kw.Graph2D().contour(
        lambda x, y: np.sin(x) * np.cos(y), filled=True,
        levels=8, colorbar=True, colorbar_label="Temperature",
    )
    filled.plot(show=False, xlim=(-3, 3), ylim=(-3, 3))
    assert filled._colorbars[0].ax.get_ylabel() == "Temperature"


def test_streamplot_alias_and_plot_ranges_are_friendly_defaults():
    graph = kw.Graph2D().streamplot(1, 0, samples=12, color="teal")
    graph.plot(show=False, xlim=(2, 4), ylim=(5, 8))
    segments = graph.artists[0].get_segments()
    points = np.concatenate(segments)
    assert points[:, 0].min() >= 2
    assert points[:, 0].max() <= 4
    assert points[:, 1].min() >= 5
    assert points[:, 1].max() <= 8


def test_replotting_and_clearing_do_not_accumulate_colorbar_axes():
    graph = kw.Graph2D().theme("engineering").vector_field(1, 1, colorbar=True)
    graph.plot(show=False)
    assert len(graph.fig.axes) == 2
    assert graph._colorbars[0].ax.get_facecolor() == pytest.approx(to_rgba("#f7f9fb"))
    graph.plot(show=False)
    assert len(graph.fig.axes) == 2
    graph.clear()
    assert len(graph.fig.axes) == 1
    assert graph._colorbars == []


def test_field_layers_round_trip_with_portable_formulas():
    graph = (
        kw.Graph2D()
        .vector_field("f(x,y)=-y", "f(x,y)=x", density=8)
        .slope_field("f(x,y)=x-y", density=9)
        .gradient_field("f(x,y)=x^2+y^2", density=7)
        .streamlines("f(x,y)=-y", "f(x,y)=x", samples=20)
        .contour_map("f(x,y)=x^2-y^2", levels=[-2, 0, 2])
    )
    restored = kw.Graph.from_dict(json.loads(graph.to_json()))
    assert [item["kind"] for item in restored._decorations] == [
        "vector_field", "slope_field", "gradient_field", "streamlines", "contour_map"
    ]
    assert restored.plot(show=False, xlim=(-2, 2), ylim=(-2, 2), return_artists=True)


def test_unplotted_graphs_do_not_leak_empty_notebook_figures_during_round_trip():
    initial_figures = set(plt.get_fignums())
    portable = (
        kw.Graph2D()
        .vector_field("f(x,y)=-y", "f(x,y)=x", density=12)
        .contour_map("f(x,y)=x^2+y^2", levels=[1, 4, 9])
    )
    assert set(plt.get_fignums()) == initial_figures

    restored = kw.Graph.from_json(portable.to_json())
    assert set(plt.get_fignums()) == initial_figures
    artists = restored.plot(
        show=False, return_artists=True, xlim=(-3, 3), ylim=(-3, 3),
        equal_aspect=True,
    )
    assert len(artists) == 2
    assert len(set(plt.get_fignums()) - initial_figures) == 1


def test_phase_3_public_types_match_the_runtime_api():
    levels_type = get_type_hints(kw.Graph2D.contour_map)["levels"]
    assert int in get_args(levels_type)
    assert any("Iterable" in str(option) for option in get_args(levels_type))
    equal_aspect_type = get_type_hints(kw.Graph2D.plot)["equal_aspect"]
    assert set(get_args(equal_aspect_type)) == {bool, type(None)}

    base_plot = inspect.signature(kw.Graph.plot)
    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in base_plot.parameters.values())


def test_callable_fields_have_a_clear_serialization_error():
    graph = kw.Graph2D().vector_field(lambda x, y: x, lambda x, y: y)
    with pytest.raises(TypeError, match="callables cannot be serialized"):
        graph.to_dict()


@pytest.mark.parametrize(
    "action,message",
    [
        (lambda: make_grid((0, 1), (0, 1), 1), "at least 2"),
        (lambda: make_grid((0, 1), (0, 1), (2, 3, 4)), "integer or"),
        (lambda: kw.Graph2D().vector_field(1, 1, x_range=(2, -2)).plot(show=False), "smaller"),
        (lambda: kw.Graph2D().slope_field(object()).plot(show=False), "numbers, callables"),
        (lambda: kw.Graph2D().streamlines(1, 1, density=0).plot(show=False), "must be positive"),
    ],
)
def test_field_validation_errors_are_helpful(action, message):
    with pytest.raises((TypeError, ValueError), match=message):
        action()
