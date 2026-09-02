import matplotlib.pyplot as plt
import numpy as np
import pytest
from typing import get_args, get_type_hints

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_marks_annotations_and_guide_lines_are_chainable():
    graph = kw.Graph2D()
    returned = (
        graph
        .mark((1, 2), label="point", color="red")
        .annotate("maximum", at=(1, 2), offset=(8, 10), color="navy")
        .vertical_line(1, label="x=1", linestyle="--")
        .horizontal_line(0, label="axis", color="gray")
    )
    assert returned is graph
    artists = graph.plot(show=False, legend=True, return_artists=True)
    assert len(artists) == 4
    assert len(graph.ax.texts) == 1
    assert graph.ax.texts[0].get_text() == "maximum"
    assert {line.get_label() for line in graph.ax.lines} >= {"x=1", "axis"}


def test_graph2d_titles_are_explicit_and_title_is_the_preferred_keyword():
    graph = kw.Graph2D().add(lambda x: np.sin(x), label="sin(x)").mark((np.pi, 0))
    graph.plot(show=False)
    assert graph.ax.get_title() == ""

    titled = kw.Graph2D([lambda x: x])
    titled.plot(show=False, title="A useful title")
    assert titled.ax.get_title() == "A useful title"

    legacy = kw.Graph2D([lambda x: x])
    legacy.plot(show=False, text="Legacy title")
    assert legacy.ax.get_title() == "Legacy title"

    with pytest.raises(ValueError, match="either title or text"):
        kw.Graph2D().plot(show=False, title="title", text="text")


def test_fill_between_functions_expressions_and_constants():
    graph = kw.Graph2D()
    graph.fill_between(lambda x: x * x, 0, values=[-1, 0, 1], label="area", color="skyblue", alpha=0.5)
    artists = graph.plot(show=False, return_artists=True)
    assert len(artists) == 1
    assert artists[0] in graph.ax.collections

    expression_graph = kw.Graph2D()
    expression_graph.fill_between(kw.Poly("x"), "f(x)=x^2", values=[0, 0.5, 1])
    assert len(expression_graph.plot(show=False, return_artists=True)) == 1


def test_fill_between_accepts_a_callable_lower_boundary_in_its_public_type():
    upper = lambda x: 4 - x * x
    lower = lambda x: x * x
    graph = kw.Graph2D().fill_between(upper, lower, values=np.linspace(-2, 2, 31))
    assert len(graph.plot(show=False, return_artists=True)) == 1

    second_type = get_type_hints(kw.Graph2D.fill_between)["second"]
    assert second_type is not int
    assert any("Callable" in str(option) for option in get_args(second_type))


def test_fill_between_sampleable_curves():
    upper = kw.ParametricCurve2D("t", "1-t^2", t_range=(-1, 1), samples=101)
    lower = kw.ParametricCurve2D("t", "t^2-1", t_range=(-1, 1), samples=101)
    graph = kw.Graph2D().fill_between(upper, lower, values=np.linspace(-1, 1, 51), color="gold")
    artist = graph.plot(show=False, return_artists=True)[0]
    assert artist in graph.ax.collections


def test_decorations_clear_with_the_graph():
    graph = kw.Graph2D().vertical_line(1).annotate("note", (0, 0)).fill_between(lambda x: x)
    assert len(graph._decorations) == 3
    graph.clear()
    assert graph._decorations == []


@pytest.mark.parametrize(
    "action",
    [
        lambda: kw.Graph2D().mark((1, 2, 3)),
        lambda: kw.Graph2D().annotate("bad", at=(1,)),
        lambda: kw.Graph2D().fill_between(object()).plot(show=False),
    ],
)
def test_invalid_annotations_raise_friendly_errors(action):
    with pytest.raises((TypeError, ValueError)):
        action()
