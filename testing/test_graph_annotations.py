import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_fill_between_functions_expressions_and_constants():
    graph = kw.Graph2D()
    graph.fill_between(lambda x: x * x, 0, values=[-1, 0, 1], label="area", color="skyblue", alpha=0.5)
    artists = graph.plot(show=False, return_artists=True)
    assert len(artists) == 1
    assert artists[0] in graph.ax.collections

    expression_graph = kw.Graph2D()
    expression_graph.fill_between(kw.Poly("x"), "f(x)=x^2", values=[0, 0.5, 1])
    assert len(expression_graph.plot(show=False, return_artists=True)) == 1


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

