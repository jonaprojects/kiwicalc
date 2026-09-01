import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting import plots


@pytest.fixture(autouse=True)
def close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_scatter_dot_and_function_paths():
    with pytest.raises(ValueError):
        plots.scatter_dots([1], [1, 2], show=False)
    plots.scatter_dots([0, 1], [1, 2], title="dots", show=False)
    plots.scatter_dots_3d([0], [1], [2], title="3d", show=False, write_labels=True)
    plots.scatter_dots_3d([0], [1], [2], show=False, write_labels=False)
    plots.scatter_function("x^2", values=[-1, 0, 1], show=False)
    plots.scatter_function(lambda x: x + 1, start=0, stop=1, step=0.5, show=False)


def test_scatter_function_3d_mesh_and_exception_paths():
    mesh = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    plots.scatter_function_3d("x+y", meshgrid=mesh, show=False, title="surface")

    def partially_defined(x, y):
        if x == 1 and y == 1:
            raise ValueError("undefined")
        return x - y

    plots.scatter_function_3d(partially_defined, meshgrid=mesh, show=False, write_labels=False)


def test_plot_function_2d_title_and_values_paths():
    plots.plot_function(lambda x: x, values=[-1, 0, 1], title="x^2", formatText=True, show=False)
    plots.plot_function("x+1", start=0, stop=1, step=0.5, title="plain", formatText=False, show=False)
    fig, ax = plt.subplots()
    plots.plot_function(lambda x: x, values=[0, 1], fig=fig, ax=ax, show_axis=False, show=False)


def test_plot_function_3d_dispatch_and_validation_paths():
    mesh = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.warns(UserWarning):
        plots.plot_function_3d("x+y", step=0.01, meshgrid=mesh, show=False)
    plots.plot_function_3d(kw.Poly("x+y"), meshgrid=mesh, show=False, write_labels=False)
    with pytest.raises(ValueError):
        plots.plot_function_3d(kw.Poly("x"), meshgrid=mesh, show=False)

    def nullable(x, y):
        return None if x == y else x + y

    plots.plot_function_3d(nullable, meshgrid=mesh, show=False)

    def invalid(x, y):
        raise ValueError("outside domain")

    plots.plot_function_3d(invalid, meshgrid=mesh, show=False)


def test_multiple_function_plotting_dispatch_paths():
    plots.plot_functions(
        ["x", kw.Function("x^2"), kw.Poly("x+1"), lambda x: x - 1],
        start=0,
        stop=1,
        step=0.5,
        title="mixed",
        formatText=True,
        show=False,
        with_legend=True,
    )
    plots.plot_functions([lambda x: x], start=0, stop=1, step=0.5, show_axis=False, show=False, with_legend=False)
    plots.plot_functions([lambda x: x], start=0, stop=1, step=0.5, show=True, with_legend=False)
    plots.scatter_functions(["x", lambda x: x**2], start=0, stop=1, step=0.5, show=False)


def test_vector_complex_and_subplot_paths():
    plots.plot_vector_2d(0, 0, 1, 2, show=False)
    fig, ax = plt.subplots()
    plots.plot_vector_2d(0, 0, 1, 2, fig=fig, ax=ax, show=False)
    plots.plot_vector_3d((0, 0, 0), (1, 2, 3), show=False)
    fig = plt.figure()
    plots.plot_vector_3d((0, 0, 0), (1, 1, 1), fig=fig, ax=None, show=False)
    ax = plt.figure().add_subplot(111, projection="3d")
    plots.plot_vector_3d((0, 0, 0), (1, 1, 1), fig=None, ax=ax, show=True)
    plots.plot_complex(1 + 1j, 3 + 4j, title="complex", show=False)
    plots.plot_complex(1 + 0j, title="shown", show=True)
    assert plots.generate_subplot_shape(4) == (2, 2)
    assert plots.generate_subplot_shape(6) == (2, 3)
    assert plots.generate_subplot_shape(5) == (3, 3)


def test_plot_multiple_and_graph_objects():
    funcs = ["x", "x+1", "x+2", "x+3"]
    plots.plot_multiple(funcs, shape=(2, 2), start=0, stop=1, step=0.5, title="grid", show=False)
    plots.plot_multiple(funcs[:2], shape=(1, 3), start=0, stop=1, step=0.5, show_axis=False, show=False)
    with pytest.warns(UserWarning):
        plots.plot_multiple(funcs, shape=(2, 2), start=0, stop=1, step=0.5, show=True)

    fig, ax = plt.subplots()
    graph = plots.Graph([], fig, ax)
    assert graph.is_empty()
    graph.add(kw.Poly("x"))
    assert graph.items
    with pytest.raises(NotImplementedError):
        graph.plot()
    with pytest.raises(NotImplementedError):
        graph.scatter()

    graph2d = plots.Graph2D([kw.Poly("x"), kw.Poly("x^2"), kw.Poly("x^3")])
    graph2d.plot(start=0, stop=1, step=0.5, show=False)
    graph2d.plot(start=0, stop=1, step=0.5, text="custom", show_axis=False, show=False)
    graph2d.plot(start=0, stop=1, step=0.5, text="shown", show=True)


def test_plot_functions_3d_and_graph3d_paths():
    plots.plot_functions_3d([lambda x, y: x + y], start=0, stop=1, step=0.5)
    plots.scatter_functions_3d([lambda x, y: x - y], start=0, stop=1, step=0.5)
    graph = plots.Graph3D()
    graph.plot([lambda x, y: x * y], start=0, stop=1, step=0.5)
    graph.scatter([lambda x, y: x * y], start=0, stop=1, step=0.5)
