import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting.explanations import roots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_standalone_helpers_honor_non_current_axes_and_return_artists():
    fig, ax = plt.subplots()
    plt.figure()

    dots = kw.scatter_dots([0, 1], [1, 2], fig=fig, ax=ax, show=False)
    line = kw.plot_function(lambda x: x, values=[0, 1], ax=ax, show=False)

    assert dots.axes is ax
    assert line.axes is ax
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1


def test_multi_scatter_honors_show_false_and_returns_each_artist(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    artists = kw.scatter_functions([lambda x: x, lambda x: x * x], show=False)

    assert len(artists) == 2
    assert shown == []


def test_3d_scatter_validates_lengths_meshes_and_steps():
    with pytest.raises(ValueError, match="equal lengths"):
        kw.scatter_dots_3d([0], [0, 1], [0], show=False)
    with pytest.raises(ValueError, match="matching shapes"):
        kw.scatter_function_3d(lambda x, y: x + y, meshgrid=(np.zeros((2, 2)), np.zeros(3)), show=False)
    with pytest.raises(ValueError, match="positive"):
        kw.plot_function_3d(lambda x, y: x + y, step=0, show=False)


@pytest.mark.parametrize(
    "helper, kwargs",
    [
        (kw.plot_function, {}),
        (kw.scatter_function, {}),
        (kw.plot_functions, {"functions": [lambda x: x]}),
        (kw.scatter_functions, {"functions": [lambda x: x]}),
    ],
)
def test_2d_sampling_helpers_reject_nonpositive_steps(helper, kwargs):
    function_args = () if kwargs else (lambda x: x,)
    with pytest.raises(ValueError, match="positive finite"):
        helper(*function_args, step=0, show=False, **kwargs)


def test_plot_multiple_handles_single_and_column_layouts():
    single_fig, single_axes = kw.plot_multiple([lambda x: x], values=[0, 1], show=False)
    column_fig, column_axes = kw.plot_multiple(
        [lambda x: x, lambda x: x * x], shape=(2, 1), values=[0, 1],
        title="Functions", show=False,
    )

    assert single_axes.shape == (1, 1)
    assert len(single_fig.axes) == 1
    assert column_axes.shape == (2, 1)
    assert column_fig._suptitle.get_text() == "Functions"
    assert [axis.get_title() for axis in column_axes.flat] == ["", ""]


def test_plot_multiple_uses_only_friendly_or_explicit_subplot_titles():
    formula_fig, formula_axes = kw.plot_multiple(
        ["x", kw.Function("x^2")], values=[0, 1], show=False,
    )
    named_fig, named_axes = kw.plot_multiple(
        [lambda x: x, lambda x: x*x], values=[0, 1], show=False,
        subplot_titles=("Linear", "Quadratic"),
    )

    assert [axis.get_title() for axis in formula_axes.flat][:2] == ["x", "f(x)=x^2"]
    assert [axis.get_title() for axis in named_axes.flat][:2] == ["Linear", "Quadratic"]
    assert all("lambda" not in axis.get_title() for axis in named_axes.flat)

    plt.close(formula_fig)
    plt.close(named_fig)


def test_plot_multiple_rejects_empty_and_undersized_layouts():
    with pytest.raises(ValueError, match="at least one"):
        kw.plot_multiple([], show=False)
    with pytest.raises(ValueError, match="room for every"):
        kw.plot_multiple([lambda x: x, lambda x: x], shape=(1, 1), show=False)


def test_graph_replot_and_clear_remove_owned_artists_and_titles():
    graph = kw.Graph2D().add(lambda x: x, label="line")
    graph.plot(show=False, title="First", legend=True)
    graph.plot(show=False, title="", legend=False)

    assert len(graph.ax.lines) == 1
    assert graph.ax.get_title() == ""
    assert graph.ax.get_legend() is None

    graph.clear()
    assert len(graph.ax.lines) == 0
    assert graph.ax.get_title() == ""


def test_graph2d_scatter_replaces_sampled_lines_without_accumulating(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    graph = kw.Graph2D().add(lambda x: x, label="line")

    artists = graph.scatter(values=[0, 1], show=True, return_artists=True)
    graph.scatter(values=[0, 1], show=False)

    assert len(artists) == 1
    assert artists[0].__class__.__name__ == "PathCollection"
    assert len(graph.ax.lines) == 0
    assert len(graph.ax.collections) == 1
    assert shown == [True]


@pytest.mark.parametrize("helper", [kw.scatter_functions_3d, kw.plot_functions_3d])
def test_multi_3d_helpers_validate_step(helper):
    with pytest.raises(ValueError, match="positive"):
        helper([lambda x, y: x + y], step=0, show=False)


def test_graph3d_is_lazy_and_vector_styles_are_flattened():
    before = len(plt.get_fignums())
    graph = kw.Graph3D().add(kw.Vector3D(1, 2, 3), label="v", color="red")
    assert len(plt.get_fignums()) == before

    artists = graph.plot(show=False, legend=True, return_artists=True)
    graph.plot(show=False)

    assert len(artists) == 1
    assert not isinstance(artists[0], list)
    assert len(graph.ax.collections) == 1


def test_unrendered_graph_serializes_theme_and_axis_configuration():
    graph = kw.Graph2D().theme("classroom").configure_axes(
        xlabel="Distance", units=("m", None), minor_ticks=True,
    )

    payload = graph.to_dict()
    restored = kw.Graph.from_dict(payload)
    restored.plot(show=False)

    assert payload["version"] == 1
    assert restored._theme.name == "classroom"
    assert restored.ax.get_xlabel() == "Distance (m)"


def test_root_detection_rejects_near_zero_minima_but_keeps_even_roots():
    values = np.linspace(-1, 1, 1201)

    assert roots(lambda x: (x - 0.12345) ** 2 + 1e-7, values) == []
    found = roots(lambda x: (x - 0.12345) ** 2, values)
    assert found == pytest.approx([0.12345], abs=1e-8)


def test_integral_overlay_does_not_bridge_function_gaps():
    values = np.linspace(-1, 1, 5)
    graph = kw.Graph2D().show_integral(lambda x: 1 / x)
    artist = graph.plot(values=values, show=False, return_artists=True)[0]
    result = np.asarray(artist.get_ydata())

    assert np.isnan(result[2])
    assert result[0] == pytest.approx(0)
    assert result[3] == pytest.approx(0)


def test_interaction_envelope_includes_interior_parameter_values():
    control = kw.Graph2D().interact(
        lambda x, a: np.sin(2 * np.pi * a) * x,
        (0, 2), initial=1, show=False,
    )
    control.set_value(0.25)

    assert control.ax.get_ylim()[0] < -10
    assert control.ax.get_ylim()[1] > 10
    with pytest.raises(ValueError, match="inside"):
        control.set_value(3)


@pytest.mark.parametrize(
    "theme",
    [
        kw.PlotTheme,
        lambda: kw.PlotTheme(font_size=-1),
        lambda: kw.PlotTheme(grid_alpha=2),
        lambda: kw.PlotTheme(color_cycle=()),
        lambda: kw.PlotTheme(foreground="not-a-color"),
    ],
)
def test_theme_values_are_validated(theme):
    if theme is kw.PlotTheme:
        assert theme().font_size > 0
    else:
        with pytest.raises(ValueError):
            theme()


def test_distribution_ignores_irrelevant_sampling_counts_and_validates_jitter():
    assert kw.Binomial(2).plot(points=1, show=False) is not None
    assert kw.Normal().scatter(values=[1, 2], size=0, jitter=0, show=False) is not None
    with pytest.raises(ValueError, match="non-negative finite"):
        kw.Normal().scatter(values=[1, 2], jitter=-1, show=False)
