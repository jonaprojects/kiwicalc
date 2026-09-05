import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_fixed_sampling_remains_the_plotting_default():
    line = kw.plot_function(lambda x: x, start=0, stop=0.02, step=0.01, show=False)

    assert list(line.get_xdata()) == pytest.approx([0, 0.01, 0.02])
    assert not hasattr(line, "kiwicalc_sample")


def test_adaptive_sampling_resolves_a_uniform_grid_alias():
    function = lambda x: np.sin(100 * np.pi * x)
    fixed = kw.sample_for_plot(function, 0, 0.2, 0.01, sampling="fixed")
    adaptive = kw.sample_for_plot(function, 0, 0.2, 0.01, sampling="adaptive")

    assert np.max(np.abs(fixed.y)) < 1e-10
    assert np.max(np.abs(adaptive.y)) == pytest.approx(1)
    assert adaptive.point_count > fixed.point_count
    assert adaptive.evaluations == adaptive.point_count
    assert adaptive.refined_points == adaptive.point_count - adaptive.initial_points
    assert adaptive.truncated is False


def test_adaptive_sampling_finds_a_peak_between_fixed_points():
    function = lambda x: np.exp(-((x - 0.005) / 0.0004) ** 2)

    sample = kw.sample_for_plot(
        function, -0.02, 0.02, 0.01, sampling="adaptive",
    )

    assert np.max(sample.y) == pytest.approx(1)
    assert np.min(np.abs(sample.x - 0.005)) < 1e-12


def test_adaptive_sampling_breaks_a_pole_and_reports_depth_limit():
    function = lambda x: 1 / (x - 0.005)

    sample = kw.sample_for_plot(
        function, -0.02, 0.02, 0.01, sampling="adaptive",
    )

    assert sample.discontinuities >= 1
    assert np.isnan(sample.y).any()
    assert sample.truncated is True


def test_adaptive_sampler_has_hard_point_limits():
    sample = kw.sample_for_plot(
        lambda x: np.sin(100 * np.pi * x), 0, 0.2, 0.01,
        sampling="adaptive", max_points=50,
    )

    assert sample.point_count == 50
    assert sample.truncated is True
    with pytest.raises(ValueError, match="initial sample count"):
        kw.sample_for_plot(lambda x: x, 0, 1, 0.01, sampling="adaptive", max_points=10)


@pytest.mark.parametrize("sampling", ["unknown", "", 1])
def test_sampling_mode_is_validated(sampling):
    expected = TypeError if sampling == 1 else ValueError
    with pytest.raises(expected, match="fixed.*adaptive"):
        kw.sample_for_plot(lambda x: x, 0, 1, sampling=sampling)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"tolerance": 0}, "tolerance"),
        ({"max_points": 1}, "at least 2"),
        ({"max_depth": -1}, "non-negative"),
        ({"values": [0, 1, 0.5]}, "strictly increasing"),
    ],
)
def test_adaptive_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        kw.sample_for_plot(lambda x: x, 0, 1, sampling="adaptive", **options)


def test_adaptive_options_are_irrelevant_to_fixed_sampling():
    sample = kw.sample_for_plot(
        lambda x: x, 0, 1, sampling="fixed",
        tolerance=0, max_points=1, max_depth=-1,
    )

    assert sample.sampling == "fixed"


def test_public_fixed_sampler_preserves_explicit_coordinate_order():
    sample = kw.sample_for_plot(lambda x: x * x, values=[2, -1, 2], sampling=" fixed ")

    assert list(sample.x) == [2, -1, 2]
    assert list(sample.y) == [4, 1, 4]
    assert sample.initial_points == 3
    assert sample.point_count == 3
    assert sample.evaluations == 2


def test_standalone_plot_and_scatter_expose_sampling_diagnostics():
    function = lambda x: np.sin(100 * np.pi * x)
    line = kw.plot_function(
        function, start=0, stop=0.2, step=0.01,
        sampling="adaptive", show=False,
    )
    dots = kw.scatter_function(
        function, start=0, stop=0.2, step=0.01,
        sampling="adaptive", show=False,
    )

    assert line.kiwicalc_sample.sampling == "adaptive"
    assert dots.kiwicalc_sample.sampling == "adaptive"
    assert len(line.get_xdata()) == line.kiwicalc_sample.point_count


def test_multi_function_helpers_sample_each_function_independently():
    functions = [lambda x: x, lambda x: np.sin(100 * np.pi * x)]
    lines = kw.plot_functions(
        functions, start=0, stop=0.2, step=0.01,
        sampling="adaptive", show=False,
    )
    dots = kw.scatter_functions(
        functions, start=0, stop=0.2, step=0.01,
        sampling="adaptive", show=False,
    )

    assert len(lines[0].get_xdata()) < len(lines[1].get_xdata())
    assert dots[0].kiwicalc_sample.point_count < dots[1].kiwicalc_sample.point_count


def test_plot_multiple_supports_adaptive_sampling():
    _, axes = kw.plot_multiple(
        [lambda x: x, lambda x: np.sin(100 * np.pi * x)],
        shape=(2, 1), start=0, stop=0.2, step=0.01,
        sampling="adaptive", show=False,
    )

    assert len(axes[0, 0].lines[0].get_xdata()) < len(axes[1, 0].lines[0].get_xdata())
    assert axes[1, 0].lines[0].kiwicalc_sample.sampling == "adaptive"


def test_graph_plot_and_scatter_retain_latest_sampling_results():
    graph = kw.Graph2D().add(lambda x: np.sin(100 * np.pi * x))

    graph.plot(
        start=0, stop=0.2, step=0.01, sampling="adaptive", show=False,
    )
    first = graph.sampling_results
    graph.scatter(
        start=0, stop=0.2, step=0.01, sampling="adaptive", show=False,
    )

    assert len(first) == 1
    assert first[0].sampling == "adaptive"
    assert len(graph.sampling_results) == 1
    assert graph.sampling_results[0].point_count == len(graph.ax.collections[0].get_offsets())


def test_public_sampling_result_contract():
    sample = kw.sample_for_plot("x^2", values=[-1, 0, 1], sampling="adaptive")

    assert isinstance(sample, kw.PlotSample)
    assert sample.x.shape == sample.y.shape
    assert sample.point_count == len(sample.x)
