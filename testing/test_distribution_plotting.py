import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.container import BarContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_continuous_plot_has_friendly_defaults_and_optional_fill():
    distribution = kw.Normal(mean=2, std=3)
    artist = distribution.plot(show=False, fill=True, label="density", theme="colorblind")

    assert isinstance(artist, Line2D)
    assert artist.axes.get_title() == "Normal PDF"
    assert artist.axes.get_ylabel() == "Density"
    assert artist.get_label() == "density"
    assert len(artist.axes.collections) == 1
    assert artist.get_color() == kw.THEMES["colorblind"].color_cycle[0]
    assert min(artist.get_xdata()) < distribution.mean < max(artist.get_xdata())


def test_continuous_cdf_accepts_bounds_and_existing_axes():
    fig, ax = plt.subplots()
    artist = kw.Exponential(2).plot(
        "cdf", start=0, stop=3, points=31, fig=fig, ax=ax, show=False,
        title="Waiting time", xlabel="seconds", ylabel="chance", grid=False,
    )

    assert artist.axes is ax
    assert len(artist.get_xdata()) == 31
    assert ax.get_title() == "Waiting time"
    assert ax.get_xlabel() == "seconds"
    assert ax.get_ylabel() == "chance"


def test_discrete_pmf_and_cdf_render_expected_artists():
    pmf = kw.Poisson(3).plot(stop=9, show=False)
    cdf = kw.Binomial(4, 0.5).plot("cdf", show=False)

    assert isinstance(pmf, BarContainer)
    assert len(pmf.patches) == 10
    assert isinstance(cdf, Line2D)
    assert np.allclose(cdf.get_ydata(), kw.Binomial(4, 0.5).cdf(np.arange(5)))


def test_categorical_plot_uses_readable_tick_labels():
    distribution = kw.Categorical({"low": 0.2, "medium": 0.5, "high": 0.3})
    artist = distribution.plot(show=False, color="teal")

    assert isinstance(artist, BarContainer)
    assert [tick.get_text() for tick in artist.patches[0].axes.get_xticklabels()] == [
        "low", "medium", "high"
    ]


@pytest.mark.parametrize("kind", ["contour", "contourf"])
def test_multivariate_normal_density_contours(kind):
    distribution = kw.MultivariateNormal([0, 1, 2], [[1, 0.2, 0], [0.2, 2, 0], [0, 0, 3]])
    artist = distribution.plot(
        kind, dimensions=(2, 0), points=25, levels=5,
        xlim=(-4, 8), ylim=(-3, 3), show=False, colorbar=False,
    )

    assert artist.axes.get_title() == "Multivariate normal density"
    assert artist.axes.get_xlabel() == "Dimension 2"
    assert artist.axes.get_ylabel() == "Dimension 0"


def test_independent_continuous_joint_surface():
    distribution = kw.IndependentJointDistribution(kw.Normal(), kw.Uniform(-2, 2))
    artist = distribution.plot("surface", points=15, show=False, cmap="viridis")

    assert isinstance(artist, PolyCollection)
    assert hasattr(artist.axes, "zaxis")
    assert artist.axes.get_zlabel() == "Density"


def test_finite_joint_heatmap_aggregates_unselected_dimensions_and_annotates():
    distribution = kw.JointDiscreteDistribution({
        ("rain", "walk", 0): 0.1,
        ("rain", "stay", 1): 0.2,
        ("sun", "walk", 0): 0.3,
        ("sun", "stay", 1): 0.4,
    })
    artist = distribution.plot(dimensions=(0, 1), annotate=True, show=False)

    assert isinstance(artist, AxesImage)
    assert np.allclose(artist.get_array(), [[0.1, 0.3], [0.2, 0.4]])
    assert len(artist.axes.texts) == 4
    assert len(artist.figure.axes) == 2  # plot plus colorbar


def test_finite_joint_bubble_plot():
    distribution = kw.JointDiscreteDistribution({(0, 0): 0.25, (1, 1): 0.75})
    artist = distribution.plot("bubble", show=False, colorbar=False, s=80)

    assert isinstance(artist, PathCollection)
    assert np.all(artist.get_sizes() == 80)


def test_generic_multivariate_plot_falls_back_to_samples():
    artist = kw.Dirichlet([2, 3, 4]).plot(
        dimensions=(0, 2), show=False, random_state=4, s=8
    )

    assert isinstance(artist, PathCollection)
    assert artist.get_offsets().shape == (500, 2)
    assert artist.axes.get_title() == "Samples from Dirichlet"


def test_univariate_scatter_supports_numeric_and_categorical_samples():
    numeric = kw.Normal().scatter(
        values=[-1, 0, 1], jitter=0, show=False, label="draws", legend=True
    )
    categorical = kw.Categorical({"A": 0.5, "B": 0.5}).scatter(
        values=["A", "B", "A"], random_state=2, show=False
    )

    assert np.allclose(numeric.get_offsets()[:, 1], 0)
    assert numeric.axes.get_legend() is not None
    assert [tick.get_text() for tick in categorical.axes.get_xticklabels()] == ["A", "B"]


def test_multivariate_scatter_supports_supplied_named_and_numeric_values():
    named = kw.JointDiscreteDistribution({("A", "yes"): 0.5, ("B", "no"): 0.5})
    named_artist = named.scatter(values=[["A", "yes"], ["B", "no"]], show=False)
    numeric_artist = kw.Multinomial(3, [0.2, 0.3, 0.5]).scatter(
        values=[[0, 1, 2], [1, 1, 1]], dimensions=(2, 0), show=False
    )

    assert [tick.get_text() for tick in named_artist.axes.get_xticklabels()] == ["A", "B"]
    assert np.allclose(numeric_artist.get_offsets(), [[2, 0], [1, 1]])


def test_one_dimensional_joint_distributions_delegate_to_univariate_plot():
    independent = kw.IndependentJointDistribution(kw.Normal())
    finite = kw.JointDiscreteDistribution({("yes",): 0.6, ("no",): 0.4})

    assert isinstance(independent.plot(show=False), Line2D)
    assert isinstance(finite.plot(show=False), BarContainer)


def test_standalone_functions_are_exported():
    line = kw.plot_distribution(kw.Uniform(), show=False)
    points = kw.scatter_distribution(kw.Bernoulli(), size=7, random_state=1, show=False)

    assert isinstance(line, Line2D)
    assert points.get_offsets().shape == (7, 2)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: kw.Normal().plot("pmf", show=False), "kind='pdf'"),
        (lambda: kw.Binomial(2).plot("pdf", show=False), "kind='pmf'"),
        (lambda: kw.Normal().plot(points=1, show=False), "at least 2"),
        (lambda: kw.Normal().plot(tail_probability=0.8, show=False), "between 0 and 0.5"),
        (lambda: kw.Categorical({"a": 1}).plot(start=0, show=False), "not used"),
        (lambda: kw.Normal().plot(start=2, stop=1, show=False), "start must be smaller"),
        (lambda: kw.Poisson(2).plot(start=3, stop=1, show=False), "do not contain"),
        (lambda: kw.DiscreteUniform(1, 20).plot(max_discrete_points=10, show=False),
         "more than 10 outcomes"),
        (lambda: kw.Binomial(2).plot(max_discrete_points=0, show=False), "positive integer"),
        (lambda: kw.Dirichlet([2, 2, 2]).plot("contour", show=False), "kind='samples'"),
        (lambda: kw.MultivariateNormal([0, 0], np.eye(2)).plot(dimensions=(0, 0), show=False),
         "distinct indices"),
        (lambda: kw.Normal().scatter(size=0, show=False), "positive integer"),
        (lambda: kw.MultivariateNormal([0, 0], np.eye(2)).scatter(
            values=[[1, 2, 3]], show=False), "shape"),
    ],
)
def test_validation_errors_are_clear(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_axes_validation_and_invalid_objects():
    first, first_ax = plt.subplots()
    second, _ = plt.subplots()
    with pytest.raises(ValueError, match="same figure"):
        kw.Normal().plot(fig=second, ax=first_ax, show=False)
    with pytest.raises(ValueError, match="3D axes"):
        kw.MultivariateNormal([0, 0], np.eye(2)).plot(
            "surface", ax=first_ax, show=False
        )
    with pytest.raises(TypeError, match="KiwiCalc probability"):
        kw.plot_distribution(object(), show=False)
    with pytest.raises(TypeError, match="KiwiCalc probability"):
        kw.scatter_distribution(object(), show=False)
    with pytest.raises(TypeError, match="kind must be"):
        kw.Normal().plot(3, show=False)
