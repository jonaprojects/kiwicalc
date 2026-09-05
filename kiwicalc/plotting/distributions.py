"""Friendly Matplotlib renderers for probability distributions.

The probability classes import this module lazily, keeping numerical use of
KiwiCalc independent from Matplotlib's import and start-up cost.
"""

from __future__ import annotations

import math
import re
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np

from kiwicalc.plotting.themes import apply_theme, get_theme


def _axes(fig=None, ax=None, *, projection=None, theme=None):
    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("fig and ax must refer to the same figure")
        if projection == "3d" and not hasattr(ax, "zaxis"):
            raise ValueError("kind='surface' requires a 3D axes")
        apply_theme(ax.figure, ax, get_theme(theme))
        return ax.figure, ax
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection=projection)
    apply_theme(fig, ax, get_theme(theme))
    return fig, ax


def _name(distribution):
    custom_name = getattr(distribution, "name", None)
    if custom_name:
        return custom_name
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", type(distribution).__name__)
    return words.capitalize()


def _title(value, default):
    return default if value is None else value


def _finish(fig, ax, *, title, xlabel, ylabel, theme, grid, label,
            legend, show):
    resolved = get_theme(theme)
    apply_theme(fig, ax, resolved)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid is not None:
        ax.grid(bool(grid))
    if legend is True or (legend is None and label):
        handles, labels = ax.get_legend_handles_labels()
        if handles and any(not item.startswith("_") for item in labels):
            ax.legend()
    if show:
        plt.show()


def _continuous_domain(distribution, start, stop, tail_probability):
    lower, upper = distribution.support
    if start is None:
        start = lower if math.isfinite(lower) else distribution.ppf(tail_probability)
    if stop is None:
        stop = upper if math.isfinite(upper) else distribution.ppf(1 - tail_probability)
    start, stop = float(start), float(stop)
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        raise ValueError("plot bounds must be finite and start must be smaller than stop")
    return start, stop


def _discrete_values(distribution, start, stop, tail_probability, max_points):
    from kiwicalc.probability.distributions import Categorical
    from kiwicalc.probability.formula_distributions import DiscreteFormulaDistribution

    if isinstance(distribution, Categorical):
        if start is not None or stop is not None:
            raise ValueError("start and stop are not used for categorical distributions")
        return np.asarray(distribution.values, dtype=object), np.arange(len(distribution.values))
    if isinstance(distribution, DiscreteFormulaDistribution):
        values = np.asarray(distribution.values, dtype=float)
        if start is not None:
            values = values[values >= float(start)]
        if stop is not None:
            values = values[values <= float(stop)]
        if values.size == 0:
            raise ValueError("plot bounds do not contain any support values")
        if values.size > max_points:
            raise ValueError(
                f"plot contains more than {max_points} outcomes; choose narrower start and stop bounds"
            )
        return values, values
    lower, upper = distribution.support
    if start is None:
        start = lower if math.isfinite(lower) else distribution.ppf(tail_probability)
    if stop is None:
        stop = upper if math.isfinite(upper) else distribution.ppf(1 - tail_probability)
    start, stop = int(math.ceil(float(start))), int(math.floor(float(stop)))
    if start > stop:
        raise ValueError("plot bounds do not contain any integer support values")
    if stop - start + 1 > max_points:
        raise ValueError(
            f"plot contains more than {max_points} outcomes; choose narrower start and stop bounds"
        )
    values = np.arange(start, stop + 1)
    return values, values


def _dimension_pair(distribution, dimensions):
    if distribution.dimension < 2:
        raise ValueError("a two-dimensional plot requires at least two dimensions")
    dimensions = (0, 1) if dimensions is None else tuple(dimensions)
    if len(dimensions) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral)
                                   for i in dimensions):
        raise ValueError("dimensions must contain exactly two integer indices")
    dimensions = tuple(int(index) for index in dimensions)
    if len(set(dimensions)) != 2 or min(dimensions) < 0 or max(dimensions) >= distribution.dimension:
        raise ValueError(f"dimensions must be distinct indices below {distribution.dimension}")
    return dimensions


def _encoded(values):
    labels = []
    mapping = {}
    encoded = []
    for value in values:
        key = value.item() if isinstance(value, np.generic) else value
        if key not in mapping:
            mapping[key] = len(labels)
            labels.append(key)
        encoded.append(mapping[key])
    return np.asarray(encoded, dtype=float), labels


def _numeric_or_encoded(values):
    try:
        return np.asarray(values, dtype=float), []
    except (TypeError, ValueError):
        return _encoded(values)


def plot_distribution(distribution, kind=None, *, start=None, stop=None,
                      xlim=None, ylim=None, points=300, dimensions=None,
                      levels=12, fill=False, colorbar=True, annotate=False,
                      tail_probability=0.001, max_discrete_points=1000,
                      fig=None, ax=None, show=True,
                      title=None, label=None, xlabel=None, ylabel=None,
                      theme=None, grid=None, legend=None, **style):
    """Plot a probability distribution and return its primary Matplotlib artist.

    Defaults are selected from the distribution: PDF lines for continuous
    variables, PMF bars for discrete variables, contours for supported
    continuous joint distributions, and heatmaps for finite joint tables.
    """
    from kiwicalc.probability.distributions import (
        ContinuousDistribution, DiscreteDistribution,
    )
    from kiwicalc.probability.multivariate import (
        IndependentJointDistribution, JointDiscreteDistribution,
        MultivariateDistribution, MultivariateNormal,
    )

    if not isinstance(points, int) or isinstance(points, bool) or points < 2:
        raise ValueError("points must be an integer of at least 2")
    if (not isinstance(max_discrete_points, Integral) or isinstance(max_discrete_points, bool)
            or max_discrete_points < 1):
        raise ValueError("max_discrete_points must be a positive integer")
    if not 0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be between 0 and 0.5")
    if kind is not None and not isinstance(kind, str):
        raise TypeError("kind must be a string or None")

    if isinstance(distribution, ContinuousDistribution):
        kind = "pdf" if kind is None else kind.lower()
        if kind not in {"pdf", "cdf"}:
            raise ValueError("continuous distributions support kind='pdf' or kind='cdf'")
        start, stop = _continuous_domain(distribution, start, stop, tail_probability)
        values = np.linspace(start, stop, points)
        result = np.asarray(distribution.pdf(values) if kind == "pdf" else distribution.cdf(values))
        fig, ax = _axes(fig, ax, theme=theme)
        artist, = ax.plot(values, result, label=label, **style)
        if fill:
            ax.fill_between(values, 0, result, alpha=0.22,
                            color=artist.get_color())
        _finish(fig, ax, title=_title(title, f"{_name(distribution)} {kind.upper()}"),
                xlabel=xlabel or "x", ylabel=ylabel or ("Density" if kind == "pdf" else
                "Cumulative probability"), theme=theme, grid=grid, label=label,
                legend=legend, show=show)
        return artist

    if isinstance(distribution, DiscreteDistribution):
        kind = "pmf" if kind is None else kind.lower()
        if kind not in {"pmf", "cdf"}:
            raise ValueError("discrete distributions support kind='pmf' or kind='cdf'")
        values, positions = _discrete_values(
            distribution, start, stop, tail_probability, int(max_discrete_points)
        )
        probabilities = np.asarray(distribution.pmf(values) if kind == "pmf" else
                                   distribution.cdf(values), dtype=float)
        fig, ax = _axes(fig, ax, theme=theme)
        if kind == "pmf":
            artist = ax.bar(positions, probabilities, label=label, **style)
        else:
            artist = ax.step(positions, probabilities, where="post", label=label, **style)[0]
        if values.dtype == object:
            ax.set_xticks(positions, [str(value) for value in values])
        _finish(fig, ax, title=_title(title, f"{_name(distribution)} {kind.upper()}"),
                xlabel=xlabel or "Outcome", ylabel=ylabel or ("Probability" if kind == "pmf"
                else "Cumulative probability"), theme=theme, grid=grid, label=label,
                legend=legend, show=show)
        return artist

    if not isinstance(distribution, MultivariateDistribution):
        raise TypeError("distribution must be a KiwiCalc probability distribution")

    if distribution.dimension == 1:
        if isinstance(distribution, IndependentJointDistribution):
            return plot_distribution(distribution.components[0], kind, start=start, stop=stop,
                                     points=points, fill=fill, fig=fig, ax=ax, show=show,
                                     title=title, label=label, xlabel=xlabel, ylabel=ylabel,
                                     theme=theme, grid=grid, legend=legend, **style)
        if isinstance(distribution, JointDiscreteDistribution):
            return plot_distribution(distribution.marginal(0), kind, fig=fig, ax=ax, show=show,
                                     title=title, label=label, xlabel=xlabel, ylabel=ylabel,
                                     theme=theme, grid=grid, legend=legend, **style)

    pair = _dimension_pair(distribution, dimensions)
    if isinstance(distribution, JointDiscreteDistribution):
        kind = "heatmap" if kind is None else kind.lower()
        if kind not in {"heatmap", "bubble"}:
            raise ValueError("finite joint distributions support kind='heatmap' or kind='bubble'")
        selected = distribution.marginal(pair)
        x_unique = list(dict.fromkeys(outcome[0] for outcome in selected.outcomes))
        y_unique = list(dict.fromkeys(outcome[1] for outcome in selected.outcomes))
        x_map, y_map = {v: i for i, v in enumerate(x_unique)}, {v: i for i, v in enumerate(y_unique)}
        x_values = np.asarray([x_map[o[0]] for o in selected.outcomes])
        y_values = np.asarray([y_map[o[1]] for o in selected.outcomes])
        probabilities = np.asarray(selected.probabilities)
        fig, ax = _axes(fig, ax, theme=theme)
        if kind == "bubble":
            sizes = style.pop("s", 1600 * probabilities)
            artist = ax.scatter(x_values, y_values, s=sizes, c=probabilities, **style)
        else:
            matrix = np.zeros((len(y_unique), len(x_unique)))
            matrix[y_values, x_values] = probabilities
            artist = ax.imshow(matrix, origin="lower", aspect="auto", **style)
            if annotate:
                for row, column in np.ndindex(matrix.shape):
                    ax.text(column, row, f"{matrix[row, column]:.3g}", ha="center", va="center")
        ax.set_xticks(range(len(x_unique)), [str(value) for value in x_unique])
        ax.set_yticks(range(len(y_unique)), [str(value) for value in y_unique])
        if colorbar:
            fig.colorbar(artist, ax=ax, label="Probability")
        _finish(fig, ax, title=_title(title, f"{_name(distribution)} PMF"),
                xlabel=xlabel or f"Dimension {pair[0]}", ylabel=ylabel or f"Dimension {pair[1]}",
                theme=theme, grid=grid, label=label, legend=False, show=show)
        return artist

    density = None
    if isinstance(distribution, MultivariateNormal):
        density = distribution.marginal(pair)
    elif isinstance(distribution, IndependentJointDistribution) and not distribution.is_discrete:
        density = distribution.marginal(pair)
    if density is None:
        if kind not in {None, "samples", "scatter"}:
            raise ValueError("this distribution supports kind='samples'; use scatter() for sample clouds")
        return scatter_distribution(distribution, dimensions=pair, fig=fig, ax=ax, show=show,
                                    title=title, label=label, xlabel=xlabel, ylabel=ylabel,
                                    theme=theme, grid=grid, legend=legend, **style)

    kind = "contour" if kind is None else kind.lower()
    if kind not in {"contour", "contourf", "surface"}:
        raise ValueError("continuous joint distributions support contour, contourf, or surface")
    components = (density.marginal(0), density.marginal(1)) if isinstance(
        density, MultivariateNormal) else density.components
    x_bounds = xlim or _continuous_domain(components[0], None, None, tail_probability)
    y_bounds = ylim or _continuous_domain(components[1], None, None, tail_probability)
    x = np.linspace(*x_bounds, points)
    y = np.linspace(*y_bounds, points)
    xx, yy = np.meshgrid(x, y)
    values = density.pdf(np.stack((xx, yy), axis=-1))
    fig, ax = _axes(fig, ax, projection="3d" if kind == "surface" else None,
                    theme=theme)
    if kind == "surface":
        artist = ax.plot_surface(xx, yy, values, **style)
        ax.set_zlabel("Density")
    else:
        renderer = ax.contourf if kind == "contourf" else ax.contour
        artist = renderer(xx, yy, values, levels=levels, **style)
        if colorbar:
            fig.colorbar(artist, ax=ax, label="Density")
    _finish(fig, ax, title=_title(title, f"{_name(distribution)} density"),
            xlabel=xlabel or f"Dimension {pair[0]}", ylabel=ylabel or f"Dimension {pair[1]}",
            theme=theme, grid=grid, label=label, legend=False, show=show)
    return artist


def scatter_distribution(distribution, *, size=500, values=None, dimensions=None,
                         jitter=0.04, random_state=None, fig=None, ax=None,
                         show=True, title=None, label=None, xlabel=None, ylabel=None,
                         theme=None, grid=None, legend=None, **style):
    """Scatter random samples (or supplied ``values``) from a distribution."""
    from kiwicalc.probability.distributions import Distribution
    from kiwicalc.probability.multivariate import MultivariateDistribution

    if not isinstance(size, Integral) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    size = int(size)
    if not isinstance(distribution, (Distribution, MultivariateDistribution)):
        raise TypeError("distribution must be a KiwiCalc probability distribution")
    samples = np.asarray(distribution.sample(size, random_state=random_state)
                         if values is None else values)
    fig, ax = _axes(fig, ax, theme=theme)
    if isinstance(distribution, MultivariateDistribution) and distribution.dimension >= 2:
        pair = _dimension_pair(distribution, dimensions)
        if samples.ndim != 2 or samples.shape[1] != distribution.dimension:
            raise ValueError(f"values must have shape (n, {distribution.dimension})")
        x, x_labels = _numeric_or_encoded(samples[:, pair[0]])
        y, y_labels = _numeric_or_encoded(samples[:, pair[1]])
        artist = ax.scatter(x, y, label=label, **style)
        if x_labels:
            ax.set_xticks(range(len(x_labels)), [str(value) for value in x_labels])
        if y_labels:
            ax.set_yticks(range(len(y_labels)), [str(value) for value in y_labels])
        default_x, default_y = f"Dimension {pair[0]}", f"Dimension {pair[1]}"
    else:
        samples = samples.reshape(-1)
        x, labels = _numeric_or_encoded(samples)
        generator = np.random.default_rng(random_state)
        y = generator.normal(0, jitter, x.size) if jitter else np.zeros(x.size)
        artist = ax.scatter(x, y, label=label, **style)
        if labels:
            ax.set_xticks(range(len(labels)), [str(value) for value in labels])
        ax.set_yticks([])
        default_x, default_y = "Sample value", None
    _finish(fig, ax, title=_title(title, f"Samples from {_name(distribution)}"),
            xlabel=xlabel or default_x, ylabel=ylabel or default_y, theme=theme,
            grid=grid, label=label, legend=legend, show=show)
    return artist


__all__ = ["plot_distribution", "scatter_distribution"]
