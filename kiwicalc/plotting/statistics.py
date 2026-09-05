"""Matplotlib renderers for statistical diagnostics."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from kiwicalc.plotting.distributions import _axes, _finish
from kiwicalc.probability.diagnostics import (
    ECDFResult, PPData, QQData, _reference_distribution, _sample,
    ecdf, pp_data, qq_data,
)


def _distribution_name(distribution):
    custom = getattr(distribution, 'name', None)
    if custom:
        return custom
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', type(distribution).__name__)


def plot_ecdf(data, *, nan_policy='omit', fig=None, ax=None, show=True,
              title=None, label=None, xlabel='Value', ylabel='Cumulative probability',
              theme=None, grid=None, legend=None, **style):
    """Plot an empirical cumulative distribution and return its axes."""
    result = data if isinstance(data, ECDFResult) else ecdf(data, nan_policy=nan_policy)
    fig, ax = _axes(fig, ax, theme=theme)
    style.setdefault('where', 'post')
    ax.step(result.values, result.probabilities, label=label, **style)
    ax.set_ylim(-0.02, 1.02)
    _finish(fig, ax, title='Empirical cumulative distribution' if title is None else title,
            xlabel=xlabel, ylabel=ylabel, theme=theme, grid=grid, label=label,
            legend=legend, show=show)
    return ax


def qq_plot(data, distribution=None, *, nan_policy='omit', reference=True,
            fig=None, ax=None, show=True, title=None, label=None,
            xlabel='Theoretical quantiles', ylabel='Observed quantiles',
            theme=None, grid=None, legend=None, **style):
    """Create a Q-Q plot against a fitted Normal or supplied distribution."""
    result = data if isinstance(data, QQData) else qq_data(
        data, distribution, nan_policy=nan_policy,
    )
    fig, ax = _axes(fig, ax, theme=theme)
    style.setdefault('s', 28)
    ax.scatter(result.theoretical, result.observed, label=label, **style)
    if reference:
        lower, upper = float(np.min(result.theoretical)), float(np.max(result.theoretical))
        ax.plot([lower, upper], [result.reference_intercept + result.reference_slope * lower,
                                result.reference_intercept + result.reference_slope * upper],
                color='#555555', linestyle='--', linewidth=1.5, label='Reference')
    default = f'{_distribution_name(result.distribution)} Q-Q plot'
    _finish(fig, ax, title=default if title is None else title,
            xlabel=xlabel, ylabel=ylabel, theme=theme, grid=grid, label=label,
            legend=legend, show=show)
    return ax


def pp_plot(data, distribution=None, *, nan_policy='omit', reference=True,
            fig=None, ax=None, show=True, title=None, label=None,
            xlabel='Theoretical probability', ylabel='Empirical probability',
            theme=None, grid=None, legend=None, **style):
    """Create a P-P plot against a fitted Normal or supplied distribution."""
    result = data if isinstance(data, PPData) else pp_data(
        data, distribution, nan_policy=nan_policy,
    )
    fig, ax = _axes(fig, ax, theme=theme)
    style.setdefault('s', 28)
    ax.scatter(result.theoretical, result.empirical, label=label, **style)
    if reference:
        ax.plot([0, 1], [0, 1], color='#555555', linestyle='--',
                linewidth=1.5, label='Reference')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    default = f'{_distribution_name(result.distribution)} P-P plot'
    _finish(fig, ax, title=default if title is None else title,
            xlabel=xlabel, ylabel=ylabel, theme=theme, grid=grid, label=label,
            legend=legend, show=show)
    return ax


def histogram_plot(data, *, bins='auto', fit=None, nan_policy='omit',
                   points=300, fig=None, ax=None, show=True, title=None,
                   label=None, xlabel='Value', ylabel='Density', theme=None,
                   grid=None, legend=None, alpha=0.65, **style):
    """Plot a density histogram with an optional fitted or supplied density.

    Use ``fit='normal'`` to estimate a Normal distribution from the sample, or
    pass a continuous KiwiCalc distribution to overlay it without fitting.
    """
    values, _ = _sample(data, nan_policy=nan_policy)
    if not isinstance(points, int) or isinstance(points, bool) or points < 2:
        raise ValueError('points must be an integer of at least 2')
    fig, ax = _axes(fig, ax, theme=theme)
    ax.hist(values, bins=bins, density=True, alpha=alpha, label=label, **style)
    fitted = None
    if fit is not None:
        fitted, was_fitted = _reference_distribution(values, fit)
        lower, upper = float(np.min(values)), float(np.max(values))
        if lower == upper:
            raise ValueError('a density overlay requires non-constant data')
        x = np.linspace(lower, upper, points)
        description = f'Fitted {_distribution_name(fitted)}' if was_fitted else _distribution_name(fitted)
        ax.plot(x, fitted.pdf(x), linewidth=2, label=description)
    default = 'Distribution of observations'
    if fitted is not None:
        default += f' with {_distribution_name(fitted)} density'
    _finish(fig, ax, title=default if title is None else title,
            xlabel=xlabel, ylabel=ylabel, theme=theme, grid=grid, label=label,
            legend=True if fitted is not None and legend is None else legend,
            show=show)
    return ax


def _interval_arrays(intervals):
    from kiwicalc.probability.inference import ConfidenceInterval

    if isinstance(intervals, ConfidenceInterval):
        items = [intervals]
    else:
        try:
            items = list(intervals)
        except TypeError as exc:
            raise TypeError('intervals must be a ConfidenceInterval or an iterable of them') from exc
    if not items or any(not isinstance(item, ConfidenceInterval) for item in items):
        raise TypeError('intervals must contain ConfidenceInterval objects')
    estimates, lower, upper, methods = [], [], [], []
    for item in items:
        estimate = np.asarray(item.estimate, dtype=float)
        lo = np.asarray(item.lower, dtype=float)
        hi = np.asarray(item.upper, dtype=float)
        try:
            estimate, lo, hi = np.broadcast_arrays(estimate, lo, hi)
        except ValueError as exc:
            raise ValueError('confidence interval estimates and bounds must broadcast') from exc
        estimates.extend(estimate.reshape(-1).tolist())
        lower.extend(lo.reshape(-1).tolist())
        upper.extend(hi.reshape(-1).tolist())
        methods.extend([item.method] * estimate.size)
    estimates = np.asarray(estimates)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if np.any(~np.isfinite(estimates)) or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError('confidence interval plot requires finite estimates and bounds')
    if np.any(lower > estimates) or np.any(estimates > upper):
        raise ValueError('each estimate must lie within its confidence interval')
    return estimates, lower, upper, methods


def confidence_interval_plot(intervals, *, labels=None, reference=None,
                             orientation='horizontal', fig=None, ax=None,
                             show=True, title=None, theme=None, grid=None,
                             color=None, **style):
    """Plot one or more ``ConfidenceInterval`` results and return the axes."""
    estimates, lower, upper, methods = _interval_arrays(intervals)
    count = estimates.size
    if labels is None:
        labels = methods if any(methods) else [str(index + 1) for index in range(count)]
    else:
        labels = list(labels)
        if len(labels) != count:
            raise ValueError('labels must match the number of plotted intervals')
    if orientation not in {'horizontal', 'vertical'}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    positions = np.arange(count)
    negative = estimates - lower
    positive = upper - estimates
    fig, ax = _axes(fig, ax, theme=theme)
    style.setdefault('fmt', 'o')
    style.setdefault('capsize', 4)
    if color is not None:
        style['color'] = color
    if orientation == 'horizontal':
        ax.errorbar(estimates, positions, xerr=np.vstack((negative, positive)), **style)
        ax.set_yticks(positions, labels)
        ax.set_xlabel('Estimate')
        if reference is not None:
            ax.axvline(float(reference), color='#555555', linestyle='--', linewidth=1.5)
    else:
        ax.errorbar(positions, estimates, yerr=np.vstack((negative, positive)), **style)
        ax.set_xticks(positions, labels)
        ax.set_ylabel('Estimate')
        if reference is not None:
            ax.axhline(float(reference), color='#555555', linestyle='--', linewidth=1.5)
    _finish(fig, ax, title='Confidence intervals' if title is None else title,
            xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), theme=theme, grid=grid,
            label=None, legend=False, show=show)
    return ax


def diagnostic_plots(data, distribution=None, *, nan_policy='omit', bins='auto',
                     theme=None, grid=None, show=True,
                     title='Statistical diagnostics'):
    """Create a compact histogram, ECDF, Q-Q, and P-P diagnostic dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    reference = 'normal' if distribution is None else distribution
    histogram_plot(data, bins=bins, fit=reference, nan_policy=nan_policy,
                   fig=fig, ax=axes[0, 0], show=False, theme=theme, grid=grid)
    plot_ecdf(data, nan_policy=nan_policy, fig=fig, ax=axes[0, 1],
              show=False, theme=theme, grid=grid)
    qq_plot(data, distribution, nan_policy=nan_policy, fig=fig, ax=axes[1, 0],
            show=False, theme=theme, grid=grid)
    pp_plot(data, distribution, nan_policy=nan_policy, fig=fig, ax=axes[1, 1],
            show=False, theme=theme, grid=grid)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


__all__ = [
    'plot_ecdf', 'qq_plot', 'pp_plot', 'histogram_plot',
    'confidence_interval_plot', 'diagnostic_plots',
]
