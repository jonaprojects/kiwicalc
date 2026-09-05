"""Standalone scientific-field plotting functions.

These functions share the same renderers as :class:`kiwicalc.Graph2D`, keeping
the standalone and compositional APIs visually and numerically consistent.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt

from kiwicalc.plotting.fields import _range, render
from kiwicalc.plotting.plots import _figure_and_axes


AxisRange = Optional[Iterable[float]]
Density = Union[int, Sequence[int]]


def _finish(ax, artists, colorbar, *, title, xlabel, ylabel, show):
    ax.set_title(title or '')
    ax.set_xlabel(xlabel or '')
    ax.set_ylabel(ylabel or '')
    primary = artists[0]
    primary.kiwicalc_artists = tuple(artists)
    primary.kiwicalc_colorbar = colorbar
    if show:
        plt.show()
    return primary


def _field_plot(decoration, *, x_range, y_range, fig, ax, title, xlabel,
                ylabel, show):
    fig, ax = _figure_and_axes(fig, ax)
    resolved_x = _range(x_range, (-10, 10), 'x_range')
    resolved_y = _range(y_range, (-10, 10), 'y_range')
    decoration = dict(decoration, x_range=resolved_x, y_range=resolved_y)
    artists, colorbar = render(ax, decoration, resolved_x, resolved_y)
    ax.set_xlim(*resolved_x)
    ax.set_ylim(*resolved_y)
    return _finish(
        ax, artists, colorbar, title=title, xlabel=xlabel, ylabel=ylabel,
        show=show,
    )


def plot_vector_field(
    u, v, *, x_range: AxisRange=None, y_range: AxisRange=None,
    density: Density=20, normalize: bool=False, color: Any='magnitude',
    colorbar: bool=False, colorbar_label: Optional[str]='Magnitude',
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, **style,
):
    """Plot vectors ``(u(x, y), v(x, y))`` over a rectangular grid."""
    decoration = dict(
        kind='vector_field', u=u, v=v, x_range=x_range, y_range=y_range,
        density=density, normalize=bool(normalize), color=color,
        colorbar=bool(colorbar), colorbar_label=colorbar_label,
        style=dict(style),
    )
    return _field_plot(
        decoration, x_range=x_range, y_range=y_range, fig=fig, ax=ax,
        title=title, xlabel=xlabel, ylabel=ylabel, show=show,
    )


def plot_slope_field(
    derivative, *, x_range: AxisRange=None, y_range: AxisRange=None,
    density: Density=20, normalize: bool=True, color: Any='black',
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, **style,
):
    """Plot the direction field for ``dy/dx = derivative(x, y)``."""
    decoration = dict(
        kind='slope_field', source=derivative, x_range=x_range,
        y_range=y_range, density=density, normalize=bool(normalize),
        color=color, colorbar=False, style=dict(style),
    )
    return _field_plot(
        decoration, x_range=x_range, y_range=y_range, fig=fig, ax=ax,
        title=title, xlabel=xlabel, ylabel=ylabel, show=show,
    )


def plot_gradient_field(
    function, *, x_range: AxisRange=None, y_range: AxisRange=None,
    density: Density=20, normalize: bool=False, color: Any='magnitude',
    colorbar: bool=False, colorbar_label: Optional[str]='Gradient magnitude',
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, **style,
):
    """Plot the numerical gradient of a scalar function ``f(x, y)``."""
    decoration = dict(
        kind='gradient_field', source=function, x_range=x_range,
        y_range=y_range, density=density, normalize=bool(normalize),
        color=color, colorbar=bool(colorbar),
        colorbar_label=colorbar_label, style=dict(style),
    )
    return _field_plot(
        decoration, x_range=x_range, y_range=y_range, fig=fig, ax=ax,
        title=title, xlabel=xlabel, ylabel=ylabel, show=show,
    )


def plot_streamlines(
    u, v, *, x_range: AxisRange=None, y_range: AxisRange=None,
    samples: int=60, density: float=1.0, color: Any='magnitude',
    colorbar: bool=False, colorbar_label: Optional[str]='Magnitude',
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, **style,
):
    """Plot continuous streamlines of a two-dimensional vector field."""
    decoration = dict(
        kind='streamlines', u=u, v=v, x_range=x_range, y_range=y_range,
        samples=samples, stream_density=density, color=color,
        colorbar=bool(colorbar), colorbar_label=colorbar_label,
        style=dict(style),
    )
    return _field_plot(
        decoration, x_range=x_range, y_range=y_range, fig=fig, ax=ax,
        title=title, xlabel=xlabel, ylabel=ylabel, show=show,
    )


def plot_contour(
    function, *, x_range: AxisRange=None, y_range: AxisRange=None,
    levels=10, filled: bool=False, labels: bool=False, samples: int=100,
    colorbar: bool=False, colorbar_label: Optional[str]=None,
    label_size: float=8, title: str='', xlabel: str='x', ylabel: str='y',
    show: bool=True, fig=None, ax=None, **style,
):
    """Plot line or filled contours of a scalar function ``f(x, y)``."""
    decoration = dict(
        kind='contour_map', source=function, x_range=x_range,
        y_range=y_range, levels=levels, filled=bool(filled),
        labels=bool(labels), samples=samples, colorbar=bool(colorbar),
        colorbar_label=colorbar_label, label_size=label_size,
        style=dict(style),
    )
    return _field_plot(
        decoration, x_range=x_range, y_range=y_range, fig=fig, ax=ax,
        title=title, xlabel=xlabel, ylabel=ylabel, show=show,
    )


plot_streamplot = plot_streamlines
plot_contour_map = plot_contour


__all__ = [
    'plot_vector_field', 'plot_slope_field', 'plot_gradient_field',
    'plot_streamlines', 'plot_streamplot', 'plot_contour', 'plot_contour_map',
]
