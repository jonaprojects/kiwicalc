"""Standalone plotting functions for common mathematical constructions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import re
from numbers import Real
from typing import Any, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from kiwicalc.core.interfaces import IExpression
from kiwicalc.functions.function import Function
from kiwicalc.geometry.curves import ParametricCurve2D, PolarCurve2D
from kiwicalc.plotting.fields import evaluate_xy, make_grid, _range
from kiwicalc.plotting.plots import _figure_and_axes, plot_curve_2d
from kiwicalc.plotting.sampling import PlotSample, normalize_sampling, sample_for_plot
from kiwicalc.sequences.sequences import Sequence


@dataclass(frozen=True)
class PiecewisePlotResult:
    """Artists and sampling diagnostics produced by :func:`plot_piecewise`."""

    lines: Tuple[Any, ...]
    endpoint_markers: Tuple[Any, ...]
    samples: Tuple[PlotSample, ...]


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return int(value)


def _interval(value, name='interval'):
    try:
        start, stop = map(float, value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must contain two finite bounds')
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        raise ValueError(f'{name} must contain two increasing finite bounds')
    return start, stop


def _function(source):
    if isinstance(source, Real) and not isinstance(source, bool):
        value = float(source)
        if not math.isfinite(value):
            raise ValueError('function constants must be finite')
        return lambda x: np.zeros_like(np.asarray(x), dtype=float) + value
    if isinstance(source, str):
        source = Function(source)
    if isinstance(source, Function):
        return source.lambda_expression if callable(source.lambda_expression) else source
    if isinstance(source, IExpression):
        source = source.to_lambda()
    if callable(source):
        return source
    raise TypeError('function must be numeric, callable, an expression, or a formula string')


def _evaluate(source, coordinates):
    function = _function(source)
    x = np.asarray(coordinates, dtype=float)
    try:
        with np.errstate(all='ignore'):
            result = np.asarray(function(x), dtype=float)
        if result.shape == ():
            result = np.full(x.shape, float(result))
        return np.where(np.isfinite(np.broadcast_to(result, x.shape)), result, np.nan)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        result = np.empty(x.shape, dtype=float)
        for index, value in np.ndenumerate(x):
            try:
                candidate = float(function(float(value)))
                result[index] = candidate if math.isfinite(candidate) else np.nan
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                result[index] = np.nan
        return result


def _closure(value):
    aliases = {
        'closed-closed': (True, True), '[]': (True, True),
        'closed-open': (True, False), '[)': (True, False),
        'open-closed': (False, True), '(]': (False, True),
        'open-open': (False, False), '()': (False, False),
    }
    if isinstance(value, str):
        try:
            return aliases[value.strip().lower()]
        except KeyError:
            raise ValueError('closure must be closed-closed, closed-open, open-closed, or open-open')
    try:
        left, right = value
    except (TypeError, ValueError):
        raise ValueError('closure must describe the left and right endpoints')
    if not isinstance(left, bool) or not isinstance(right, bool):
        raise ValueError('closure endpoint values must be booleans')
    return left, right


def plot_piecewise(
    pieces, *, samples: int=201, sampling: str='fixed', tolerance: float=1e-3,
    max_points: int=5000, max_depth: int=12, endpoint_size: float=45,
    title: str='', xlabel: str='x', ylabel: str='y', label: Optional[str]=None,
    show: bool=True, fig=None, ax=None, **style,
) -> PiecewisePlotResult:
    """Plot ``(interval, function[, closure])`` pieces with endpoint markers."""
    samples = _count(samples, 'samples', 2)
    mode = normalize_sampling(sampling)
    if isinstance(endpoint_size, bool) or not math.isfinite(float(endpoint_size)) or float(endpoint_size) <= 0:
        raise ValueError('endpoint_size must be a positive finite number')
    try:
        pieces = list(pieces)
    except TypeError:
        raise TypeError('pieces must be an iterable of piece specifications')
    if not pieces:
        raise ValueError('pieces must not be empty')
    fig, ax = _figure_and_axes(fig, ax)
    lines, markers, diagnostics = [], [], []
    base_color = style.get('color')
    for index, piece in enumerate(pieces):
        try:
            if len(piece) == 2:
                bounds, source = piece
                closed = (True, False)
            elif len(piece) == 3:
                bounds, source, closed = piece
            else:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('each piece must be (interval, function[, closure])')
        start, stop = _interval(bounds, 'piece interval')
        closed = _closure(closed)
        sample = sample_for_plot(
            source, start, stop, (stop - start) / (samples - 1),
            sampling=mode, tolerance=tolerance, max_points=max_points,
            max_depth=max_depth,
        )
        line_label = label if index == 0 else None
        line, = ax.plot(sample.x, sample.y, label=line_label, **style)
        if mode == 'adaptive':
            line.kiwicalc_sample = sample
        lines.append(line)
        diagnostics.append(sample)
        color = base_color or line.get_color()
        for coordinate, is_closed in zip((start, stop), closed):
            value = _evaluate(source, [coordinate])[0]
            if not np.isfinite(value):
                continue
            marker = ax.scatter(
                [coordinate], [value], s=float(endpoint_size),
                facecolors=color if is_closed else ax.get_facecolor(),
                edgecolors=color, zorder=line.get_zorder() + 1,
            )
            markers.append(marker)
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return PiecewisePlotResult(tuple(lines), tuple(markers), tuple(diagnostics))


def plot_region(
    lower, upper, *, interval=(-10, 10), samples: int=501,
    sampling: str='fixed', tolerance: float=1e-3, max_points: int=5000,
    max_depth: int=12, where=None, title: str='', xlabel: str='x',
    ylabel: str='y', label: Optional[str]=None, show: bool=True,
    fig=None, ax=None, **style,
):
    """Shade the vertical region between two scalar functions."""
    start, stop = _interval(interval)
    samples = _count(samples, 'samples', 2)
    lower_function, upper_function = _function(lower), _function(upper)

    def separation(x):
        return _evaluate(upper_function, x) - _evaluate(lower_function, x)

    sample = sample_for_plot(
        separation, start, stop, (stop - start) / (samples - 1),
        sampling=sampling, tolerance=tolerance, max_points=max_points,
        max_depth=max_depth,
    )
    low, high = _evaluate(lower_function, sample.x), _evaluate(upper_function, sample.x)
    mask = np.isfinite(low) & np.isfinite(high)
    if where is not None:
        if callable(where):
            try:
                selected = np.asarray(where(sample.x, low, high), dtype=bool)
            except TypeError:
                selected = np.asarray(where(sample.x), dtype=bool)
        else:
            selected = np.asarray(where, dtype=bool)
        try:
            mask &= np.broadcast_to(selected, mask.shape)
        except ValueError:
            raise ValueError('where must match the sampled coordinate shape')
    style.setdefault('alpha', 0.25)
    fig, ax = _figure_and_axes(fig, ax)
    artist = ax.fill_between(sample.x, low, high, where=mask, label=label, **style)
    artist.kiwicalc_sample = sample
    artist.kiwicalc_lower = low
    artist.kiwicalc_upper = high
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return artist


_INEQUALITY = re.compile(r'^\s*(.+?)\s*(<=|>=|<|>)\s*(.+?)\s*$')


def _field_operand(source, X, Y):
    if isinstance(source, str):
        try:
            return np.full_like(X, float(source), dtype=float)
        except ValueError:
            pass
    if isinstance(source, Real) and not isinstance(source, bool):
        return np.full_like(X, float(source), dtype=float)
    return evaluate_xy(source, X, Y)


def plot_inequality(
    inequality, right=None, *, relation=None, x_range=(-10, 10),
    y_range=(-10, 10), samples: int=250, max_points: int=1000000,
    boundary: bool=True,
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, **style,
):
    """Shade a two-variable inequality or boolean predicate.

    String input may use forms such as ``"x^2 + y^2 <= 4"``. Alternatively,
    pass a boolean callable ``predicate(x, y)`` or ``left, right, relation=``.
    """
    x_range, y_range = _range(x_range, (-10, 10), 'x_range'), _range(y_range, (-10, 10), 'y_range')
    samples = _count(samples, 'samples', 2)
    max_points = _count(max_points, 'max_points', 1)
    if samples * samples > max_points:
        raise ValueError('samples squared exceeds max_points')
    x, y, X, Y = make_grid(x_range, y_range, samples)
    delta = None
    if isinstance(inequality, str) and right is None and relation is None:
        match = _INEQUALITY.match(inequality)
        if match is None:
            raise ValueError('inequality strings must contain <, <=, >, or >=')
        inequality, relation, right = match.groups()
    if right is None and relation is None and callable(inequality):
        try:
            selected = np.asarray(inequality(X, Y), dtype=bool)
            selected = np.broadcast_to(selected, X.shape)
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            selected = np.empty(X.shape, dtype=bool)
            for index in np.ndindex(X.shape):
                selected[index] = bool(inequality(float(X[index]), float(Y[index])))
    else:
        if relation not in {'<', '<=', '>', '>='}:
            raise ValueError("relation must be '<', '<=', '>', or '>='")
        left_values = _field_operand(inequality, X, Y)
        right_values = _field_operand(right, X, Y)
        delta = left_values - right_values
        comparisons = {
            '<': np.less, '<=': np.less_equal,
            '>': np.greater, '>=': np.greater_equal,
        }
        selected = comparisons[relation](left_values, right_values)
    selected = np.asarray(selected, dtype=float)
    if selected.shape != X.shape:
        raise ValueError('inequality predicate must return one value per grid coordinate')
    fig, ax = _figure_and_axes(fig, ax)
    style.setdefault('alpha', 0.3)
    if 'color' in style and 'colors' not in style:
        style['colors'] = [style.pop('color')]
    filled = ax.contourf(X, Y, selected, levels=[0.5, 1.5], **style)
    boundary_artist = None
    if boundary:
        boundary_values = delta if delta is not None else selected - 0.5
        boundary_artist = ax.contour(X, Y, boundary_values, levels=[0], colors='black', linewidths=1)
    filled.kiwicalc_boundary = boundary_artist
    filled.kiwicalc_mask = selected.astype(bool)
    ax.set(xlim=x_range, ylim=y_range, title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return filled


def plot_parametric(
    x, y, *, t_range=(0, 2 * np.pi), samples: int=500,
    sampling: str='fixed', tolerance: float=1e-3, max_depth: int=10,
    title: str='', xlabel: str='x', ylabel: str='y', label: Optional[str]=None,
    equal_aspect: bool=False, show: bool=True, fig=None, ax=None, **style,
):
    """Plot ``(x(t), y(t))`` without explicitly constructing a curve."""
    curve = ParametricCurve2D(
        x, y, t_range=_interval(t_range, 't_range'), samples=samples,
        sampling=sampling, tolerance=tolerance, max_depth=max_depth,
    )
    fig, ax = _figure_and_axes(fig, ax)
    line = plot_curve_2d(curve, show=False, fig=fig, ax=ax, label=label, **style)
    line.kiwicalc_curve = curve
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    return line


def plot_polar(
    radius, *, theta_range=(0, 2 * np.pi), samples: int=500,
    sampling: str='fixed', tolerance: float=1e-3, max_depth: int=10,
    title: str='', label: Optional[str]=None, equal_aspect: bool=True,
    show: bool=True, fig=None, ax=None, **style,
):
    """Plot a polar function ``r(theta)`` on Cartesian axes."""
    curve = PolarCurve2D(
        radius, theta_range=_interval(theta_range, 'theta_range'),
        samples=samples, sampling=sampling, tolerance=tolerance,
        max_depth=max_depth,
    )
    fig, ax = _figure_and_axes(fig, ax)
    line = plot_curve_2d(curve, show=False, fig=fig, ax=ax, label=label, **style)
    line.kiwicalc_curve = curve
    ax.set_title(title or '')
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    return line


def plot_sequence(
    sequence, *, start: int=1, stop: int=20, step: int=1,
    indices: Optional[Iterable[int]]=None, title: str='', xlabel: str='n',
    ylabel: str='a[n]', label: Optional[str]=None, show: bool=True,
    fig=None, ax=None, **style,
):
    """Draw a discrete sequence as a stem plot; ``stop`` is inclusive."""
    if indices is None:
        for value, name in ((start, 'start'), (stop, 'stop'), (step, 'step')):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f'{name} must be an integer')
        if step == 0:
            raise ValueError('step must not be zero')
        indices = list(range(start, stop + (1 if step > 0 else -1), step))
    else:
        indices = list(indices)
        if not indices or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in indices):
            raise ValueError('indices must contain integers')
    if not indices:
        raise ValueError('the requested sequence range is empty')
    if isinstance(sequence, Sequence):
        values = [sequence.in_index(int(index)) for index in indices]
    elif callable(sequence):
        values = [sequence(int(index)) for index in indices]
    else:
        values = list(sequence)
        if len(values) != len(indices):
            raise ValueError('sequence values must match the number of indices')
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('sequence values must be finite real numbers')
    fig, ax = _figure_and_axes(fig, ax)
    container = ax.stem(indices, values, label=label, **style)
    container.kiwicalc_indices = np.asarray(indices, dtype=int)
    container.kiwicalc_values = values
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return container


def plot_error_band(
    function, error, *, start: float=-10, stop: float=10, samples: int=501,
    sampling: str='fixed', tolerance: float=1e-3, max_points: int=5000,
    max_depth: int=12, label: Optional[str]=None, band_label: Optional[str]=None,
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, line_style=None, **band_style,
):
    """Plot a function with a symmetric scalar or callable error band."""
    start, stop = _interval((start, stop), 'range')
    samples = _count(samples, 'samples', 2)
    sample = sample_for_plot(
        function, start, stop, (stop - start) / (samples - 1),
        sampling=sampling, tolerance=tolerance, max_points=max_points,
        max_depth=max_depth,
    )
    errors = _evaluate(error, sample.x)
    if np.any(errors < 0):
        raise ValueError('error values must be non-negative')
    fig, ax = _figure_and_axes(fig, ax)
    line, = ax.plot(sample.x, sample.y, label=label, **dict(line_style or {}))
    if sample.sampling == 'adaptive':
        line.kiwicalc_sample = sample
    band_style.setdefault('alpha', 0.2)
    band_style.setdefault('color', line.get_color())
    band = ax.fill_between(
        sample.x, sample.y - errors, sample.y + errors,
        label=band_label, **band_style,
    )
    band.kiwicalc_line = line
    band.kiwicalc_sample = sample
    band.kiwicalc_error = errors
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return band


__all__ = [
    'PiecewisePlotResult', 'plot_piecewise', 'plot_region', 'plot_inequality',
    'plot_parametric', 'plot_polar', 'plot_sequence', 'plot_error_band',
]
