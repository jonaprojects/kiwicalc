"""Advanced standalone visualizations for numerical and dynamical mathematics."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from numbers import Real
from typing import Any, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

from kiwicalc.geometry.curves import Curve2D
from kiwicalc.plotting.field_plots import plot_streamlines
from kiwicalc.plotting.fields import evaluate_xy, make_grid, _range
from kiwicalc.plotting.math_plots import _count, _interval
from kiwicalc.plotting.plots import _figure_and_axes


@dataclass(frozen=True)
class PhasePortraitResult:
    field: Any
    trajectories: Tuple[Any, ...]
    equilibria: Optional[Any]


@dataclass(frozen=True)
class TransformPlotResult:
    original: Any
    transformed: Any
    connectors: Tuple[Any, ...]
    original_points: np.ndarray
    transformed_points: np.ndarray


def _field_scalar(source, x, y):
    value = evaluate_xy(
        source, np.asarray([[float(x)]], dtype=float),
        np.asarray([[float(y)]], dtype=float),
    )[0, 0]
    return float(value)


def _scalar_component(source):
    if isinstance(source, Real) and not isinstance(source, bool):
        value = float(source)
        return lambda x, y: value
    if callable(source):
        def evaluate(x, y):
            value = float(source(float(x), float(y)))
            return value if math.isfinite(value) else np.nan
        return evaluate
    return lambda x, y: _field_scalar(source, x, y)


def _system_functions(u, v):
    if v is not None:
        return _scalar_component(u), _scalar_component(v)
    if not callable(u):
        raise TypeError('system must be two field components or a callable returning two values')

    def component(index):
        def evaluate(x, y):
            result = np.asarray(u(float(x), float(y)), dtype=float)
            if result.shape != (2,):
                raise ValueError('system callable must return two values')
            return float(result[index])
        return evaluate

    return component(0), component(1)


def _trajectory(u, v, initial, t_span, steps, escape):
    start, stop = _interval(t_span, 't_span')
    times = np.linspace(start, stop, steps)
    dt = float(times[1] - times[0])
    points = np.empty((steps, 2), dtype=float)
    points[0] = initial

    def field(point):
        return np.asarray((u(*point), v(*point)), dtype=float)

    used = 1
    for index in range(1, steps):
        point = points[index - 1]
        try:
            k1 = field(point)
            k2 = field(point + dt*k1/2)
            k3 = field(point + dt*k2/2)
            k4 = field(point + dt*k3)
            candidate = point + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            break
        if not np.isfinite(candidate).all() or np.max(np.abs(candidate)) > escape:
            break
        points[index] = candidate
        used += 1
    return points[:used]


def plot_phase_portrait(
    u, v=None, *, x_range=(-5, 5), y_range=(-5, 5), density: int=30,
    initial_conditions: Optional[Iterable[Iterable[float]]]=None,
    t_span=(0, 10), trajectory_steps: int=500, escape: float=1e6,
    equilibria: bool=True, equilibrium_tolerance: float=1e-8,
    max_field_points: int=250000, max_trajectory_points: int=100000,
    title: str='', xlabel: str='x', ylabel: str='y', show: bool=True,
    fig=None, ax=None, field_style=None, trajectory_style=None,
) -> PhasePortraitResult:
    """Plot a planar autonomous system with optional RK4 trajectories."""
    x_range = _range(x_range, (-5, 5), 'x_range')
    y_range = _range(y_range, (-5, 5), 'y_range')
    density = _count(density, 'density', 2)
    trajectory_steps = _count(trajectory_steps, 'trajectory_steps', 2)
    max_field_points = _count(max_field_points, 'max_field_points', 1)
    max_trajectory_points = _count(max_trajectory_points, 'max_trajectory_points', 1)
    if density * density > max_field_points:
        raise ValueError('density squared exceeds max_field_points')
    escape = float(escape)
    equilibrium_tolerance = float(equilibrium_tolerance)
    if not math.isfinite(escape) or escape <= 0:
        raise ValueError('escape must be a positive finite number')
    if not math.isfinite(equilibrium_tolerance) or equilibrium_tolerance < 0:
        raise ValueError('equilibrium_tolerance must be a non-negative finite number')
    u_function, v_function = _system_functions(u, v)
    if initial_conditions is not None:
        try:
            initial_conditions = list(initial_conditions)
        except TypeError:
            raise TypeError('initial_conditions must contain coordinate pairs')
        if len(initial_conditions) * trajectory_steps > max_trajectory_points:
            raise ValueError('trajectory count times trajectory_steps exceeds max_trajectory_points')
    fig, ax = _figure_and_axes(fig, ax)
    options = dict(field_style or {})
    color = options.pop('color', 'magnitude')
    field = plot_streamlines(
        u, v if v is not None else (lambda x, y: np.asarray(u(x, y))[1]),
        x_range=x_range, y_range=y_range, samples=density, color=color,
        fig=fig, ax=ax, show=False, **options,
    ) if v is not None else plot_streamlines(
        lambda x, y: np.asarray(u(x, y))[0],
        lambda x, y: np.asarray(u(x, y))[1],
        x_range=x_range, y_range=y_range, samples=density, color=color,
        fig=fig, ax=ax, show=False, **options,
    )
    trajectory_style = dict(trajectory_style or {})
    trajectory_style.setdefault('linewidth', 1.5)
    trajectory_style.setdefault('zorder', 3)
    trajectories = []
    if initial_conditions is not None:
        for initial in initial_conditions:
            point = np.asarray(initial, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError('each initial condition must contain two finite coordinates')
            values = _trajectory(u_function, v_function, point, t_span, trajectory_steps, escape)
            line, = ax.plot(values[:, 0], values[:, 1], **trajectory_style)
            line.kiwicalc_trajectory = values
            trajectories.append(line)
    equilibrium_artist = None
    if equilibria:
        _, _, X, Y = make_grid(x_range, y_range, density)
        U = evaluate_xy(
            u if v is not None else (lambda x, y: np.asarray(u(x, y))[0]), X, Y,
        )
        V = evaluate_xy(
            v if v is not None else (lambda x, y: np.asarray(u(x, y))[1]), X, Y,
        )
        mask = np.isfinite(U) & np.isfinite(V) & (np.hypot(U, V) <= equilibrium_tolerance)
        if np.any(mask):
            equilibrium_artist = ax.scatter(
                X[mask], Y[mask], marker='o', s=45, facecolors='white',
                edgecolors='black', zorder=4, label='equilibrium',
            )
    ax.set(xlim=x_range, ylim=y_range, title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return PhasePortraitResult(field, tuple(trajectories), equilibrium_artist)


def _complex_callable(source):
    if not callable(source):
        raise TypeError('complex function must be callable')
    return source


def _evaluate_complex(source, Z):
    function = _complex_callable(source)
    try:
        with np.errstate(all='ignore'):
            values = np.asarray(function(Z), dtype=complex)
        if values.shape == ():
            values = np.full(Z.shape, values.item(), dtype=complex)
        return np.broadcast_to(values, Z.shape).astype(complex, copy=False)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        values = np.empty(Z.shape, dtype=complex)
        for index, value in np.ndenumerate(Z):
            try:
                values[index] = complex(function(complex(value)))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                values[index] = complex(np.nan, np.nan)
        return values


def plot_complex_function(
    function, *, real_range=(-2, 2), imag_range=(-2, 2), samples: int=400,
    max_points: int=1000000, mode: str='domain', cmap: str='viridis', colorbar: bool=False,
    title: str='', xlabel: str='Re(z)', ylabel: str='Im(z)', show: bool=True,
    fig=None, ax=None, **style,
):
    """Visualize a complex function using domain coloring or a scalar component."""
    real_range = _range(real_range, (-2, 2), 'real_range')
    imag_range = _range(imag_range, (-2, 2), 'imag_range')
    samples = _count(samples, 'samples', 2)
    max_points = _count(max_points, 'max_points', 1)
    if samples * samples > max_points:
        raise ValueError('samples squared exceeds max_points')
    if not isinstance(mode, str):
        raise TypeError('mode must be a string')
    mode = mode.strip().lower()
    if mode not in {'domain', 'magnitude', 'phase', 'real', 'imaginary'}:
        raise ValueError('mode must be domain, magnitude, phase, real, or imaginary')
    real = np.linspace(*real_range, samples)
    imaginary = np.linspace(*imag_range, samples)
    X, Y = np.meshgrid(real, imaginary)
    values = _evaluate_complex(function, X + 1j*Y)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if mode == 'domain':
        hue = (np.angle(values) + np.pi) / (2*np.pi)
        magnitude = np.abs(values)
        saturation = np.full(values.shape, 0.85)
        brightness = 1 - 0.45/(1 + magnitude)
        data = hsv_to_rgb(np.stack((hue, saturation, brightness), axis=-1))
        data[~finite] = 0
        image_options = dict(origin='lower', extent=(*real_range, *imag_range), aspect='auto')
    else:
        components = {
            'magnitude': np.abs(values), 'phase': np.angle(values),
            'real': values.real, 'imaginary': values.imag,
        }
        data = np.where(finite, components[mode], np.nan)
        image_options = dict(
            origin='lower', extent=(*real_range, *imag_range),
            aspect='auto', cmap=cmap,
        )
    image_options.update(style)
    fig, ax = _figure_and_axes(fig, ax)
    image = ax.imshow(data, **image_options)
    image.kiwicalc_values = values
    image.kiwicalc_mode = mode
    image.kiwicalc_colorbar = fig.colorbar(image, ax=ax) if colorbar and mode != 'domain' else None
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return image


def plot_convergence(
    history, *, x=None, metric: str='residual', log_scale: Optional[bool]=None,
    title: Optional[str]=None, xlabel: str='Iteration', ylabel: Optional[str]=None,
    label: Optional[str]=None, show: bool=True, fig=None, ax=None, **style,
):
    """Plot a convergence history or a traced ``NumericalExplanation``."""
    if metric not in {'residual', 'estimate'}:
        raise ValueError("metric must be 'residual' or 'estimate'")
    if log_scale is not None and not isinstance(log_scale, bool):
        raise TypeError('log_scale must be a boolean or None')
    method = None
    if hasattr(history, 'steps'):
        method = getattr(history, 'method', None)
        records = list(history.steps)
        if metric == 'residual':
            records = [record for record in records if record.residual is not None]
            values = [record.residual for record in records]
        elif metric == 'estimate':
            values = [record.estimate for record in records]
        coordinates = [record.index for record in records]
    else:
        try:
            values = np.asarray(list(history), dtype=float)
        except (TypeError, ValueError):
            raise TypeError('history must be an iterable or a numerical explanation')
        coordinates = np.arange(1, len(values) + 1) if x is None else np.asarray(list(x), dtype=float)
    values = np.asarray(values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or not values.size or coordinates.shape != values.shape:
        raise ValueError('history and x must be non-empty one-dimensional arrays of equal length')
    if not np.isfinite(values).all() or not np.isfinite(coordinates).all():
        raise ValueError('convergence coordinates and values must be finite')
    if log_scale is None:
        log_scale = metric == 'residual'
    if log_scale and np.any(values < 0):
        raise ValueError('log-scaled convergence values must be non-negative')
    fig, ax = _figure_and_axes(fig, ax)
    style.setdefault('marker', 'o')
    line, = ax.plot(coordinates, values, label=label, **style)
    if log_scale:
        if np.any(values == 0):
            positive = values[values > 0]
            threshold = max(float(positive.min()) * 0.1, 1e-15) if positive.size else 1e-15
            ax.set_yscale('symlog', linthresh=threshold)
        else:
            ax.set_yscale('log')
    resolved_ylabel = ylabel or ('Residual' if metric == 'residual' else 'Estimate')
    resolved_title = title if title is not None else (f'{method} convergence' if method else 'Convergence')
    ax.set(title=resolved_title, xlabel=xlabel or '', ylabel=resolved_ylabel)
    if show:
        plt.show()
    return line


def _points(source, samples):
    if isinstance(source, Curve2D):
        return np.column_stack(source.sample(samples=samples)), True
    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError('geometry must be a Curve2D or an (n, 2) coordinate array')
    if not np.isfinite(values).all():
        raise ValueError('geometry coordinates must be finite')
    return values, False


def _transform_points(points, transform):
    matrix = None
    try:
        candidate = np.asarray(transform, dtype=float)
        if candidate.shape == (3, 3):
            matrix = candidate
    except (TypeError, ValueError):
        pass
    if matrix is not None:
        homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed = homogeneous @ matrix.T
        if np.any(np.isclose(transformed[:, 2], 0)):
            raise ValueError('transform maps points to zero homogeneous scale')
        return transformed[:, :2] / transformed[:, 2, None]
    if not callable(transform):
        raise ValueError('transform must be a 3x3 matrix, callable, or Curve2D')
    try:
        transformed = np.asarray(transform(points.copy()), dtype=float)
        if transformed.shape == points.shape:
            return transformed
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        pass
    transformed = np.asarray([transform(tuple(point)) for point in points], dtype=float)
    if transformed.shape != points.shape:
        raise ValueError('transform callable must return two coordinates per point')
    return transformed


def plot_transform(
    original, transform, *, samples: int=200, connect: Optional[bool]=None,
    connectors: int=0,
    equal_aspect: bool=True, title: str='', xlabel: str='x', ylabel: str='y',
    original_label: str='original', transformed_label: str='transformed',
    show: bool=True, fig=None, ax=None, original_style=None,
    transformed_style=None, connector_style=None,
) -> TransformPlotResult:
    """Compare a 2D curve or point set before and after a transformation."""
    samples = _count(samples, 'samples', 2)
    connectors = _count(connectors, 'connectors', 0)
    points, connected = _points(original, samples)
    if connect is not None and not isinstance(connect, bool):
        raise TypeError('connect must be a boolean or None')
    if isinstance(transform, Curve2D):
        transformed_points, transformed_connected = _points(transform, samples)
        if transformed_points.shape != points.shape:
            raise ValueError('original and transformed curves must have matching samples')
        connected = connected and transformed_connected
    else:
        transformed_points = _transform_points(points, transform)
    if connect is not None:
        connected = connect
    if not np.isfinite(transformed_points).all():
        raise ValueError('transformed coordinates must be finite')
    fig, ax = _figure_and_axes(fig, ax)
    original_style, transformed_style = dict(original_style or {}), dict(transformed_style or {})
    original_style.setdefault('color', '0.55')
    original_style.setdefault('linestyle', '--')
    transformed_style.setdefault('color', 'tab:blue')
    if connected:
        original_artist, = ax.plot(points[:, 0], points[:, 1], label=original_label, **original_style)
        transformed_artist, = ax.plot(transformed_points[:, 0], transformed_points[:, 1], label=transformed_label, **transformed_style)
    else:
        original_style.pop('linestyle', None)
        original_artist = ax.scatter(points[:, 0], points[:, 1], label=original_label, **original_style)
        transformed_artist = ax.scatter(transformed_points[:, 0], transformed_points[:, 1], label=transformed_label, **transformed_style)
    connector_artists = []
    if connectors:
        connector_style = dict(connector_style or {})
        connector_style.setdefault('color', '0.7')
        connector_style.setdefault('linewidth', 0.7)
        connector_style.setdefault('alpha', 0.7)
        indices = np.unique(np.linspace(0, len(points) - 1, min(connectors, len(points))).astype(int))
        for index in indices:
            line, = ax.plot(
                [points[index, 0], transformed_points[index, 0]],
                [points[index, 1], transformed_points[index, 1]],
                **connector_style,
            )
            connector_artists.append(line)
    ax.set(title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if equal_aspect:
        ax.set_aspect('equal', adjustable='datalim')
    if show:
        plt.show()
    return TransformPlotResult(
        original_artist, transformed_artist, tuple(connector_artists),
        points, transformed_points,
    )


def plot_bifurcation(
    function, *, parameter_range, initial_state: float=0.5,
    parameter_samples: int=500, burn_in: int=300, keep: int=100,
    escape: float=1e6, max_points: int=200000, max_iterations: int=1000000,
    title: str='Bifurcation diagram',
    xlabel: str='Parameter', ylabel: str='State', show: bool=True,
    fig=None, ax=None, **style,
):
    """Plot the long-run states of ``x[n+1] = function(x[n], parameter)``."""
    parameter_range = _interval(parameter_range, 'parameter_range')
    parameter_samples = _count(parameter_samples, 'parameter_samples', 2)
    burn_in = _count(burn_in, 'burn_in', 0)
    keep = _count(keep, 'keep', 1)
    max_points = _count(max_points, 'max_points', 1)
    max_iterations = _count(max_iterations, 'max_iterations', 1)
    if parameter_samples * keep > max_points:
        raise ValueError('parameter_samples * keep exceeds max_points')
    if parameter_samples * (burn_in + keep) > max_iterations:
        raise ValueError('requested bifurcation work exceeds max_iterations')
    initial_state, escape = float(initial_state), float(escape)
    if not math.isfinite(initial_state):
        raise ValueError('initial_state must be finite')
    if not math.isfinite(escape) or escape <= 0:
        raise ValueError('escape must be a positive finite number')
    if not callable(function):
        raise TypeError('function must be callable')
    parameters = np.linspace(*parameter_range, parameter_samples)
    plotted_parameters, states = [], []
    actual_iterations = 0
    for parameter in parameters:
        state = initial_state
        for iteration in range(burn_in + keep):
            actual_iterations += 1
            try:
                state = float(function(state, float(parameter)))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                state = np.nan
            if not math.isfinite(state) or abs(state) > escape:
                break
            if iteration >= burn_in:
                plotted_parameters.append(parameter)
                states.append(state)
    fig, ax = _figure_and_axes(fig, ax)
    style.setdefault('s', 0.3)
    style.setdefault('color', 'black')
    style.setdefault('rasterized', True)
    artist = ax.scatter(plotted_parameters, states, **style)
    artist.kiwicalc_parameters = np.asarray(plotted_parameters, dtype=float)
    artist.kiwicalc_states = np.asarray(states, dtype=float)
    artist.kiwicalc_iterations = actual_iterations
    ax.set(xlim=parameter_range, title=title or '', xlabel=xlabel or '', ylabel=ylabel or '')
    if show:
        plt.show()
    return artist


__all__ = [
    'PhasePortraitResult', 'TransformPlotResult', 'plot_phase_portrait',
    'plot_complex_function', 'plot_convergence', 'plot_transform',
    'plot_bifurcation',
]
