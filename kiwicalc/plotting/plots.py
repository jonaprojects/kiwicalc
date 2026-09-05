from __future__ import annotations
import math
from math import ceil, sqrt
import cmath
import warnings
from itertools import combinations, cycle
from pathlib import Path
from typing import Union, Tuple, List, Optional, Any, Callable, Iterable, Sequence
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kiwicalc.core.interfaces import IExpression, IPlottable
from kiwicalc.core.utils import (
    decimal_range, is_lambda, create_grid, draw_axis,
    format_matplot
)
from kiwicalc.core.ranges import values_in_range
from kiwicalc.geometry.curves import Curve2D
from kiwicalc.geometry.points import Circle, process_to_points
from kiwicalc.functions.function import Function
from kiwicalc.plotting.axis import (
    axis_label, configure_minor_ticks, configure_ticks, normalize_units,
    set_axes_at_origin,
)
from kiwicalc.plotting.themes import PlotTheme, ThemeInput, apply_theme, get_theme
from kiwicalc.plotting.motion import GraphAnimation, GraphInteraction
from kiwicalc.plotting.sampling import normalize_sampling, sample_for_plot


FillBoundary = Union[int, float, str, Callable[[float], Any], IExpression, Function, Curve2D]
FieldInput = Union[int, float, str, Callable[[float, float], Any], IExpression, Function]
AxisRange = Optional[Iterable[float]]
FieldDensity = Union[int, Sequence[int]]
ContourLevels = Union[int, Iterable[float]]


def _validate_positive_step(step):
    try:
        numeric = float(step)
    except (TypeError, ValueError):
        raise ValueError('step must be a positive finite number')
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError('step must be a positive finite number')


def _sample_plot_points(func, start, stop, step, ymin, ymax, values, sampling,
                        tolerance, max_points, max_depth):
    mode = normalize_sampling(sampling)
    if mode == 'fixed':
        if values is None:
            _validate_positive_step(step)
        sampled_values, results = process_to_points(
            func, start, stop, step, ymin, ymax, values
        )
        return sampled_values, results, None
    sample = sample_for_plot(
        func, start=start, stop=stop, step=step, values=values,
        sampling=mode, tolerance=tolerance, max_points=max_points,
        max_depth=max_depth,
    )
    return sample.x, sample.y, sample


def _figure_and_axes(fig=None, ax=None, projection=None):
    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("fig and ax must refer to the same figure")
        if projection == '3d' and not hasattr(ax, 'zaxis'):
            raise ValueError("a 3D plotting method requires 3D axes")
        return ax.figure, ax
    if fig is None:
        fig = plt.figure() if projection == '3d' else plt.subplots(figsize=(10, 8))[0]
    if projection == '3d':
        ax = next((candidate for candidate in fig.axes if hasattr(candidate, 'zaxis')), None)
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
    else:
        ax = next((candidate for candidate in fig.axes if not hasattr(candidate, 'zaxis')), None)
        if ax is None:
            ax = fig.add_subplot(111)
    return fig, ax


def plot_curve_2d(curve, show=True, fig=None, ax=None, label=None, **style):
    """Plot any sampleable 2D curve and return its Matplotlib line."""
    fig, ax = _figure_and_axes(fig, ax)
    x_values, y_values = curve.sample()
    line, = ax.plot(x_values, y_values, label=label, **style)
    if show:
        plt.show()
    return line


def scatter_curve_2d(curve, show=True, fig=None, ax=None, label=None, **style):
    """Scatter the sampled points of a 2D curve."""
    fig, ax = _figure_and_axes(fig, ax)
    x_values, y_values = curve.sample()
    artist = ax.scatter(x_values, y_values, label=label, **style)
    if show:
        plt.show()
    return artist


def plot_implicit_curve_2d(curve, show=True, fig=None, ax=None, label=None, **style):
    """Plot an implicit curve with a zero-level contour."""
    fig, ax = _figure_and_axes(fig, ax)
    X, Y, Z = curve.sample()
    contour = ax.contour(X, Y, Z, levels=[curve.level], **style)
    if label:
        if hasattr(contour, 'set_label'):
            contour.set_label('_nolegend_')
        elif getattr(contour, 'collections', None):
            contour.collections[0].set_label('_nolegend_')
        proxy_style = {}
        color = style.get('colors', style.get('color'))
        if isinstance(color, (list, tuple)) and color:
            color = color[0]
        if color is not None:
            proxy_style['color'] = color
        linewidth = style.get('linewidths', style.get('linewidth'))
        if linewidth is not None:
            proxy_style['linewidth'] = linewidth
        proxy, = ax.plot([], [], label=label, **proxy_style)
        contour._kiwicalc_proxy = proxy
    if show:
        plt.show()
    return contour


def plot_curve_3d(curve, show=True, fig=None, ax=None, label=None, **style):
    """Plot any sampleable 3D curve and return its Matplotlib line."""
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    x_values, y_values, z_values = curve.sample()
    line, = ax.plot(x_values, y_values, z_values, label=label, **style)
    if show:
        plt.show()
    return line


def scatter_curve_3d(curve, show=True, fig=None, ax=None, label=None, **style):
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    x_values, y_values, z_values = curve.sample()
    artist = ax.scatter(x_values, y_values, z_values, label=label, **style)
    if show:
        plt.show()
    return artist


def plot_surface_3d(surface, show=True, fig=None, ax=None, label=None, wireframe=False, **style):
    """Plot a sampleable 3D surface and return its Matplotlib artist."""
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    X, Y, Z = surface.sample()
    if wireframe:
        artist = ax.plot_wireframe(X, Y, Z, **style)
    else:
        artist = ax.plot_surface(X, Y, Z, **style)
    if label:
        artist.set_label(label)
    if show:
        plt.show()
    return artist

def scatter_dots(x_values, y_values, title: str='', ymin: float=-10, ymax: float=10, color=None, show_axis=True, show=True, fig=None, ax=None, **style):
    if (length := len(x_values)) != (y_length := len(y_values)):
        raise ValueError(f'You must enter an equal number of x and y values. Got {length} x values and {y_length} y values.')
    fig, ax = _figure_and_axes(fig, ax)
    if show_axis:
        draw_axis(ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(ymin, ymax)
    style.setdefault('s', 90)
    if color is not None:
        style.setdefault('c', color)
    artist = ax.scatter(x=x_values, y=y_values, **style)
    if show:
        plt.show()
    return artist

def scatter_dots_3d(x_values, y_values, z_values, title: str='', xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', fig=None, ax=None, show=True, write_labels=True, label=None, **style):
    lengths = tuple(map(len, (x_values, y_values, z_values)))
    if len(set(lengths)) != 1:
        raise ValueError("x, y, and z values must have equal lengths")
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    ax.set_title(title)
    artist = ax.scatter(x_values, y_values, z_values, label=label, **style)
    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    if show:
        plt.show()
    return artist

def scatter_function(func: Union[Callable, str], start: float=-10, stop: float=10, step: float=0.5, ymin: float=-10, ymax: float=10, title='', color=None, show_axis=True, show=True, fig=None, ax=None, values=None, sampling='fixed', tolerance=1e-3, max_points=5000, max_depth=12):
    mode = normalize_sampling(sampling)
    sample = None
    if mode == 'fixed':
        if isinstance(func, str):
            func = Function(func)
        if values is not None:
            results = [func(value) for value in values]
        else:
            _validate_positive_step(step)
            values, results = values_in_range(func, start, stop, step)
    else:
        values, results, sample = _sample_plot_points(
            func, start, stop, step, ymin, ymax, values, mode,
            tolerance, max_points, max_depth,
        )
    artist = scatter_dots(values, results, title=title, ymin=ymin, ymax=ymax, color=color, show_axis=show_axis, show=show, fig=fig, ax=ax)
    if sample is not None:
        artist.kiwicalc_sample = sample
    return artist

def scatter_function_3d(func: 'Union[Callable, str, IExpression]', start: float=-3, stop: float=3, step: float=0.3, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None, write_labels=True, meshgrid=None, title='', label=None, **style):
    if isinstance(func, str):
        func = Function(func)
    if meshgrid is None:
        _validate_positive_step(step)
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("meshgrid arrays must have matching shapes")
    zs = np.empty(X.size, dtype=float)
    for index, (x_value, y_value) in enumerate(zip(np.ravel(X), np.ravel(Y))):
        try:
            value = float(func(x_value, y_value))
            zs[index] = value if math.isfinite(value) else np.nan
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            zs[index] = np.nan
    Z = zs.reshape(X.shape)
    return scatter_dots_3d(X.ravel(), Y.ravel(), Z.ravel(), fig=fig, ax=ax, title=title, show=show, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, write_labels=write_labels, label=label, **style)

def scatter_functions_3d(functions: 'Iterable[Union[Callable, str, IExpression]]', start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None):
    _validate_positive_step(step)
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    x = y = np.arange(start, stop, step)
    meshgrid = np.meshgrid(x, y)
    artists = []
    for func in functions:
        before = len(ax.collections)
        scatter_function_3d(
            func,
            show=False,
            write_labels=False,
            fig=fig,
            ax=ax,
            meshgrid=meshgrid,
        )
        artists.extend(ax.collections[before:])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()
    return artists

def plot_function(func: Union[Callable, str], start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title=None, show_axis=True, show=True, fig=None, ax=None, formatText=False, values=None, label=None, sampling='fixed', tolerance=1e-3, max_points=5000, max_depth=12, **style):
    fig, ax = _figure_and_axes(fig, ax)
    if show_axis:
        draw_axis(ax)
    values, results, sample = _sample_plot_points(
        func, start, stop, step, ymin, ymax, values, sampling,
        tolerance, max_points, max_depth,
    )
    if title is not None:
        if formatText:
            ax.set_title(f'${format_matplot(title)}$', fontsize=14)
        else:
            ax.set_title(f'{title}', fontsize=14)
    ax.set_ylim(ymin, ymax)
    line, = ax.plot(values, results, label=label, **style)
    if sample is not None:
        line.kiwicalc_sample = sample
    if show:
        plt.show()
    return line

def plot_function_3d(given_function: 'Union[Callable, str, IExpression]', start: float=-3, stop: float=3, step: float=0.3, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None, write_labels=True, meshgrid=None, label=None, wireframe=False, **style):
    if meshgrid is None:
        _validate_positive_step(step)
    if step < 0.1:
        step = 0.3
        warnings.warn('step parameter modified to 0.3 to avoid lag when plotting in 3D')
    if isinstance(given_function, str):
        given_function = Function(given_function)
    elif isinstance(given_function, IExpression):
        num_of_variables = len(given_function.variables)
        if num_of_variables != 2:
            raise ValueError(f'Invalid expression: {given_function}. Found {num_of_variables} variables, expected 2.')
        if hasattr(given_function, 'to_lambda'):
            given_function = given_function.to_lambda()
        elif hasattr(given_function, '__call__'):
            pass
        else:
            raise ValueError(f"This type of algebraic expression isn't supported for plotting in 3D!")
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("meshgrid arrays must have matching shapes")
    zs = np.empty(X.size, dtype=float)
    for index, (x_value, y_value) in enumerate(zip(np.ravel(X), np.ravel(Y))):
        try:
            result = given_function(x_value, y_value)
            if result is None:
                result = np.nan
            result = float(result)
            zs[index] = result if math.isfinite(result) else np.nan
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            zs[index] = np.nan
    Z = zs.reshape(X.shape)
    artist = ax.plot_wireframe(X, Y, Z, **style) if wireframe else ax.plot_surface(X, Y, Z, **style)
    if label:
        artist.set_label(label)
    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    if show:
        plt.show()
    return artist

def plot_functions_3d(functions: 'Iterable[Union[Callable, str, IExpression]]', start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None):
    _validate_positive_step(step)
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    x = y = np.arange(start, stop, step)
    artists = []
    for func in functions:
        artists.append(plot_function_3d(func, show=False, write_labels=False, fig=fig, ax=ax, meshgrid=np.meshgrid(x, y)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()
    return artists

def plot_functions(functions, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title: str=None, formatText: bool=False, show_axis: bool=True, show: bool=True, with_legend=True, fig=None, ax=None, sampling='fixed', tolerance=1e-3, max_points=5000, max_depth=12):
    mode = normalize_sampling(sampling)
    _validate_positive_step(step)
    fig, ax = _figure_and_axes(fig, ax)
    if show_axis:
        draw_axis(ax)
    values = np.arange(start, stop, step) if mode == 'fixed' else None
    ax.set_ylim(ymin, ymax)
    if title is not None:
        if formatText:
            ax.set_title(f'${format_matplot(title)}$', fontsize=14)
        else:
            ax.set_title(title, fontsize=14)
    artists = []
    for given_function in functions:
        if isinstance(given_function, str):
            label = given_function
            given_function = Function(given_function).lambda_expression
        elif isinstance(given_function, Function):
            label = given_function.function_string
            given_function = given_function.lambda_expression
        elif isinstance(given_function, IExpression):
            label = given_function.__str__()
            if hasattr(given_function, 'to_lambda'):
                given_function = given_function.to_lambda()
            else:
                raise ValueError(f'Invalid algebraic expression for plotting: {given_function}')
        else:
            label = None
        if mode == 'fixed':
            sampled_values = values
            results = [given_function(value) for value in values]
            sample = None
        else:
            sample = sample_for_plot(
                given_function, start=start, stop=stop, step=step,
                sampling=mode, tolerance=tolerance, max_points=max_points,
                max_depth=max_depth,
            )
            sampled_values, results = sample.x, sample.y
        line, = ax.plot(sampled_values, results, label=label)
        if sample is not None:
            line.kiwicalc_sample = sample
        artists.append(line)
    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and any(not item.startswith('_') for item in labels):
            ax.legend()
    if show:
        plt.show()
    return artists

def scatter_functions(functions, start: float=-10, stop: float=10, step: float=0.5, ymin: float=-10, ymax: float=10, title: str=None, show_axis: bool=True, show: bool=True, fig=None, ax=None, sampling='fixed', tolerance=1e-3, max_points=5000, max_depth=12):
    mode = normalize_sampling(sampling)
    _validate_positive_step(step)
    fig, ax = _figure_and_axes(fig, ax)
    cycol = cycle('bgrcmykw')
    values = np.arange(start, stop, step) if mode == 'fixed' else None
    artists = []
    for current_function in functions:
        artists.append(scatter_function(func=current_function, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title, color=next(cycol), show_axis=show_axis, show=False, fig=fig, ax=ax, values=values, sampling=mode, tolerance=tolerance, max_points=max_points, max_depth=max_depth))
    if show:
        plt.show()
    return artists

def plot_vector_2d(x_start: float, y_start: float, x_distance: float, y_distance: float, show=True, fig=None, ax=None, label=None, **style):
    fig, ax = _figure_and_axes(fig, ax)
    style.setdefault('head_width', 0.1)
    style.setdefault('width', 0.01)
    artist = ax.arrow(x_start, y_start, x_distance, y_distance, label=label, **style)
    if show:
        plt.show()
    return artist

def plot_vector_3d(starts: Tuple[float, float, float], distances: Tuple[float, float, float], arrow_length_ratio=0.08, show=True, fig=None, ax=None, label=None, **style):
    """plot a 3d vector"""
    u, v, w = distances
    start_x, start_y, start_z = starts
    supplied = fig is not None or ax is not None
    fig, ax = _figure_and_axes(fig, ax, projection='3d')
    if not supplied:
        ax.set_xlim([start_x, start_x + u])
        ax.set_ylim([start_y, start_y + v])
        ax.set_zlim([start_z, start_z + w])
    artist = ax.quiver(start_x, start_y, start_z, u, v, w, arrow_length_ratio=arrow_length_ratio, label=label, **style)
    if show:
        plt.show()
    return artist

def plot_complex(*numbers: complex, title: str='', show=True, fig=None, ax=None, **style):
    """
    plot complex numbers on the complex plane

    :param numbers: The complex numbers to be plotted
    :param show: If set to false, the plotted
    :return: fig, ax
    """
    if not numbers:
        raise ValueError('at least one complex number is required')
    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("fig and ax must refer to the same figure")
        if getattr(ax, 'name', None) != 'polar':
            raise ValueError('plot_complex requires polar axes')
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    ax.set_title(title, va='bottom')
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    max_radius = abs(numbers[0])
    for c in numbers:
        radius = abs(c)
        if radius > max_radius:
            max_radius = radius
        ax.scatter(cmath.phase(c), radius, **style)
    radial_max = max(max_radius * 1.25, 1.0)
    ax.set_rticks(np.linspace(0, radial_max, num=5))
    ax.set_rmax(radial_max)
    if show:
        plt.show()
    return (fig, ax)

def generate_subplot_shape(num_of_functions: int):
    if not isinstance(num_of_functions, int) or isinstance(num_of_functions, bool) or num_of_functions < 1:
        raise ValueError('num_of_functions must be a positive integer')
    square_root = sqrt(num_of_functions)
    if square_root == int(square_root):
        return (int(square_root), int(square_root))
    try:
        result = min([(first, second) for first, second in combinations(range(1, num_of_functions), 2) if first * second == num_of_functions], key=lambda x: abs(x[1] - x[0]))
        if result[0] > result[1]:
            return (result[1], result[0])
        return result
    except ValueError:
        return (ceil(square_root), ceil(square_root))


def _subplot_title(function):
    """Return descriptive formula text, never a callable's Python repr."""
    if isinstance(function, str):
        return function
    if isinstance(function, Function):
        return function.function_string
    if isinstance(function, IExpression):
        return str(function)
    return None


def plot_multiple(funcs, shape: Tuple[int, int]=None, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title=None, show_axis=True, show=True, values=None, subplot_titles=None, sampling='fixed', tolerance=1e-3, max_points=5000, max_depth=12):
    mode = normalize_sampling(sampling)
    num_of_functions = len(funcs)
    if num_of_functions < 1:
        raise ValueError('funcs must contain at least one function')
    if shape is None:
        shape = generate_subplot_shape(num_of_functions)
    try:
        rows, columns = map(int, shape)
    except (TypeError, ValueError):
        raise ValueError('shape must be a (rows, columns) pair')
    if rows < 1 or columns < 1:
        raise ValueError('shape rows and columns must be positive')
    if rows * columns < num_of_functions:
        raise ValueError('shape must have room for every function')
    if values is None:
        _validate_positive_step(step)
    if subplot_titles is not None:
        subplot_titles = tuple(subplot_titles)
        if len(subplot_titles) != num_of_functions:
            raise ValueError('subplot_titles must contain one title per function')
    fig, ax = plt.subplots(rows, columns, squeeze=False)
    fig.tight_layout()
    func_index = 0
    for i in range(rows):
        if func_index >= num_of_functions:
            break
        for j in range(columns):
            if func_index >= num_of_functions:
                break
            sampled_values, results, sample = _sample_plot_points(
                funcs[func_index], start, stop, step, ymin, ymax, values,
                mode, tolerance, max_points, max_depth,
            )
            current_ax = ax[i, j]
            item_title = subplot_titles[func_index] if subplot_titles is not None else _subplot_title(funcs[func_index])
            line, = current_ax.plot(sampled_values, results, label=item_title)
            if sample is not None:
                line.kiwicalc_sample = sample
            if item_title:
                current_ax.set_title(str(item_title))
            if show_axis:
                draw_axis(current_ax)
            func_index += 1
    if title is not None:
        fig.suptitle(title)
    for unused in np.asarray(ax).reshape(-1)[num_of_functions:]:
        unused.remove()
    if title is not None:
        fig.subplots_adjust(top=0.9)
    if show:
        try:
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
        except:
            warnings.warn("Couldn't plot in full screen!")
        plt.show()
    return fig, ax

class Graph:

    def __init__(self, objs, fig, ax):
        self._items = [obj for obj in objs]
        self._item_options = [dict(label=None, visible=True, style={}) for _ in self._items]
        self._fig, self._ax = (fig, ax)
        self._artists = []
        self._decorations = []
        self._restored_view = {}
        self._has_plotted = False
        self._theme = None
        self._axis_options = {}
        self._secondary_axis_specs = []
        self._secondary_axes = []
        self._colorbars = []
        self._legend_artist = None
        self._animations = []
        self._interactions = []

    @property
    def items(self):
        return self._items

    @property
    def fig(self):
        self._ensure_figure()
        return self._fig

    @property
    def ax(self):
        self._ensure_figure()
        return self._ax

    def _ensure_figure(self):
        """Create deferred plotting state when a graph implementation needs it."""
        if self._fig is None or self._ax is None:
            raise RuntimeError(f"{type(self).__name__} has no figure or axes")

    @property
    def artists(self):
        return list(self._artists)

    @property
    def sampling_results(self):
        """Adaptive sampling diagnostics for artists from the latest render."""
        results = []
        for artist in self._artists:
            candidates = artist if isinstance(artist, (list, tuple)) else (artist,)
            for candidate in candidates:
                sample = getattr(candidate, 'kiwicalc_sample', None)
                if sample is not None:
                    results.append(sample)
        return tuple(results)

    def is_empty(self):
        return not self._items

    def add(self, obj, *, label=None, visible=True, **style):
        self._items.append(obj)
        self._item_options.append(dict(label=label, visible=bool(visible), style=dict(style)))
        return self

    def _entries(self):
        while len(self._item_options) < len(self._items):
            self._item_options.append(dict(label=None, visible=True, style={}))
        del self._item_options[len(self._items):]
        return zip(self._items, self._item_options)

    def remove(self, obj):
        for index, item in enumerate(self._items):
            if item is obj:
                self._items.pop(index)
                self._item_options.pop(index)
                return obj
        raise ValueError('The object is not in this graph')

    @staticmethod
    def _remove_artist(artist):
        if isinstance(artist, (list, tuple)):
            for child in reversed(artist):
                Graph._remove_artist(child)
            return
        proxy = getattr(artist, '_kiwicalc_proxy', None)
        if proxy is not None:
            Graph._remove_artist(proxy)
        try:
            artist.remove()
        except (AttributeError, KeyError, NotImplementedError, ValueError):
            pass

    def _clear_rendered(self):
        """Remove artists created by the previous render without touching user artists."""
        for colorbar in reversed(self._colorbars):
            try:
                colorbar.remove()
            except (KeyError, ValueError):
                pass
        self._colorbars.clear()
        for secondary in reversed(self._secondary_axes):
            try:
                secondary.remove()
            except (KeyError, ValueError):
                pass
        self._secondary_axes.clear()
        if self._legend_artist is not None:
            self._remove_artist(self._legend_artist)
            self._legend_artist = None
        for artist in reversed(self._artists):
            self._remove_artist(artist)
        self._artists.clear()

    def clear(self):
        self._clear_rendered()
        self._items.clear()
        self._item_options.clear()
        self._decorations.clear()
        if self._ax is not None:
            self._ax.set_title('')
        for animation in self._animations:
            animation.pause()
        self._animations.clear()
        for interaction in self._interactions:
            interaction.disconnect()
            try:
                interaction.slider.ax.remove()
            except (KeyError, ValueError):
                pass
        self._interactions.clear()
        return self

    def theme(self, theme: ThemeInput = None, **overrides):
        """Choose a graph-local theme and return this graph for chaining."""
        self._theme = get_theme(theme, **overrides)
        return self

    def configure_axes(self, **options):
        """Store axis options for later ``plot`` calls and return this graph."""
        allowed = {
            'xlabel', 'ylabel', 'zlabel', 'units', 'x_ticks', 'y_ticks', 'z_ticks',
            'origin', 'minor_ticks', 'minor_grid', 'xscale', 'yscale', 'zscale',
            'pi_step', 'degree_step',
        }
        unknown = set(options) - allowed
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f"Unknown axis option(s): {names}")
        self._axis_options.update(options)
        return self

    @staticmethod
    def _export_format(format, suffix=''):
        chosen_format = format or suffix.lstrip('.') or 'png'
        if not isinstance(chosen_format, str):
            raise TypeError("Graph export format must be a string")
        chosen_format = chosen_format.lower()
        if chosen_format not in {'png', 'svg', 'pdf'}:
            raise ValueError("Graph export format must be PNG, SVG, or PDF")
        return chosen_format

    def _prepare_export(self, render, plot_options):
        if render is not None and not isinstance(render, bool):
            raise TypeError("render must be True, False, or None")
        options = dict(plot_options or {})
        if options and render is False:
            raise ValueError("plot_options require rendering to be enabled")
        should_render = bool(options) or render is True or (render is None and not self._has_plotted)
        if should_render:
            options.pop('return_artists', None)
            options['show'] = False
            self.plot(**options)
        else:
            self._ensure_figure()

    @staticmethod
    def _export_options(dpi, transparent, tight, chosen_format, kwargs):
        try:
            numeric_dpi = float(dpi)
        except (TypeError, ValueError):
            raise ValueError("dpi must be a positive finite number")
        if isinstance(dpi, bool) or not math.isfinite(numeric_dpi) or numeric_dpi <= 0:
            raise ValueError("dpi must be a positive finite number")
        options = dict(dpi=numeric_dpi, transparent=bool(transparent), format=chosen_format)
        if tight:
            options['bbox_inches'] = 'tight'
        options.update(kwargs)
        return options

    def save(self, path, *, dpi=300, transparent=False, tight=True, format=None,
             render=None, plot_options=None, **kwargs):
        """Save this graph as PNG, SVG, or PDF and return the resulting path.

        An unrendered graph is plotted automatically. Pass ``plot_options`` to
        control that render, ``render=True`` to force a fresh render, or
        ``render=False`` to export the current Matplotlib figure unchanged.
        """
        output = Path(path)
        chosen_format = self._export_format(format, output.suffix)
        if output.suffix and output.suffix.lstrip('.').lower() != chosen_format:
            raise ValueError("Graph export format must match the file extension")
        if not output.suffix:
            output = output.with_suffix(f'.{chosen_format}')
        self._prepare_export(render, plot_options)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_options = self._export_options(dpi, transparent, tight, chosen_format, kwargs)
        self._fig.savefig(output, **save_options)
        return output

    def export(self, path, **kwargs):
        """Alias for :meth:`save`."""
        return self.save(path, **kwargs)

    def to_bytes(self, *, format='png', dpi=300, transparent=False, tight=True,
                 render=None, plot_options=None, **kwargs):
        """Return an in-memory PNG, SVG, or PDF representation of this graph."""
        from io import BytesIO

        chosen_format = self._export_format(format)
        self._prepare_export(render, plot_options)
        save_options = self._export_options(dpi, transparent, tight, chosen_format, kwargs)
        data = BytesIO()
        self._fig.savefig(data, **save_options)
        return data.getvalue()

    def to_svg(self, *, encoding='utf-8', **kwargs):
        """Return this graph as SVG text."""
        return self.to_bytes(format='svg', **kwargs).decode(encoding)

    def to_dict(self):
        from kiwicalc.serialization import graph_to_dict
        return graph_to_dict(self)

    def to_json(self, **kwargs):
        import json
        return json.dumps(self.to_dict(), **kwargs)

    def export_json(self, path, **kwargs):
        from pathlib import Path
        Path(path).write_text(self.to_json(**kwargs), encoding='utf-8')
        return Path(path)

    @staticmethod
    def from_dict(data):
        from kiwicalc.serialization import graph_from_dict
        return graph_from_dict(data)

    @staticmethod
    def from_json(value):
        import json
        from pathlib import Path
        if isinstance(value, Path):
            value = value.read_text(encoding='utf-8')
        elif isinstance(value, str) and not value.lstrip().startswith(('{', '[')):
            candidate = Path(value)
            if candidate.is_file():
                value = candidate.read_text(encoding='utf-8')
        return Graph.from_dict(json.loads(value))

    def plot(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def scatter(self):
        raise NotImplementedError

class Graph2D(Graph):

    def __init__(self, objs: Iterable[IPlottable]=tuple()):
        super(Graph2D, self).__init__(objs, None, None)

    def _ensure_figure(self):
        if self._fig is None or self._ax is None:
            self._fig, self._ax = create_grid()

    def mark(self, point, label=None, **style):
        from kiwicalc.geometry.points import Point2D
        if isinstance(point, Point2D):
            marker = point
        else:
            try:
                marker = Point2D(*point)
            except (TypeError, ValueError):
                raise ValueError("point must contain exactly two coordinates")
        style.setdefault('color', 'crimson')
        style.setdefault('s', 60)
        return self.add(marker, label=label, **style)

    def annotate(self, text, at, offset=(6, 6), **style):
        try:
            point = tuple(at.coordinates) if hasattr(at, 'coordinates') else tuple(at)
        except TypeError:
            raise ValueError("at must contain exactly two coordinates")
        if len(point) != 2:
            raise ValueError("at must contain exactly two coordinates")
        self._decorations.append(dict(kind='annotation', text=str(text), at=point, offset=tuple(offset), style=dict(style)))
        return self

    def vertical_line(self, x, label=None, **style):
        self._decorations.append(dict(kind='vertical', value=float(x), label=label, style=dict(style)))
        return self

    def horizontal_line(self, y, label=None, **style):
        self._decorations.append(dict(kind='horizontal', value=float(y), label=label, style=dict(style)))
        return self

    def _explain(self, kind, **options):
        options['kind'] = kind
        self._decorations.append(options)
        return self

    def show_roots(self, function, *, domain=None, samples=1201, label=True, **style):
        """Mark the real roots visible in ``domain`` (or the plotted domain)."""
        return self._explain('roots', source=function, domain=domain, samples=samples, label=label, style=dict(style))

    def show_intersections(self, first, second, *, domain=None, samples=1201, label=True, **style):
        """Mark intersections between two function-like objects."""
        return self._explain('intersections', first=first, second=second, domain=domain, samples=samples, label=label, style=dict(style))

    def show_extrema(self, function, *, domain=None, samples=1201, label=True, **style):
        """Mark and classify local minima and maxima."""
        return self._explain('extrema', source=function, domain=domain, samples=samples, label=label, style=dict(style))

    def show_inflections(self, function, *, domain=None, samples=1201, label=True, **style):
        """Mark approximate inflection points."""
        return self._explain('inflections', source=function, domain=domain, samples=samples, label=label, style=dict(style))

    def tangent(self, function_or_curve, at, *, span=None, label='tangent', **style):
        """Draw a tangent at x=``at`` or normalized curve position ``at``."""
        return self._explain('tangent', source=function_or_curve, at=float(at), span=span, label=label, style=dict(style))

    def normal(self, function_or_curve, at, *, span=None, label='normal', **style):
        """Draw a normal at x=``at`` or normalized curve position ``at``."""
        return self._explain('normal', source=function_or_curve, at=float(at), span=span, label=label, style=dict(style))

    def secant(self, function, between, *, label='secant', **style):
        """Draw the secant segment between two x coordinates."""
        between = tuple(between)
        if len(between) != 2:
            raise ValueError("between must contain two x coordinates")
        return self._explain('secant', source=function, between=between, domain=None, label=label, style=dict(style))

    def show_asymptotes(self, function, *, domain=None, vertical='auto', horizontal='auto', samples=1201, **style):
        """Draw automatic or explicitly supplied vertical/horizontal asymptotes."""
        return self._explain('asymptotes', source=function, domain=domain, samples=samples, vertical=vertical, horizontal=horizontal, style=dict(style))

    def show_monotonicity(self, function, *, domain=None, samples=1201, colors=('#54a24b', '#e45756'), **style):
        """Shade increasing regions green and decreasing regions red."""
        return self._explain('monotonicity', source=function, domain=domain, samples=samples, colors=tuple(colors), style=dict(style))

    def shade_solution(self, function, relation='>=', other=0, *, domain=None, samples=1201, label=None, **style):
        """Shade where ``function relation other`` is true."""
        return self._explain('solution', source=function, other=other, relation=relation, domain=domain, samples=samples, label=label, style=dict(style))

    def riemann_sum(self, function, interval, *, rectangles=8, method='midpoint', **style):
        """Show a left, right, midpoint, or trapezoidal integration estimate."""
        interval = tuple(interval)
        if len(interval) != 2:
            raise ValueError("interval must contain two bounds")
        return self._explain('riemann', source=function, interval=interval, rectangles=rectangles, method=method, domain=interval, style=dict(style))

    def slope_triangle(self, function, at, *, run=1, label=True, **style):
        """Draw rise and run legs using the local derivative at x=``at``."""
        return self._explain('slope_triangle', source=function, at=float(at), run=float(run), domain=None, label=label, style=dict(style))

    def show_derivative(self, function, *, domain=None, samples=1201, label='derivative', **style):
        """Overlay a numerical derivative."""
        return self._explain('derivative', source=function, domain=domain, samples=samples, label=label, style=dict(style))

    def show_integral(self, function, *, domain=None, samples=1201, constant=0, label='integral', **style):
        """Overlay a cumulative numerical integral."""
        return self._explain('integral', source=function, domain=domain, samples=samples, constant=constant, label=label, style=dict(style))

    def vector_field(
        self, u: FieldInput, v: FieldInput, *, x_range: AxisRange=None,
        y_range: AxisRange=None, density: FieldDensity=20,
        normalize: bool=False, color: Any='magnitude', colorbar: bool=False,
        colorbar_label: Optional[str]='Magnitude', **style: Any,
    ) -> 'Graph2D':
        """Draw vectors ``(u(x, y), v(x, y))`` over a rectangular domain."""
        return self._explain(
            'vector_field', u=u, v=v, x_range=x_range, y_range=y_range,
            density=density, normalize=bool(normalize), color=color,
            colorbar=bool(colorbar), colorbar_label=colorbar_label,
            style=dict(style),
        )

    def slope_field(
        self, derivative: FieldInput, *, x_range: AxisRange=None,
        y_range: AxisRange=None, density: FieldDensity=20,
        normalize: bool=True, color: Any='black', **style: Any,
    ) -> 'Graph2D':
        """Draw the direction field for ``dy/dx = derivative(x, y)``."""
        return self._explain(
            'slope_field', source=derivative, x_range=x_range, y_range=y_range,
            density=density, normalize=bool(normalize), color=color,
            colorbar=False, style=dict(style),
        )

    def gradient_field(
        self, function: FieldInput, *, x_range: AxisRange=None,
        y_range: AxisRange=None, density: FieldDensity=20,
        normalize: bool=False, color: Any='magnitude', colorbar: bool=False,
        colorbar_label: Optional[str]='Gradient magnitude', **style: Any,
    ) -> 'Graph2D':
        """Draw the numerical gradient of a scalar function ``f(x, y)``."""
        return self._explain(
            'gradient_field', source=function, x_range=x_range, y_range=y_range,
            density=density, normalize=bool(normalize), color=color,
            colorbar=bool(colorbar), colorbar_label=colorbar_label,
            style=dict(style),
        )

    def streamlines(
        self, u: FieldInput, v: FieldInput, *, x_range: AxisRange=None,
        y_range: AxisRange=None, samples: int=60, density: float=1.0,
        color: Any='magnitude', colorbar: bool=False,
        colorbar_label: Optional[str]='Magnitude', **style: Any,
    ) -> 'Graph2D':
        """Draw continuous flow lines for a two-dimensional vector field."""
        return self._explain(
            'streamlines', u=u, v=v, x_range=x_range, y_range=y_range,
            samples=samples, stream_density=density, color=color,
            colorbar=bool(colorbar), colorbar_label=colorbar_label,
            style=dict(style),
        )

    def contour_map(
        self, function: FieldInput, *, x_range: AxisRange=None,
        y_range: AxisRange=None, levels: ContourLevels=10,
        filled: bool=False, labels: bool=False, samples: int=100,
        colorbar: bool=False, colorbar_label: Optional[str]=None,
        label_size: float=8, **style: Any,
    ) -> 'Graph2D':
        """Draw line or filled contours of a scalar function ``f(x, y)``."""
        return self._explain(
            'contour_map', source=function, x_range=x_range, y_range=y_range,
            levels=levels, filled=bool(filled), labels=bool(labels),
            samples=samples, colorbar=bool(colorbar),
            colorbar_label=colorbar_label, label_size=label_size,
            style=dict(style),
        )

    contour = contour_map
    streamplot = streamlines

    def animate_parameter(
        self, function: Any, frames: Iterable[float], *, parameter: str='a',
        start: float=-10, stop: float=10, samples: int=400,
        values: Optional[Iterable[float]]=None, interval: float=50,
        repeat: bool=True, blit: bool=False, label: Optional[str]=None,
        title: Any=None, show: bool=True, line_style: Optional[dict]=None,
        **plot_options: Any,
    ) -> 'GraphAnimation':
        """Animate ``function(x, parameter)`` across a sequence of frames."""
        from kiwicalc.plotting.motion import animate_parameter
        return animate_parameter(
            self, function, frames, parameter=parameter, start=start, stop=stop,
            samples=samples, values=values, interval=interval, repeat=repeat,
            blit=blit, label=label, title=title, show=show,
            line_style=line_style, **plot_options,
        )

    def interactive_parameter(
        self, function: Any, parameter_range: Iterable[float], *,
        parameter: str='a', initial: Optional[float]=None,
        step: Optional[float]=None, start: float=-10, stop: float=10,
        samples: int=400, values: Optional[Iterable[float]]=None,
        label: Optional[str]=None, title: Any=None, show: bool=True,
        line_style: Optional[dict]=None, **plot_options: Any,
    ) -> 'GraphInteraction':
        """Create a live slider for ``function(x, parameter)``."""
        from kiwicalc.plotting.motion import interactive_parameter
        return interactive_parameter(
            self, function, parameter_range, parameter=parameter,
            initial=initial, step=step, start=start, stop=stop,
            samples=samples, values=values, label=label, title=title,
            show=show, line_style=line_style, **plot_options,
        )

    animate = animate_parameter
    interact = interactive_parameter

    def secondary_xaxis(self, forward, inverse, *, label=None, unit=None, location='top'):
        """Add a converted secondary x-axis, such as radians to degrees."""
        if not callable(forward) or not callable(inverse):
            raise TypeError("forward and inverse must both be callable")
        self._secondary_axis_specs.append(dict(
            axis='x', forward=forward, inverse=inverse, label=label,
            unit=unit, location=location,
        ))
        return self

    def secondary_yaxis(self, forward, inverse, *, label=None, unit=None, location='right'):
        """Add a converted secondary y-axis."""
        if not callable(forward) or not callable(inverse):
            raise TypeError("forward and inverse must both be callable")
        self._secondary_axis_specs.append(dict(
            axis='y', forward=forward, inverse=inverse, label=label,
            unit=unit, location=location,
        ))
        return self

    def _render_secondary_axes(self, theme):
        for secondary in self._secondary_axes:
            secondary.remove()
        self._secondary_axes = []
        for spec in self._secondary_axis_specs:
            factory = self._ax.secondary_xaxis if spec['axis'] == 'x' else self._ax.secondary_yaxis
            secondary = factory(spec['location'], functions=(spec['forward'], spec['inverse']))
            label = axis_label(spec['label'], spec['unit'])
            if spec['axis'] == 'x':
                secondary.set_xlabel(label)
            else:
                secondary.set_ylabel(label)
            if theme is not None:
                secondary.tick_params(colors=theme.foreground, labelsize=theme.font_size)
                axis_object = secondary.xaxis if spec['axis'] == 'x' else secondary.yaxis
                axis_object.label.set_color(theme.foreground)
                axis_object.label.set_fontsize(theme.label_size)
            self._secondary_axes.append(secondary)

    def fill_between(
        self,
        first: FillBoundary,
        second: FillBoundary = 0,
        values: Optional[Iterable[float]] = None,
        label: Optional[str] = None,
        **style: Any,
    ) -> 'Graph2D':
        """Fill the area between two numbers, functions, expressions, or curves."""
        self._decorations.append(dict(kind='fill', first=first, second=second, values=values, label=label, style=dict(style)))
        return self

    @staticmethod
    def _fill_values(item, x_values):
        from kiwicalc.geometry.curves import Curve2D
        if isinstance(item, (int, float)):
            return np.full(len(x_values), float(item))
        if isinstance(item, str):
            item = Function(item)
        if isinstance(item, Function):
            item = item.lambda_expression
        elif isinstance(item, IExpression):
            item = item.to_lambda()
        if isinstance(item, Curve2D):
            curve_x, curve_y = item.sample()
            finite = np.isfinite(curve_x) & np.isfinite(curve_y)
            curve_x, curve_y = np.asarray(curve_x)[finite], np.asarray(curve_y)[finite]
            order = np.argsort(curve_x)
            return np.interp(x_values, curve_x[order], curve_y[order], left=np.nan, right=np.nan)
        if callable(item):
            results = []
            for value in x_values:
                try:
                    results.append(float(item(float(value))))
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    results.append(np.nan)
            return np.asarray(results)
        raise TypeError("fill_between accepts numbers, functions, expressions, strings, or Curve2D objects")

    def plot(
        self, start: float=-10, stop: float=10, step: float=0.01,
        ymin: float=-10, ymax: float=10, title: Optional[str]=None,
        show_axis: bool=True, show: bool=True, formatText: bool=False,
        values: Optional[Iterable[float]]=None, legend: Optional[bool]=None,
        grid: Optional[bool]=None, xlim: AxisRange=None, ylim: AxisRange=None,
        equal_aspect: Optional[bool]=None, return_artists: bool=False,
        text: Optional[str]=None, theme: ThemeInput=None,
        xlabel: Optional[str]=None, ylabel: Optional[str]=None, units: Any=None,
        x_ticks: Any=None, y_ticks: Any=None, origin: Optional[bool]=None,
        minor_ticks: Optional[bool]=None, minor_grid: Optional[bool]=None,
        xscale: Optional[str]=None, yscale: Optional[str]=None,
        pi_step: Optional[float]=None, degree_step: Optional[float]=None,
        sampling: str='fixed', tolerance: float=1e-3,
        max_points: int=5000, max_depth: int=12,
    ):
        """Render the graph, using ``title`` only when an explicit title is wanted.

        ``text`` remains available as a backwards-compatible alias for ``title``.
        """
        self._ensure_figure()
        from kiwicalc.geometry.curves import Curve2D, ImplicitCurve2D
        from kiwicalc.geometry.points import Line2D, Point2D
        from kiwicalc.geometry.vectors import Vector2D

        if title is not None and text is not None:
            raise ValueError("Use either title or text, not both")
        sampling = normalize_sampling(sampling)
        if values is None:
            _validate_positive_step(step)
        self._clear_rendered()
        if theme is not None:
            self._theme = get_theme(theme)
        elif self._theme is None and self._restored_view.get('theme'):
            self._theme = get_theme(self._restored_view['theme'])
        resolved_theme = self._theme
        apply_theme(self._fig, self._ax, resolved_theme)

        supplied_axis_options = {
            key: value for key, value in {
                'xlabel': xlabel, 'ylabel': ylabel, 'units': units,
                'x_ticks': x_ticks, 'y_ticks': y_ticks, 'origin': origin,
                'minor_ticks': minor_ticks, 'minor_grid': minor_grid,
                'xscale': xscale, 'yscale': yscale, 'pi_step': pi_step,
                'degree_step': degree_step,
            }.items() if value is not None
        }
        self._axis_options.update(supplied_axis_options)
        axis_options = dict(self._restored_view.get('axis_options', {}))
        axis_options.update(self._axis_options)
        if values is None:
            values = list(decimal_range(start=start, stop=stop, step=step))
        requested_title = title if title is not None else text
        graph_title = self._restored_view.get('title') if requested_title is None else requested_title
        shown_title = f'${format_matplot(graph_title)}$' if graph_title and formatText else (graph_title or '')
        self._ax.set_title(shown_title)
        for obj, options in self._entries():
            if not options['visible']:
                continue
            label, style = options['label'], dict(options['style'])
            if isinstance(obj, ImplicitCurve2D):
                if resolved_theme is not None:
                    style.setdefault('linewidths', resolved_theme.line_width)
                artist = plot_implicit_curve_2d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Curve2D):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                artist = plot_curve_2d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Function):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                artist = plot_function(obj.lambda_expression, values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, sampling=sampling, tolerance=tolerance, max_points=max_points, max_depth=max_depth, **style)
            elif isinstance(obj, IExpression):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                artist = plot_function(obj.to_lambda(), values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, sampling=sampling, tolerance=tolerance, max_points=max_points, max_depth=max_depth, **style)
            elif isinstance(obj, Line2D):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                if obj.to_lambda() is None:
                    artist = self._ax.axvline(obj._point1.x, label=label, **style)
                else:
                    artist = plot_function(obj.to_lambda(), values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, sampling=sampling, tolerance=tolerance, max_points=max_points, max_depth=max_depth, **style)
            elif isinstance(obj, Circle):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                radius = obj.radius.try_evaluate()
                center_x, center_y = obj.center_x.try_evaluate(), obj.center_y.try_evaluate()
                if None in (radius, center_x, center_y):
                    raise ValueError('Can only plot circles with real numbers')
                circle_style = dict(style)
                circle_style.setdefault('fill', False)
                artist = plt.Circle((center_x, center_y), radius, label=label, **circle_style)
                self._ax.add_patch(artist)
                self._ax.set_aspect('equal', adjustable='datalim')
            elif isinstance(obj, Point2D):
                if resolved_theme is not None:
                    style.setdefault('s', resolved_theme.marker_size ** 2)
                artist = self._ax.scatter([obj.x], [obj.y], label=label, **style)
            elif isinstance(obj, Vector2D):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                artist = self._ax.arrow(obj._start_coordinate[0], obj._start_coordinate[1], obj.x_step, obj.y_step, label=label, **style)
            elif callable(obj) or isinstance(obj, str):
                if resolved_theme is not None:
                    style.setdefault('linewidth', resolved_theme.line_width)
                artist = plot_function(obj, values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, sampling=sampling, tolerance=tolerance, max_points=max_points, max_depth=max_depth, **style)
            else:
                raise TypeError(f'{type(obj).__name__} cannot be plotted on Graph2D')
            self._artists.append(artist)
        for decoration in self._decorations:
            kind = decoration['kind']
            if kind == 'annotation':
                artist = self._ax.annotate(
                    decoration['text'],
                    xy=decoration['at'],
                    xytext=decoration['offset'],
                    textcoords='offset points',
                    **decoration['style'],
                )
            elif kind == 'vertical':
                artist = self._ax.axvline(decoration['value'], label=decoration['label'], **decoration['style'])
            elif kind == 'horizontal':
                artist = self._ax.axhline(decoration['value'], label=decoration['label'], **decoration['style'])
            elif kind == 'fill':
                fill_values = decoration['values']
                if fill_values is None:
                    fill_values = values
                x_values = np.asarray(fill_values, dtype=float)
                first_y = self._fill_values(decoration['first'], x_values)
                second_y = self._fill_values(decoration['second'], x_values)
                artist = self._ax.fill_between(x_values, first_y, second_y, label=decoration['label'], **decoration['style'])
            else:
                from kiwicalc.plotting.fields import FIELD_KINDS, render as render_field
                if kind in FIELD_KINDS:
                    default_x_range = xlim if xlim is not None else self._restored_view.get('xlim', (start, stop))
                    default_y_range = ylim if ylim is not None else self._restored_view.get('ylim', (ymin, ymax))
                    field_artists, colorbar = render_field(self._ax, decoration, default_x_range, default_y_range, resolved_theme)
                    self._artists.extend(field_artists)
                    if colorbar is not None:
                        self._colorbars.append(colorbar)
                else:
                    from kiwicalc.plotting.explanations import render
                    self._artists.extend(render(self._ax, decoration, values))
                continue
            self._artists.append(artist)
        xlim = xlim if xlim is not None else self._restored_view.get('xlim', (start, stop))
        ylim = ylim if ylim is not None else self._restored_view.get('ylim', (ymin, ymax))
        if grid is None:
            grid = self._restored_view.get('grid', resolved_theme.grid if resolved_theme else False)
        legend = self._restored_view.get('legend', False) if legend is None else legend
        if equal_aspect is None:
            equal_aspect = self._restored_view.get('equal_aspect')
        if equal_aspect is None:
            equal_aspect = any(
                isinstance(obj, Circle) and options['visible']
                for obj, options in self._entries()
            )

        xscale = axis_options.get('xscale', self._restored_view.get('xscale', 'linear'))
        yscale = axis_options.get('yscale', self._restored_view.get('yscale', 'linear'))
        self._ax.set_xscale(xscale)
        self._ax.set_yscale(yscale)
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)

        units = normalize_units(axis_options.get('units'), ('x', 'y'))
        xlabel = axis_options.get('xlabel', self._restored_view.get('xlabel', ''))
        ylabel = axis_options.get('ylabel', self._restored_view.get('ylabel', ''))
        self._ax.set_xlabel(axis_label(xlabel, units[0]))
        self._ax.set_ylabel(axis_label(ylabel, units[1]))

        origin = axis_options.get('origin', bool(show_axis))
        if origin and (xscale != 'linear' or yscale != 'linear'):
            raise ValueError("origin-centered axes require linear x and y scales")
        set_axes_at_origin(self._ax, bool(origin))

        pi_step = axis_options.get('pi_step', math.pi / 2)
        degree_step = axis_options.get('degree_step', math.pi / 4)
        configure_ticks(self._ax.xaxis, axis_options.get('x_ticks'), unit=units[0], pi_step=pi_step, degree_step=degree_step)
        configure_ticks(self._ax.yaxis, axis_options.get('y_ticks'), unit=units[1], pi_step=pi_step, degree_step=degree_step)

        if axis_options.get('minor_ticks') is None:
            minor_ticks = resolved_theme.minor_grid if resolved_theme else False
        else:
            minor_ticks = bool(axis_options['minor_ticks'])
        if xscale == 'linear':
            configure_minor_ticks(self._ax.xaxis, minor_ticks)
        else:
            self._ax.minorticks_on() if minor_ticks else self._ax.minorticks_off()
        if yscale == 'linear':
            configure_minor_ticks(self._ax.yaxis, minor_ticks)
        else:
            self._ax.minorticks_on() if minor_ticks else self._ax.minorticks_off()

        if axis_options.get('minor_grid') is None:
            minor_grid = resolved_theme.minor_grid if resolved_theme else False
        else:
            minor_grid = bool(axis_options['minor_grid'])
        grid_style = {}
        minor_grid_style = {}
        if resolved_theme is not None:
            grid_style = dict(color=resolved_theme.grid_color, alpha=resolved_theme.grid_alpha)
            minor_grid_style = dict(color=resolved_theme.grid_color, alpha=resolved_theme.minor_grid_alpha)
            apply_theme(self._fig, self._ax, resolved_theme)
        if grid:
            self._ax.grid(True, which='major', **grid_style)
        else:
            self._ax.grid(False, which='major')
        if minor_grid:
            self._ax.grid(True, which='minor', **minor_grid_style)
        else:
            self._ax.grid(False, which='minor')
        if equal_aspect is True:
            self._ax.set_aspect('equal', adjustable='box')
        else:
            self._ax.set_aspect('auto')
        self._render_secondary_axes(resolved_theme)
        if legend:
            handles, labels = self._ax.get_legend_handles_labels()
            if handles and any(not item.startswith('_') for item in labels):
                self._legend_artist = self._ax.legend()
        if show:
            plt.show()
        self._has_plotted = True
        if return_artists:
            return self.artists

    def scatter(self, *args: Any, **kwargs: Any):
        """Render sampleable 2D graph items as points where practical.

        Geometric patches, vectors, implicit contours, and explanatory layers
        keep their natural renderers. All ordinary function and curve lines are
        replaced by scatter artists.
        """
        from matplotlib.lines import Line2D as MatplotlibLine2D
        from kiwicalc.geometry.curves import Curve2D, ImplicitCurve2D

        return_artists = bool(kwargs.pop('return_artists', False))
        show = bool(kwargs.pop('show', True))
        kwargs['return_artists'] = True
        kwargs['show'] = False
        self.plot(*args, **kwargs)
        visible_entries = [
            (obj, options) for obj, options in self._entries()
            if options['visible']
        ]
        for index, (obj, options) in enumerate(visible_entries):
            if index >= len(self._artists):
                break
            eligible = (
                isinstance(obj, (Function, IExpression, Curve2D))
                and not isinstance(obj, ImplicitCurve2D)
            ) or callable(obj) or isinstance(obj, str)
            line = self._artists[index]
            if not eligible or not isinstance(line, MatplotlibLine2D):
                continue
            scatter_style = {
                'color': line.get_color(),
                'label': line.get_label(),
                'alpha': line.get_alpha(),
                'zorder': line.get_zorder(),
                's': (self._theme.marker_size if self._theme else 7) ** 2,
            }
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            sample = getattr(line, 'kiwicalc_sample', None)
            self._remove_artist(line)
            self._artists[index] = self._ax.scatter(x_data, y_data, **scatter_style)
            if sample is not None:
                self._artists[index].kiwicalc_sample = sample
        if show:
            plt.show()
        if return_artists:
            return self.artists

class Graph3D(Graph):

    def __init__(self, objs=tuple()):
        super(Graph3D, self).__init__(objs, None, None)

    def _ensure_figure(self):
        if self._fig is None or self._ax is None:
            self._fig, self._ax = _figure_and_axes(projection='3d')

    def _finish(self, xlabel, ylabel, zlabel, title, legend, grid, xlim, ylim, zlim,
                show, return_artists, resolved_theme=None, axis_options=None):
        view = self._restored_view
        axis_options = axis_options or {}
        xlabel = view.get('xlabel', xlabel)
        ylabel = view.get('ylabel', ylabel)
        zlabel = view.get('zlabel', zlabel)
        title = view.get('title', title) if title is None else title
        legend = view.get('legend', False) if legend is None else legend
        grid = view.get('grid') if grid is None and 'grid' in view else grid
        xlim = view.get('xlim') if xlim is None else xlim
        ylim = view.get('ylim') if ylim is None else ylim
        zlim = view.get('zlim') if zlim is None else zlim
        units = normalize_units(axis_options.get('units'), ('x', 'y', 'z'))
        xlabel = axis_options.get('xlabel', xlabel)
        ylabel = axis_options.get('ylabel', ylabel)
        zlabel = axis_options.get('zlabel', zlabel)
        self._ax.set_xlabel(axis_label(xlabel, units[0]))
        self._ax.set_ylabel(axis_label(ylabel, units[1]))
        self._ax.set_zlabel(axis_label(zlabel, units[2]))
        self._ax.set_title(title or '')
        apply_theme(self._fig, self._ax, resolved_theme)
        if grid is None:
            grid = resolved_theme.grid if resolved_theme else False
        self._ax.grid(bool(grid))
        minor_grid = axis_options.get('minor_grid')
        if minor_grid is None:
            minor_grid = resolved_theme.minor_grid if resolved_theme else False
        self._ax.grid(bool(minor_grid), which='minor')
        self._ax.set_xscale(axis_options.get('xscale', view.get('xscale', 'linear')))
        self._ax.set_yscale(axis_options.get('yscale', view.get('yscale', 'linear')))
        self._ax.set_zscale(axis_options.get('zscale', view.get('zscale', 'linear')))
        for setter, limits in ((self._ax.set_xlim, xlim), (self._ax.set_ylim, ylim), (self._ax.set_zlim, zlim)):
            if limits is not None:
                setter(*limits)
        pi_step = axis_options.get('pi_step', math.pi / 2)
        degree_step = axis_options.get('degree_step', math.pi / 4)
        for axis, name, unit in (
            (self._ax.xaxis, 'x_ticks', units[0]),
            (self._ax.yaxis, 'y_ticks', units[1]),
            (self._ax.zaxis, 'z_ticks', units[2]),
        ):
            configure_ticks(axis, axis_options.get(name), unit=unit, pi_step=pi_step, degree_step=degree_step)
        if axis_options.get('minor_ticks'):
            self._ax.minorticks_on()
        elif axis_options.get('minor_ticks') is False:
            self._ax.minorticks_off()
        if legend:
            handles, labels = self._ax.get_legend_handles_labels()
            if handles and any(not item.startswith('_') for item in labels):
                self._legend_artist = self._ax.legend()
        if show:
            plt.show()
        self._has_plotted = True
        if return_artists:
            return self.artists

    def plot(self, functions=None, start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, title=None, legend=None, grid=None, xlim=None, ylim=None, zlim=None, return_artists=False, theme: ThemeInput=None, units=None, x_ticks=None, y_ticks=None, z_ticks=None, minor_ticks=None, minor_grid=None, xscale=None, yscale=None, zscale=None, pi_step=None, degree_step=None):
        from kiwicalc.geometry.curves import Curve3D
        from kiwicalc.geometry.surfaces import Surface3D
        from kiwicalc.geometry.vectors import Vector3D

        self._ensure_figure()
        self._clear_rendered()
        if theme is not None:
            self._theme = get_theme(theme)
        elif self._theme is None and self._restored_view.get('theme'):
            self._theme = get_theme(self._restored_view['theme'])
        resolved_theme = self._theme
        apply_theme(self._fig, self._ax, resolved_theme)
        supplied_axis_options = {
            key: value for key, value in {
                'units': units, 'x_ticks': x_ticks, 'y_ticks': y_ticks,
                'z_ticks': z_ticks, 'minor_ticks': minor_ticks,
                'minor_grid': minor_grid, 'xscale': xscale, 'yscale': yscale,
                'zscale': zscale, 'pi_step': pi_step, 'degree_step': degree_step,
            }.items() if value is not None
        }
        self._axis_options.update(supplied_axis_options)
        axis_options = dict(self._restored_view.get('axis_options', {}))
        axis_options.update(self._axis_options)

        if functions is not None:
            self._artists = plot_functions_3d(functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=False, fig=self._fig, ax=self._ax)
        else:
            self._artists = []
            for obj, options in self._entries():
                if not options['visible']:
                    continue
                label, style = options['label'], dict(options['style'])
                if isinstance(obj, Curve3D):
                    if resolved_theme is not None:
                        style.setdefault('linewidth', resolved_theme.line_width)
                    artist = plot_curve_3d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
                elif isinstance(obj, Surface3D):
                    wireframe = style.pop('wireframe', False)
                    artist = plot_surface_3d(obj, show=False, fig=self._fig, ax=self._ax, label=label, wireframe=wireframe, **style)
                elif isinstance(obj, Vector3D):
                    if resolved_theme is not None:
                        style.setdefault('linewidth', resolved_theme.line_width)
                    artist = plot_vector_3d(
                        tuple(obj._start_coordinate), tuple(obj._direction_vector),
                        show=False, fig=self._fig, ax=self._ax, label=label, **style
                    )
                elif hasattr(obj, 'to_lambda'):
                    artist = plot_function_3d(obj.to_lambda(), start=start, stop=stop, step=step, show=False, fig=self._fig, ax=self._ax, write_labels=False, label=label, **style)
                elif callable(obj) or isinstance(obj, str) or isinstance(obj, IExpression):
                    artist = plot_function_3d(obj, start=start, stop=stop, step=step, show=False, fig=self._fig, ax=self._ax, write_labels=False, label=label, **style)
                else:
                    raise TypeError(f'{type(obj).__name__} cannot be plotted on Graph3D')
                self._artists.append(artist)
        return self._finish(xlabel, ylabel, zlabel, title, legend, grid, xlim, ylim, zlim, show, return_artists, resolved_theme, axis_options)

    def scatter(self, functions=None, start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, title=None, legend=None, grid=None, return_artists=False, theme: ThemeInput=None, units=None, x_ticks=None, y_ticks=None, z_ticks=None, minor_ticks=None):
        from kiwicalc.geometry.curves import Curve3D

        self._ensure_figure()
        self._clear_rendered()
        if theme is not None:
            self._theme = get_theme(theme)
        resolved_theme = self._theme
        apply_theme(self._fig, self._ax, resolved_theme)
        supplied_axis_options = {
            key: value for key, value in {
                'units': units, 'x_ticks': x_ticks, 'y_ticks': y_ticks,
                'z_ticks': z_ticks, 'minor_ticks': minor_ticks,
            }.items() if value is not None
        }
        self._axis_options.update(supplied_axis_options)
        axis_options = dict(self._restored_view.get('axis_options', {}))
        axis_options.update(self._axis_options)

        if functions is not None:
            self._artists = scatter_functions_3d(functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=False, fig=self._fig, ax=self._ax)
        else:
            self._artists = []
            for obj, options in self._entries():
                if not options['visible']:
                    continue
                if not isinstance(obj, Curve3D):
                    raise TypeError(f'{type(obj).__name__} is not a scatterable 3D curve')
                self._artists.append(scatter_curve_3d(obj, show=False, fig=self._fig, ax=self._ax, label=options['label'], **options['style']))
        return self._finish(xlabel, ylabel, zlabel, title, legend, grid, None, None, None, show, return_artists, resolved_theme, axis_options)
