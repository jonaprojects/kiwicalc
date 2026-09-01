from __future__ import annotations
import math
from math import ceil, sqrt
import cmath
import warnings
from itertools import combinations, cycle
from typing import Union, Tuple, List, Optional, Any, Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kiwicalc.core.interfaces import IExpression, IPlottable
from kiwicalc.core.utils import (
    decimal_range, is_lambda, create_grid, draw_axis,
    format_matplot
)
from kiwicalc.core.ranges import values_in_range
from kiwicalc.geometry.points import Circle, process_to_points
from kiwicalc.functions.function import Function


def _figure_and_axes(fig=None, ax=None, projection=None):
    if ax is not None:
        return ax.figure, ax
    if fig is None:
        fig = plt.figure() if projection == '3d' else plt.subplots(figsize=(10, 8))[0]
    if projection == '3d':
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
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
        ax.plot([], [], label=label, **proxy_style)
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

def scatter_dots(x_values, y_values, title: str='', ymin: float=-10, ymax: float=10, color=None, show_axis=True, show=True, fig=None, ax=None):
    if (length := len(x_values)) != (y_length := len(y_values)):
        raise ValueError(f'You must enter an equal number of x and y values. Got {length} x values and {y_length} y values.')
    if None in (fig, ax):
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    plt.title(title, fontsize=14)
    plt.ylim(ymin, ymax)
    plt.scatter(x=x_values, y=y_values, s=90, c=color)
    if show:
        plt.show()

def scatter_dots_3d(x_values, y_values, z_values, title: str='', xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', fig=None, ax=None, show=True, write_labels=True):
    if None in (fig, ax):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if title:
        plt.title(title)
    ax.scatter(x_values, y_values, z_values)
    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    if show:
        plt.show()

def scatter_function(func: Union[Callable, str], start: float=-10, stop: float=10, step: float=0.5, ymin: float=-10, ymax: float=10, title='', color=None, show_axis=True, show=True, fig=None, ax=None, values=None):
    if isinstance(func, str):
        func = Function(func)
    if values is not None:
        results = [func(value) for value in values]
    else:
        values, results = values_in_range(func, start, stop, step)
    scatter_dots(values, results, title=title, ymin=ymin, ymax=ymax, color=color, show_axis=show_axis, show=show, fig=fig, ax=ax)

def scatter_function_3d(func: 'Union[Callable, str, IExpression]', start: float=-3, stop: float=3, step: float=0.3, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None, write_labels=True, meshgrid=None, title=''):
    if isinstance(func, str):
        func = Function(func)
    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            zs = np.append(zs, func(x_value, y_value))
        except:
            zs = np.append(zs, np.nan)
    Z = zs.reshape(X.shape)
    scatter_dots_3d(X, Y, Z, fig=fig, ax=ax, title=title, show=show, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, write_labels=write_labels)

def scatter_functions_3d(functions: 'Iterable[Union[Callable, str, IExpression]]', start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None):
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

def plot_function(func: Union[Callable, str], start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title=None, show_axis=True, show=True, fig=None, ax=None, formatText=False, values=None, label=None, **style):
    if None in (fig, ax):
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values, results = process_to_points(func, start, stop, step, ymin, ymax, values)
    if title is not None:
        if formatText:
            ax.set_title(f'${format_matplot(title)}$', fontsize=14)
        else:
            ax.set_title(f'{title}', fontsize=14)
    ax.set_ylim(ymin, ymax)
    line, = ax.plot(values, results, label=label, **style)
    if show:
        plt.show()
    return line

def plot_function_3d(given_function: 'Union[Callable, str, IExpression]', start: float=-3, stop: float=3, step: float=0.3, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None, write_labels=True, meshgrid=None, label=None, wireframe=False, **style):
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
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            result = given_function(x_value, y_value)
            if result is None:
                result = np.nan
            zs = np.append(zs, result)
        except ValueError:
            zs = np.append(zs, np.nan)
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

def plot_functions(functions, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title: str=None, formatText: bool=False, show_axis: bool=True, show: bool=True, with_legend=True):
    fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values = np.arange(start, stop, step)
    plt.ylim(ymin, ymax)
    if title is not None:
        if formatText:
            plt.title(f'${format_matplot(title)}$', fontsize=14)
        else:
            plt.title(title, fontsize=14)
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
        plt.plot(values, [given_function(value) for value in values], label=label)
    if with_legend:
        plt.legend()
    if show:
        plt.show()

def scatter_functions(functions, start: float=-10, stop: float=10, step: float=0.5, ymin: float=-10, ymax: float=10, title: str=None, show_axis: bool=True, show: bool=True):
    fig, ax = create_grid()
    cycol = cycle('bgrcmykw')
    values = np.arange(start, stop, step)
    for index, current_function in enumerate(functions):
        scatter_function(func=current_function, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title, color=next(cycol), show_axis=True, show=False, fig=fig, ax=ax, values=values)
    plt.show()

def plot_vector_2d(x_start: float, y_start: float, x_distance: float, y_distance: float, show=True, fig=None, ax=None):
    if None in (fig, ax):
        fig, ax = plt.subplots(figsize=(10, 8))
    artist = ax.arrow(x_start, y_start, x_distance, y_distance, head_width=0.1, width=0.01)
    if show:
        plt.show()
    return artist

def plot_vector_3d(starts: Tuple[float, float, float], distances: Tuple[float, float, float], arrow_length_ratio=0.08, show=True, fig=None, ax=None):
    """plot a 3d vector"""
    u, v, w = distances
    start_x, start_y, start_z = starts
    if (fig, ax) == (None, None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([start_x, start_x + u])
        ax.set_ylim([start_y, start_y + v])
        ax.set_zlim([start_z, start_z + w])
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    artist = ax.quiver(start_x, start_y, start_z, u, v, w, arrow_length_ratio=arrow_length_ratio)
    if show:
        plt.show()
    return artist

def plot_complex(*numbers: complex, title: str='', show=True):
    """
    plot complex numbers on the complex plane

    :param numbers: The complex numbers to be plotted
    :param show: If set to false, the plotted
    :return: fig, ax
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title, va='bottom')
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    plt.title(title)
    max_radius = abs(numbers[0])
    for c in numbers:
        radius = abs(c)
        if radius > max_radius:
            max_radius = radius
        ax.scatter(cmath.phase(c), radius)
    ax.set_rticks(np.linspace(0, int(max_radius) * 2, num=5))
    ax.set_rmax(max_radius * 1.25)
    if show:
        plt.show()
    return (fig, ax)

def generate_subplot_shape(num_of_functions: int):
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

def plot_multiple(funcs, shape: Tuple[int, int]=None, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, title=None, show_axis=True, show=True, values=None):
    num_of_functions = len(funcs)
    if shape is None:
        shape = generate_subplot_shape(num_of_functions)
    fig, ax = plt.subplots(shape[0], shape[1])
    fig.tight_layout()
    func_index = 0
    for i in range(shape[0]):
        if func_index >= num_of_functions:
            break
        for j in range(shape[1]):
            if func_index >= num_of_functions:
                break
            values, results = process_to_points(funcs[func_index], start, stop, step, ymin, ymax, values)
            current_ax = ax[i, j] if shape[0] > 1 else ax[j]
            current_ax.plot(values, results, label=funcs[func_index])
            current_ax.set_title(funcs[func_index])
            if show_axis:
                draw_axis(current_ax)
            func_index += 1
    if title is not None:
        plt.title(title)
    if show:
        try:
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
        except:
            warnings.warn("Couldn't plot in full screen!")
        plt.show()

class Graph:

    def __init__(self, objs, fig, ax):
        self._items = [obj for obj in objs]
        self._item_options = [dict(label=None, visible=True, style={}) for _ in self._items]
        self._fig, self._ax = (fig, ax)
        self._artists = []
        self._decorations = []
        self._restored_view = {}
        self._has_plotted = False

    @property
    def items(self):
        return self._items

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    @property
    def artists(self):
        return list(self._artists)

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

    def clear(self):
        self._items.clear()
        self._item_options.clear()
        self._artists.clear()
        self._decorations.clear()
        return self

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

    def plot(self):
        raise NotImplementedError

    def scatter(self):
        raise NotImplementedError

class Graph2D(Graph):

    def __init__(self, objs: Iterable[IPlottable]=tuple()):
        fig, ax = create_grid()
        super(Graph2D, self).__init__(objs, fig, ax)

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

    def fill_between(self, first, second=0, values=None, label=None, **style):
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

    def plot(self, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, text=None, show_axis=True, show=True, formatText=False, values=None, legend=None, grid=None, xlim=None, ylim=None, equal_aspect=None, return_artists=False):
        from kiwicalc.geometry.curves import Curve2D, ImplicitCurve2D
        from kiwicalc.geometry.points import Line2D, Point2D
        from kiwicalc.geometry.vectors import Vector2D

        if values is None:
            values = list(decimal_range(start=start, stop=stop, step=step))
        if show_axis:
            draw_axis(self._ax)
        if text is None and self._restored_view.get('title'):
            graph_title = self._restored_view['title']
        elif text is None:
            if len(self._items) >= 3:
                graph_title = ', '.join([obj.__str__() for obj in self._items[:3]]) + '...'
            else:
                graph_title = ', '.join((obj.__str__() for obj in self._items))
        else:
            graph_title = text
        if graph_title:
            self._ax.set_title(f'${format_matplot(graph_title)}$' if formatText else graph_title)
        self._artists = []
        for obj, options in self._entries():
            if not options['visible']:
                continue
            label, style = options['label'], options['style']
            if isinstance(obj, ImplicitCurve2D):
                artist = plot_implicit_curve_2d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Curve2D):
                artist = plot_curve_2d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Function):
                artist = plot_function(obj.lambda_expression, values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, IExpression):
                artist = plot_function(obj.to_lambda(), values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Line2D):
                if obj.to_lambda() is None:
                    artist = self._ax.axvline(obj._point1.x, label=label, **style)
                else:
                    artist = plot_function(obj.to_lambda(), values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, **style)
            elif isinstance(obj, Circle):
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
                artist = self._ax.scatter([obj.x], [obj.y], label=label, **style)
            elif isinstance(obj, Vector2D):
                artist = self._ax.arrow(obj._start_coordinate[0], obj._start_coordinate[1], obj.x_step, obj.y_step, label=label, **style)
            elif callable(obj) or isinstance(obj, str):
                artist = plot_function(obj, values=values, ymin=ymin, ymax=ymax, show_axis=False, show=False, fig=self._fig, ax=self._ax, label=label, **style)
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
            else:
                fill_values = decoration['values']
                if fill_values is None:
                    fill_values = values
                x_values = np.asarray(fill_values, dtype=float)
                first_y = self._fill_values(decoration['first'], x_values)
                second_y = self._fill_values(decoration['second'], x_values)
                artist = self._ax.fill_between(x_values, first_y, second_y, label=decoration['label'], **decoration['style'])
            self._artists.append(artist)
        xlim = xlim if xlim is not None else self._restored_view.get('xlim', (start, stop))
        ylim = ylim if ylim is not None else self._restored_view.get('ylim', (ymin, ymax))
        grid = self._restored_view.get('grid', False) if grid is None else grid
        legend = self._restored_view.get('legend', False) if legend is None else legend
        if equal_aspect is None:
            equal_aspect = self._restored_view.get('equal_aspect')
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.grid(bool(grid))
        if equal_aspect is True:
            self._ax.set_aspect('equal', adjustable='box')
        if legend:
            self._ax.legend()
        if show:
            plt.show()
        self._has_plotted = True
        if return_artists:
            return self.artists

class Graph3D(Graph):

    def __init__(self, objs=tuple()):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        super(Graph3D, self).__init__(objs, fig, ax)

    def _finish(self, xlabel, ylabel, zlabel, title, legend, grid, xlim, ylim, zlim, show, return_artists):
        view = self._restored_view
        xlabel = view.get('xlabel', xlabel)
        ylabel = view.get('ylabel', ylabel)
        zlabel = view.get('zlabel', zlabel)
        title = view.get('title', title) if title is None else title
        legend = view.get('legend', False) if legend is None else legend
        grid = view.get('grid', False) if grid is None else grid
        xlim = view.get('xlim') if xlim is None else xlim
        ylim = view.get('ylim') if ylim is None else ylim
        zlim = view.get('zlim') if zlim is None else zlim
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.set_zlabel(zlabel)
        if title:
            self._ax.set_title(title)
        self._ax.grid(bool(grid))
        for setter, limits in ((self._ax.set_xlim, xlim), (self._ax.set_ylim, ylim), (self._ax.set_zlim, zlim)):
            if limits is not None:
                setter(*limits)
        if legend:
            self._ax.legend()
        if show:
            plt.show()
        self._has_plotted = True
        if return_artists:
            return self.artists

    def plot(self, functions=None, start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, title=None, legend=None, grid=None, xlim=None, ylim=None, zlim=None, return_artists=False):
        from kiwicalc.geometry.curves import Curve3D
        from kiwicalc.geometry.surfaces import Surface3D
        from kiwicalc.geometry.vectors import Vector3D

        if functions is not None:
            self._artists = plot_functions_3d(functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=False, fig=self._fig, ax=self._ax)
        else:
            self._artists = []
            for obj, options in self._entries():
                if not options['visible']:
                    continue
                label, style = options['label'], dict(options['style'])
                if isinstance(obj, Curve3D):
                    artist = plot_curve_3d(obj, show=False, fig=self._fig, ax=self._ax, label=label, **style)
                elif isinstance(obj, Surface3D):
                    wireframe = style.pop('wireframe', False)
                    artist = plot_surface_3d(obj, show=False, fig=self._fig, ax=self._ax, label=label, wireframe=wireframe, **style)
                elif isinstance(obj, Vector3D):
                    before = len(self._ax.collections)
                    obj.plot(show=False, fig=self._fig, ax=self._ax)
                    artist = self._ax.collections[before:]
                elif hasattr(obj, 'to_lambda'):
                    artist = plot_function_3d(obj.to_lambda(), start=start, stop=stop, step=step, show=False, fig=self._fig, ax=self._ax, write_labels=False, label=label, **style)
                elif callable(obj) or isinstance(obj, str) or isinstance(obj, IExpression):
                    artist = plot_function_3d(obj, start=start, stop=stop, step=step, show=False, fig=self._fig, ax=self._ax, write_labels=False, label=label, **style)
                else:
                    raise TypeError(f'{type(obj).__name__} cannot be plotted on Graph3D')
                self._artists.append(artist)
        return self._finish(xlabel, ylabel, zlabel, title, legend, grid, xlim, ylim, zlim, show, return_artists)

    def scatter(self, functions=None, start: float=-5, stop: float=5, step: float=0.1, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, title=None, legend=None, grid=None, return_artists=False):
        from kiwicalc.geometry.curves import Curve3D

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
        return self._finish(xlabel, ylabel, zlabel, title, legend, grid, None, None, None, show, return_artists)
