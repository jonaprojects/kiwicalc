"""Animation and live parameter controls for KiwiCalc graphs."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np

from kiwicalc.core.interfaces import IExpression
from kiwicalc.functions.function import Function


def _callable_names(function):
    try:
        return list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return []


def _evaluate_callable(function, arguments, scalar_arguments, shape):
    try:
        result = np.asarray(function(*arguments), dtype=float)
        if result.shape == ():
            return np.full(shape, float(result), dtype=float)
        return np.broadcast_to(result, shape).astype(float, copy=False)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        values = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            try:
                value = float(function(*(argument(index) for argument in scalar_arguments)))
                values[index] = value if math.isfinite(value) else np.nan
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                values[index] = np.nan
        return values


def _sample_function(source, parameter_value, x_values, parameter):
    if isinstance(source, str):
        source = Function(source)
    if isinstance(source, Function):
        function = source.lambda_expression
        names = list(source.variables)
    elif isinstance(source, IExpression):
        function = source.to_lambda()
        names = _callable_names(function)
    elif callable(source):
        function = source
        names = _callable_names(function)
        if len(names) <= 1:
            produced = function(parameter_value)
            return sample_frame(produced, parameter_value, x_values, parameter)
    else:
        raise TypeError("Animated formulas must be callables, expressions, or formula strings")

    coordinate = {"x": x_values, parameter: parameter_value}
    arguments = [coordinate.get(name, x_values if index == 0 else parameter_value) for index, name in enumerate(names)]
    scalar_arguments = [
        (lambda index, name=name, position=index: float(x_values[index]) if name == "x" or (name not in coordinate and position == 0) else float(parameter_value))
        for index, name in enumerate(names)
    ]
    return x_values, _evaluate_callable(function, arguments, scalar_arguments, x_values.shape)


def sample_frame(source, parameter_value, x_values, parameter="a"):
    """Sample one frame from a parameterized function or curve factory."""
    from kiwicalc.geometry.curves import Curve2D

    if isinstance(source, Curve2D):
        x, y = source.sample()
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return _sample_function(source, parameter_value, np.asarray(x_values, dtype=float), str(parameter))


def _title(template, parameter, value):
    if template is None:
        return None
    if callable(template):
        return str(template(value))
    try:
        return str(template).format(value=value, **{parameter: value})
    except (KeyError, ValueError):
        return str(template)


def _automatic_ylim(y_values):
    finite = np.asarray(y_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return (-1.0, 1.0)
    low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        padding = max(1.0, abs(low) * 0.1)
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def _motion_envelope(source, parameter_values, x_values, parameter):
    sampled = [sample_frame(source, value, x_values, parameter) for value in parameter_values]
    all_x = np.concatenate([np.asarray(x, dtype=float).ravel() for x, _ in sampled])
    all_y = np.concatenate([np.asarray(y, dtype=float).ravel() for _, y in sampled])
    finite_x = all_x[np.isfinite(all_x)]
    xlim = (float(np.min(finite_x)), float(np.max(finite_x))) if finite_x.size else (-1.0, 1.0)
    if xlim[0] == xlim[1]:
        xlim = (xlim[0] - 1, xlim[1] + 1)
    return xlim, _automatic_ylim(all_y)


def _prepare(graph, source, initial, parameter, x_values, label, title, line_style, plot_options):
    frame_x, frame_y = sample_frame(source, initial, x_values, parameter)
    options = dict(plot_options)
    options.pop("return_artists", None)
    options.pop("show", None)
    legend_requested = bool(options.pop("legend", False))
    options.setdefault("values", x_values)
    options.setdefault("xlim", (float(np.min(x_values)), float(np.max(x_values))))
    options.setdefault("ylim", _automatic_ylim(frame_y))
    first_title = _title(title, parameter, initial)
    if first_title is not None:
        options["title"] = first_title
    graph.plot(show=False, **options)
    style = dict(line_style or {})
    line, = graph.ax.plot(frame_x, frame_y, label=label, **style)
    graph._artists.append(line)
    if legend_requested:
        graph._legend_artist = graph.ax.legend()
    return line


class GraphAnimation:
    """Small convenience wrapper around Matplotlib's ``FuncAnimation``."""

    def __init__(self, graph, animation, frames, parameter, line):
        self.graph = graph
        self.animation = animation
        self.frames = tuple(frames)
        self.parameter = parameter
        self.line = line

    @property
    def fig(self):
        return self.graph.fig

    @property
    def ax(self):
        return self.graph.ax

    def pause(self):
        self.animation.pause()
        return self

    def resume(self):
        self.animation.resume()
        return self

    def to_html(self, *, mode="js", **kwargs):
        """Return embeddable HTML using JavaScript or HTML5 video."""
        if mode == "js":
            return self.animation.to_jshtml(**kwargs)
        if mode in {"video", "html5"}:
            return self.animation.to_html5_video(**kwargs)
        raise ValueError("mode must be 'js' or 'video'")

    def _repr_html_(self):
        """Render directly when this controller is the last value in a notebook cell."""
        return self.to_html()

    def save(self, path, *, fps=None, writer=None, **kwargs):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        options = dict(kwargs)
        if fps is not None:
            options["fps"] = fps
        if writer is not None:
            options["writer"] = writer
        self.animation.save(str(output), **options)
        return output


class GraphInteraction:
    """Controller returned by a slider-driven graph."""

    def __init__(self, graph, slider, line, callback_id, parameter):
        self.graph = graph
        self.slider = slider
        self.line = line
        self.parameter = parameter
        self._callback_id = callback_id

    @property
    def fig(self):
        return self.graph.fig

    @property
    def ax(self):
        return self.graph.ax

    @property
    def value(self):
        return self.slider.val

    def set_value(self, value):
        value = float(value)
        if not self.slider.valmin <= value <= self.slider.valmax:
            raise ValueError("value must be inside parameter_range")
        self.slider.set_val(value)
        return self

    def disconnect(self):
        if self._callback_id is not None:
            self.slider.disconnect(self._callback_id)
            self._callback_id = None
        return self


def animate_parameter(
    graph, source, frames: Iterable[float], *, parameter="a", start=-10,
    stop=10, samples=400, values=None, interval=50, repeat=True, blit=False,
    label=None, title=None, show=True, line_style=None, **plot_options,
):
    if not isinstance(parameter, str) or not parameter:
        raise ValueError("parameter must be a non-empty string")
    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a positive finite number")
    frame_values = tuple(frames)
    if not frame_values:
        raise ValueError("frames must contain at least one parameter value")
    if values is None:
        if int(samples) < 2:
            raise ValueError("samples must be at least 2")
        x_values = np.linspace(float(start), float(stop), int(samples))
    else:
        x_values = np.asarray(tuple(values), dtype=float)
        if len(x_values) < 2:
            raise ValueError("values must contain at least two x coordinates")
    plot_options = dict(plot_options)
    envelope_x, envelope_y = _motion_envelope(source, frame_values, x_values, parameter)
    plot_options.setdefault("xlim", envelope_x)
    plot_options.setdefault("ylim", envelope_y)
    line = _prepare(graph, source, frame_values[0], parameter, x_values, label, title, line_style, plot_options)

    def update(value):
        frame_x, frame_y = sample_frame(source, value, x_values, parameter)
        line.set_data(frame_x, frame_y)
        next_title = _title(title, parameter, value)
        if next_title is not None:
            graph.ax.set_title(next_title)
        return (line,)

    animation = FuncAnimation(
        graph.fig, update, frames=frame_values, interval=interval,
        repeat=bool(repeat), blit=bool(blit),
    )
    graph.fig.canvas.draw_idle()
    controller = GraphAnimation(graph, animation, frame_values, parameter, line)
    graph._animations.append(controller)
    if show:
        plt.show()
    return controller


def interactive_parameter(
    graph, source, parameter_range, *, parameter="a", initial=None, step=None,
    start=-10, stop=10, samples=400, values=None, label=None, title=None,
    show=True, line_style=None, **plot_options,
):
    if not isinstance(parameter, str) or not parameter:
        raise ValueError("parameter must be a non-empty string")
    try:
        minimum, maximum = map(float, parameter_range)
    except (TypeError, ValueError):
        raise ValueError("parameter_range must be a (minimum, maximum) pair")
    if not minimum < maximum:
        raise ValueError("parameter minimum must be smaller than its maximum")
    initial = (minimum + maximum) / 2 if initial is None else float(initial)
    if not minimum <= initial <= maximum:
        raise ValueError("initial must be inside parameter_range")
    if step is not None and (not math.isfinite(float(step)) or float(step) <= 0):
        raise ValueError("step must be a positive finite number")
    if values is None:
        if int(samples) < 2:
            raise ValueError("samples must be at least 2")
        x_values = np.linspace(float(start), float(stop), int(samples))
    else:
        x_values = np.asarray(tuple(values), dtype=float)
        if len(x_values) < 2:
            raise ValueError("values must contain at least two x coordinates")

    plot_options = dict(plot_options)
    envelope_parameters = np.linspace(minimum, maximum, 17)
    envelope_x, envelope_y = _motion_envelope(source, envelope_parameters, x_values, parameter)
    automatic_ylim = "ylim" not in plot_options
    plot_options.setdefault("xlim", envelope_x)
    plot_options.setdefault("ylim", envelope_y)
    line = _prepare(graph, source, initial, parameter, x_values, label, title, line_style, plot_options)
    graph.fig.subplots_adjust(bottom=0.18)
    slider_ax = graph.fig.add_axes((0.18, 0.055, 0.66, 0.035))
    slider = Slider(slider_ax, str(parameter), minimum, maximum, valinit=initial, valstep=step)
    if graph._theme is not None:
        slider_ax.set_facecolor(graph._theme.axes_facecolor)
        slider.label.set_color(graph._theme.foreground)
        slider.valtext.set_color(graph._theme.foreground)

    def update(value):
        frame_x, frame_y = sample_frame(source, value, x_values, parameter)
        line.set_data(frame_x, frame_y)
        if automatic_ylim:
            low, high = _automatic_ylim(frame_y)
            current_low, current_high = graph.ax.get_ylim()
            if low < current_low or high > current_high:
                graph.ax.set_ylim(min(low, current_low), max(high, current_high))
        next_title = _title(title, parameter, value)
        if next_title is not None:
            graph.ax.set_title(next_title)
        graph.fig.canvas.draw_idle()

    callback_id = slider.on_changed(update)
    controller = GraphInteraction(graph, slider, line, callback_id, parameter)
    graph._interactions.append(controller)
    if show:
        plt.show()
    return controller
