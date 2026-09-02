"""Scientific 2D field renderers used by :class:`~kiwicalc.Graph2D`."""

from __future__ import annotations

import inspect
import math

import numpy as np

from kiwicalc.core.interfaces import IExpression
from kiwicalc.functions.function import Function


FIELD_KINDS = {"vector_field", "slope_field", "gradient_field", "streamlines", "contour_map"}


def _range(value, fallback, name):
    if value is None:
        value = fallback
    try:
        start, stop = map(float, value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (start, stop) pair")
    if not start < stop:
        raise ValueError(f"{name} start must be smaller than its stop")
    return start, stop


def _grid_density(value):
    if isinstance(value, int):
        counts = (value, value)
    else:
        try:
            counts = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            raise ValueError("density must be an integer or an (x, y) pair")
        if len(counts) != 2:
            raise ValueError("density must be an integer or an (x, y) pair")
    if min(counts) < 2:
        raise ValueError("density must provide at least 2 samples per axis")
    return counts


def make_grid(x_range, y_range, density):
    x_count, y_count = _grid_density(density)
    x = np.linspace(*x_range, x_count)
    y = np.linspace(*y_range, y_count)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def _function_and_names(source):
    if isinstance(source, str):
        source = Function(source)
    if isinstance(source, Function):
        return source.lambda_expression, list(source.variables)
    if isinstance(source, IExpression):
        source = source.to_lambda()
    if callable(source):
        try:
            names = list(inspect.signature(source).parameters)
        except (TypeError, ValueError):
            names = ["x", "y"]
        return source, names
    raise TypeError("Field formulas must be numbers, callables, expressions, or formula strings")


def evaluate_xy(source, X, Y):
    """Evaluate a scalar function over a mesh while respecting named variables."""
    if isinstance(source, (int, float)):
        return np.full_like(X, float(source), dtype=float)
    function, names = _function_and_names(source)
    coordinates = {"x": X, "y": Y}
    arguments = [coordinates.get(name, X if index == 0 else Y) for index, name in enumerate(names)]
    try:
        result = np.asarray(function(*arguments), dtype=float)
        if result.shape == ():
            return np.full_like(X, float(result), dtype=float)
        return np.broadcast_to(result, X.shape).astype(float, copy=False)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        values = np.empty_like(X, dtype=float)
        for index in np.ndindex(X.shape):
            scalar_coordinates = {"x": float(X[index]), "y": float(Y[index])}
            scalar_arguments = [scalar_coordinates.get(name, scalar_coordinates["x"] if position == 0 else scalar_coordinates["y"]) for position, name in enumerate(names)]
            try:
                value = float(function(*scalar_arguments))
                values[index] = value if math.isfinite(value) else np.nan
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                values[index] = np.nan
        return values


def _vectors(decoration, X, Y, x, y):
    kind = decoration["kind"]
    if kind == "slope_field":
        slope = evaluate_xy(decoration["source"], X, Y)
        U, V = np.ones_like(slope), slope
    elif kind == "gradient_field":
        Z = evaluate_xy(decoration["source"], X, Y)
        V, U = np.gradient(Z, y, x, edge_order=1)
    else:
        U = evaluate_xy(decoration["u"], X, Y)
        V = evaluate_xy(decoration["v"], X, Y)
    magnitude = np.hypot(U, V)
    if decoration.get("normalize", kind == "slope_field"):
        finite_magnitude = magnitude[np.isfinite(magnitude)]
        scale = float(np.max(finite_magnitude)) if finite_magnitude.size else 1.0
        active = magnitude > max(np.finfo(float).eps * 100, scale * 1e-12)
        U = np.divide(U, magnitude, out=np.zeros_like(U), where=active)
        V = np.divide(V, magnitude, out=np.zeros_like(V), where=active)
    return U, V, magnitude


def _colored_arguments(color, magnitude, style):
    if color == "magnitude":
        style.setdefault("cmap", "viridis")
        return (magnitude,)
    style.setdefault("color", color)
    return ()


def _colorbar(ax, artist, enabled, label, theme=None):
    if not enabled:
        return None
    colorbar = ax.figure.colorbar(artist, ax=ax)
    if label:
        colorbar.set_label(label)
    if theme is not None:
        colorbar.ax.set_facecolor(theme.axes_facecolor)
        colorbar.ax.tick_params(colors=theme.foreground, labelsize=theme.font_size)
        colorbar.ax.yaxis.label.set_color(theme.foreground)
        colorbar.ax.yaxis.label.set_fontsize(theme.label_size)
        colorbar.outline.set_edgecolor(theme.foreground)
    return colorbar


def render(ax, decoration, default_x_range, default_y_range, theme=None):
    """Render one field decoration, returning ``(artists, colorbar)``."""
    kind = decoration["kind"]
    x_range = _range(decoration.get("x_range"), default_x_range, "x_range")
    y_range = _range(decoration.get("y_range"), default_y_range, "y_range")
    density = decoration.get("density", 20)
    if kind in {"streamlines", "contour_map"}:
        density = decoration.get("samples", 60)
    x, y, X, Y = make_grid(x_range, y_range, density)
    style = dict(decoration.get("style", {}))

    if kind == "contour_map":
        Z = evaluate_xy(decoration["source"], X, Y)
        levels = decoration.get("levels", 10)
        filled = decoration.get("filled", False)
        factory = ax.contourf if filled else ax.contour
        contour = factory(X, Y, Z, levels=levels, **style)
        artists = [contour]
        if decoration.get("labels") and not filled:
            artists.extend(ax.clabel(contour, inline=True, fontsize=decoration.get("label_size", 8)))
        colorbar = _colorbar(ax, contour, decoration.get("colorbar", False), decoration.get("colorbar_label"), theme)
        return artists, colorbar

    U, V, magnitude = _vectors(decoration, X, Y, x, y)
    color = decoration.get("color", "magnitude")
    color_values = _colored_arguments(color, magnitude, style)
    if kind == "streamlines":
        if color_values:
            style["color"] = magnitude
        stream_density = float(decoration.get("stream_density", 1.0))
        if stream_density <= 0:
            raise ValueError("streamline density must be positive")
        style.setdefault("density", stream_density)
        stream = ax.streamplot(x, y, U, V, **style)
        artists = [stream.lines, stream.arrows]
        colorbar = _colorbar(ax, stream.lines, decoration.get("colorbar", False), decoration.get("colorbar_label", "Magnitude"), theme)
        return artists, colorbar

    style.setdefault("angles", "xy")
    if kind == "slope_field":
        style.setdefault("pivot", "mid")
    quiver = ax.quiver(X, Y, U, V, *color_values, **style)
    colorbar = _colorbar(ax, quiver, decoration.get("colorbar", False), decoration.get("colorbar_label", "Magnitude"), theme)
    return [quiver], colorbar
