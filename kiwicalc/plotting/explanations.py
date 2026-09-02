"""Numerical analysis and rendering helpers for educational 2D plots."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Polygon, Rectangle

from kiwicalc.core.interfaces import IExpression
from kiwicalc.functions.function import Function


_LABEL_OFFSETS = ((7, 8), (7, -17), (-42, 8), (-42, -17), (10, 18), (-48, 18))


def as_callable(source):
    """Return a scalar-friendly callable for a supported function-like value."""
    if isinstance(source, str):
        source = Function(source)
    if isinstance(source, Function):
        return source.lambda_expression
    if isinstance(source, IExpression):
        return source.to_lambda()
    if callable(source):
        return source
    raise TypeError("Expected a function, expression, formula string, or callable")


def evaluate(source, x_values):
    function = as_callable(source)
    results = []
    for x in np.asarray(x_values, dtype=float):
        try:
            value = float(function(float(x)))
            results.append(value if math.isfinite(value) else np.nan)
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            results.append(np.nan)
    return np.asarray(results, dtype=float)


def domain_values(domain, default_values, samples=1201):
    if domain is None:
        values = np.asarray(default_values, dtype=float)
        if len(values) >= 3:
            return values
        domain = (float(values[0]), float(values[-1]))
    try:
        start, stop = map(float, domain)
    except (TypeError, ValueError):
        raise ValueError("domain must be a (start, stop) pair")
    if not start < stop:
        raise ValueError("domain start must be smaller than domain stop")
    if int(samples) < 5:
        raise ValueError("samples must be at least 5")
    return np.linspace(start, stop, int(samples))


def _dedupe(values, tolerance):
    result = []
    for value in sorted(float(v) for v in values if math.isfinite(float(v))):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _bisect(source, left, right, iterations=50):
    function = as_callable(source)
    try:
        f_left = float(function(left))
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return (left + right) / 2
    for _ in range(iterations):
        middle = (left + right) / 2
        try:
            f_middle = float(function(middle))
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return None
        if abs(f_middle) < 1e-12:
            return middle
        if np.signbit(f_left) != np.signbit(f_middle):
            right = middle
        else:
            left, f_left = middle, f_middle
    candidate = (left + right) / 2
    try:
        return candidate if abs(float(function(candidate))) < 1e-8 else None
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None


def roots(source, x_values, y_values=None):
    x = np.asarray(x_values, dtype=float)
    y = evaluate(source, x) if y_values is None else np.asarray(y_values, dtype=float)
    finite = np.isfinite(y)
    scale = max(1.0, float(np.nanmax(np.abs(y[finite]))) if finite.any() else 1.0)
    near = max(1e-8, scale * 1e-5)
    candidates = []
    for index in range(len(x) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        if np.signbit(y[index]) != np.signbit(y[index + 1]):
            candidate = _bisect(source, x[index], x[index + 1])
            if candidate is not None:
                candidates.append(candidate)
    for index in range(1, len(x) - 1):
        if finite[index] and abs(y[index]) <= near and abs(y[index]) <= abs(y[index - 1]) and abs(y[index]) <= abs(y[index + 1]):
            candidates.append(x[index])
    tolerance = max(np.ptp(x) / max(len(x) - 1, 1) * 2.5, 1e-7)
    return [0.0 if abs(value) <= tolerance else value for value in _dedupe(candidates, tolerance)]


def derivatives(source, x_values):
    x = np.asarray(x_values, dtype=float)
    y = evaluate(source, x)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return y, np.full_like(y, np.nan), np.full_like(y, np.nan)
    filled = np.interp(x, x[finite], y[finite])
    first = np.gradient(filled, x, edge_order=2)
    second = np.gradient(first, x, edge_order=2)
    first[~finite] = np.nan
    second[~finite] = np.nan
    return y, first, second


def extrema(source, x_values):
    x = np.asarray(x_values, dtype=float)
    y, first, second = derivatives(source, x)
    points = []
    stationary = roots(lambda value: float(np.interp(value, x, first)), x, first)
    for value in stationary:
        index = int(np.argmin(np.abs(x - value)))
        if not np.isfinite(y[index]):
            continue
        left, right = max(0, index - 2), min(len(x) - 1, index + 2)
        if first[left] < 0 < first[right]:
            kind = "minimum"
        elif first[left] > 0 > first[right]:
            kind = "maximum"
        else:
            continue
        points.append((value, float(np.interp(value, x, y)), kind))
    return points


def inflections(source, x_values):
    x = np.asarray(x_values, dtype=float)
    y, _, second = derivatives(source, x)
    candidates = roots(lambda value: float(np.interp(value, x, second)), x, second)
    points = []
    for value in candidates:
        index = int(np.argmin(np.abs(x - value)))
        left, right = max(0, index - 2), min(len(x) - 1, index + 2)
        if np.isfinite(y[index]) and np.isfinite(second[left]) and np.isfinite(second[right]) and second[left] * second[right] < 0:
            points.append((value, float(np.interp(value, x, y))))
    return points


def _label(ax, text, point, index, style=None):
    if not text:
        return None
    label_style = dict(fontsize=9)
    label_style.update(style or {})
    return ax.annotate(text, xy=point, xytext=_LABEL_OFFSETS[index % len(_LABEL_OFFSETS)], textcoords="offset points", **label_style)


def _number(value):
    value = 0.0 if abs(float(value)) < 5e-9 else float(value)
    return f"{value:.3g}"


def _point_layers(ax, points, label, style, default_color="crimson"):
    if not points:
        return []
    point_style = dict(color=default_color, s=48, zorder=5)
    point_style.update(style)
    annotation_style = {}
    for key in ("fontsize", "fontweight", "ha", "va"):
        if key in point_style:
            annotation_style[key] = point_style.pop(key)
    artist = ax.scatter([p[0] for p in points], [p[1] for p in points], **point_style)
    artists = [artist]
    for index, point in enumerate(points):
        if label:
            default = point[2] if len(point) > 2 else f"({_number(point[0])}, {_number(point[1])})"
            annotation = _label(ax, label if isinstance(label, str) else default, point[:2], index, annotation_style)
            if annotation is not None:
                artists.append(annotation)
    return artists


def _line_at(ax, source, at, default_values, *, normal=False, label=None, span=None, style=None):
    from kiwicalc.geometry.curves import Curve2D

    style = dict(style or {})
    style.setdefault("linestyle", ":" if normal else "--")
    if isinstance(source, Curve2D):
        point = source.point_at(float(at)).coordinates
        direction = source.normal_at(float(at)) if normal else source.tangent_at(float(at))
        half = float(span or max(np.ptp(default_values) / 6, 1.0))
        norm = math.hypot(*direction)
        direction = (direction[0] / norm, direction[1] / norm)
        xs = [point[0] - half * direction[0], point[0] + half * direction[0]]
        ys = [point[1] - half * direction[1], point[1] + half * direction[1]]
    else:
        x0 = float(at)
        function = as_callable(source)
        h = max(abs(x0) * 1e-5, 1e-5)
        y0 = float(function(x0))
        slope = (float(function(x0 + h)) - float(function(x0 - h))) / (2 * h)
        if normal:
            slope = -1 / slope if abs(slope) > 1e-12 else math.inf
        half = float(span or max(np.ptp(default_values) / 6, 1.0))
        if math.isinf(slope):
            return [ax.axvline(x0, label=label, **style)]
        xs = [x0 - half, x0 + half]
        ys = [y0 - slope * half, y0 + slope * half]
    line, = ax.plot(xs, ys, label=label, **style)
    return [line]


def _as_values(option, automatic):
    if option is False:
        return []
    if option is True or option is None or (isinstance(option, str) and option == "auto"):
        return automatic
    if isinstance(option, (int, float)):
        return [float(option)]
    return [float(value) for value in option]


def _automatic_asymptotes(x, y):
    finite = np.isfinite(y)
    vertical = []
    for index in range(1, len(x) - 1):
        if not finite[index] and finite[index - 1] and finite[index + 1]:
            vertical.append(x[index])
    finite_y = np.abs(y[finite])
    if finite_y.size:
        threshold = max(50.0, float(np.nanmedian(finite_y)) * 50)
        for index in range(len(x) - 1):
            if finite[index] and finite[index + 1] and abs(y[index]) > threshold and abs(y[index + 1]) > threshold and np.signbit(y[index]) != np.signbit(y[index + 1]):
                vertical.append((x[index] + x[index + 1]) / 2)
    horizontal = []
    window = max(4, len(x) // 30)
    for x_segment, y_segment in ((x[:window], y[:window]), (x[-window:], y[-window:])):
        mask = np.isfinite(y_segment) & (np.abs(x_segment) > 1e-12)
        x_segment, y_segment = x_segment[mask], y_segment[mask]
        if len(y_segment) >= 3:
            inverse_x = 1 / x_segment
            slope, limit = np.polyfit(inverse_x, y_segment, 1)
            fitted = slope * inverse_x + limit
            residual = float(np.max(np.abs(y_segment - fitted)))
            tolerance = max(0.02, float(np.ptp(y_segment)) * 0.15)
            if residual <= tolerance and abs(slope * inverse_x[-1]) <= max(1.0, abs(limit) + 1.0):
                horizontal.append(float(limit))
    tolerance = max(np.ptp(x) / max(len(x), 1) * 3, 1e-5)
    return _dedupe(vertical, tolerance), _dedupe(horizontal, 0.05)


def render(ax, decoration, default_values):
    """Render one explanation decoration and return all created artists."""
    kind = decoration["kind"]
    source = decoration.get("source")
    style = dict(decoration.get("style", {}))
    x = domain_values(decoration.get("domain"), default_values, decoration.get("samples", 1201))

    if kind == "roots":
        return _point_layers(ax, [(value, 0.0) for value in roots(source, x)], decoration.get("label", True), style)
    if kind == "intersections":
        first_y = evaluate(decoration["first"], x)
        second_y = evaluate(decoration["second"], x)
        difference = first_y - second_y
        root_values = roots(lambda value: float(np.interp(value, x, difference)), x, difference)
        points = [(value, float(np.interp(value, x, first_y))) for value in root_values]
        return _point_layers(ax, points, decoration.get("label", True), style, "darkviolet")
    if kind == "extrema":
        return _point_layers(ax, extrema(source, x), decoration.get("label", True), style)
    if kind == "inflections":
        return _point_layers(ax, [(a, b, "inflection") for a, b in inflections(source, x)], decoration.get("label", True), style, "darkorange")
    if kind in ("tangent", "normal"):
        return _line_at(ax, source, decoration["at"], x, normal=kind == "normal", label=decoration.get("label"), span=decoration.get("span"), style=style)
    if kind == "secant":
        left, right = map(float, decoration["between"])
        line, = ax.plot([left, right], evaluate(source, [left, right]), label=decoration.get("label"), **style)
        return [line]
    if kind == "asymptotes":
        auto_vertical, auto_horizontal = _automatic_asymptotes(x, evaluate(source, x))
        style.setdefault("linestyle", "--")
        style.setdefault("color", "gray")
        artists = [ax.axvline(value, **style) for value in _as_values(decoration.get("vertical", "auto"), auto_vertical)]
        artists.extend(ax.axhline(value, **style) for value in _as_values(decoration.get("horizontal", "auto"), auto_horizontal))
        return artists
    if kind == "monotonicity":
        _, first, _ = derivatives(source, x)
        signs = np.where(first >= 0, 1, -1)
        artists, start = [], 0
        colors = decoration.get("colors", ("#54a24b", "#e45756"))
        alpha = style.pop("alpha", 0.12)
        for index in range(1, len(x) + 1):
            if index == len(x) or signs[index] != signs[start]:
                color = colors[0] if signs[start] > 0 else colors[1]
                artists.append(ax.axvspan(x[start], x[index - 1], color=color, alpha=alpha, **style))
                start = index
        return artists
    if kind == "solution":
        first = evaluate(source, x)
        other = decoration.get("other", 0)
        second = np.full_like(x, float(other)) if isinstance(other, (int, float)) else evaluate(other, x)
        relation = decoration.get("relation", ">=")
        operators = {">": first > second, ">=": first >= second, "<": first < second, "<=": first <= second, "==": np.isclose(first, second), "!=": ~np.isclose(first, second)}
        if relation not in operators:
            raise ValueError("relation must be one of >, >=, <, <=, ==, or !=")
        style.setdefault("alpha", 0.25)
        return [ax.fill_between(x, first, second, where=operators[relation], interpolate=True, label=decoration.get("label"), **style)]
    if kind == "riemann":
        left, right = map(float, decoration["interval"])
        count = int(decoration.get("rectangles", 8))
        method = decoration.get("method", "midpoint")
        if count < 1:
            raise ValueError("rectangles must be at least 1")
        if method not in {"left", "right", "midpoint", "trapezoid"}:
            raise ValueError("method must be left, right, midpoint, or trapezoid")
        edges = np.linspace(left, right, count + 1)
        style.setdefault("alpha", 0.28)
        style.setdefault("edgecolor", "black")
        style.setdefault("facecolor", "cornflowerblue")
        artists = []
        for index in range(count):
            a, b = edges[index], edges[index + 1]
            if method == "trapezoid":
                ya, yb = evaluate(source, [a, b])
                patch = Polygon([(a, 0), (a, ya), (b, yb), (b, 0)], closed=True, **style)
            else:
                sample = a if method == "left" else b if method == "right" else (a + b) / 2
                height = float(evaluate(source, [sample])[0])
                patch = Rectangle((a, min(0, height)), b - a, abs(height), **style)
            ax.add_patch(patch)
            artists.append(patch)
        return artists
    if kind == "slope_triangle":
        at, run = float(decoration["at"]), float(decoration.get("run", 1))
        if run == 0:
            raise ValueError("run must not be zero")
        y0 = float(evaluate(source, [at])[0])
        h = max(abs(at) * 1e-5, 1e-5)
        slope = float(evaluate(source, [at + h])[0] - evaluate(source, [at - h])[0]) / (2 * h)
        rise = slope * run
        style.setdefault("color", "darkorange")
        line1, = ax.plot([at, at + run], [y0, y0], **style)
        line2, = ax.plot([at + run, at + run], [y0, y0 + rise], **style)
        artists = [line1, line2]
        if decoration.get("label", True):
            artists.extend(filter(None, [_label(ax, f"run = {_number(run)}", (at + run / 2, y0), 0), _label(ax, f"rise = {_number(rise)}", (at + run, y0 + rise / 2), 1)]))
        return artists
    if kind == "derivative":
        _, first, _ = derivatives(source, x)
        line, = ax.plot(x, first, label=decoration.get("label", "derivative"), **style)
        return [line]
    if kind == "integral":
        y = evaluate(source, x)
        finite = np.isfinite(y)
        y = np.where(finite, y, 0.0)
        cumulative = np.zeros_like(y)
        cumulative[1:] = np.cumsum((y[:-1] + y[1:]) * np.diff(x) / 2)
        cumulative += float(decoration.get("constant", 0))
        cumulative[~finite] = np.nan
        line, = ax.plot(x, cumulative, label=decoration.get("label", "integral"), **style)
        return [line]
    raise ValueError(f"Unknown explanation kind: {kind}")


EXPLANATION_KINDS = {"roots", "intersections", "extrema", "inflections", "tangent", "normal", "secant", "asymptotes", "monotonicity", "solution", "riemann", "slope_triangle", "derivative", "integral"}
