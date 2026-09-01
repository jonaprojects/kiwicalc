from __future__ import annotations

"""Friendly, sampleable two- and three-dimensional curves."""

from abc import abstractmethod
from math import comb, cos, cosh, exp, pi, sin, sinh
from typing import Callable, Iterable, Sequence, Tuple, Union

import numpy as np

from kiwicalc.core.interfaces import IExpression, IPlottable, IPlottable3D
from kiwicalc.core.utils import to_lambda


ExpressionLike = Union[str, Callable, IExpression, int, float]


def _expression_body(value: str, implicit: bool = False) -> str:
    value = value.strip()
    if "=" not in value:
        return value
    left, right = (part.strip() for part in value.split("=", 1))
    if "(" in left and ")" in left:
        return right
    if implicit:
        return f"({left})-({right})"
    return value


def _as_callable(value: ExpressionLike, variables: Tuple[str, ...], implicit: bool = False) -> Callable:
    if isinstance(value, IExpression):
        return value.to_lambda(variables=variables)
    if isinstance(value, str):
        return to_lambda(_expression_body(value, implicit=implicit), variables)
    if callable(value):
        return value
    if isinstance(value, (int, float)):
        return lambda *args: value
    raise TypeError("Curve coordinates must be numbers, callables, strings, or KiwiCalc expressions")


def _evaluate_1d(func: Callable, values: np.ndarray) -> np.ndarray:
    results = []
    with np.errstate(all="ignore"):
        for value in values:
            try:
                result = func(float(value))
                result = float(result)
                results.append(result if np.isfinite(result) else np.nan)
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                results.append(np.nan)
    return np.asarray(results, dtype=float)


def _validate_range(bounds: Sequence[float], name: str) -> Tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    start, stop = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError(f"{name} must be a finite increasing range")
    return start, stop


def _validate_samples(samples: int) -> int:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    return samples


def _points_array(points: Iterable[Sequence[float]], dimensions: int, minimum: int = 2) -> np.ndarray:
    array = np.asarray(list(points), dtype=float)
    if array.ndim != 2 or array.shape[1] != dimensions or len(array) < minimum:
        raise ValueError(f"control_points must contain at least {minimum} points in {dimensions}D")
    if not np.all(np.isfinite(array)):
        raise ValueError("control_points must contain finite numbers")
    return array


def _rotate_2d(x, y, angle: float, center: Tuple[float, float]):
    cosine, sine = cos(angle), sin(angle)
    return (
        center[0] + cosine * x - sine * y,
        center[1] + sine * x + cosine * y,
    )


def _position(value: float) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("t must be between 0 and 1")
    return value


def _sample_point(axes, position: float):
    position = _position(position)
    length = len(axes[0])
    index = position * (length - 1)
    lower, upper = int(np.floor(index)), int(np.ceil(index))
    weight = index - lower
    values = np.asarray([(1 - weight) * axis[lower] + weight * axis[upper] for axis in axes], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("t falls in a gap or outside the curve's finite domain")
    return values


def _unit(vector, message="The curve has no defined direction at this point"):
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(message)
    return np.asarray(vector, dtype=float) / norm


def _derivatives(axes, position: float):
    count = len(axes[0])
    epsilon = max(1 / max(count - 1, 1), 1e-6)
    left, right = max(0, position - epsilon), min(1, position + epsilon)
    if left == position:
        second = min(1, position + 2 * epsilon)
        p0, p1, p2 = (_sample_point(axes, value) for value in (position, right, second))
    elif right == position:
        first = max(0, position - 2 * epsilon)
        p0, p1, p2 = (_sample_point(axes, value) for value in (first, left, position))
    else:
        p0, p1, p2 = (_sample_point(axes, value) for value in (left, position, right))
    first_derivative = (p2 - p0) / 2
    second_derivative = p2 - 2 * p1 + p0
    return first_derivative, second_derivative


def _arc_length(axes):
    points = np.column_stack(axes)
    finite = np.all(np.isfinite(points), axis=1)
    valid_pairs = finite[:-1] & finite[1:]
    differences = np.diff(points, axis=0)[valid_pairs]
    return float(np.linalg.norm(differences, axis=1).sum())


def _bounds(axes):
    result = []
    for axis in axes:
        finite = np.asarray(axis)[np.isfinite(axis)]
        if not len(finite):
            raise ValueError("Cannot calculate bounds for a curve with no finite points")
        result.append((float(finite.min()), float(finite.max())))
    return tuple(result)


def _adaptive_options(sampling, tolerance, max_depth):
    sampling = str(sampling).lower()
    if sampling not in ("fixed", "adaptive"):
        raise ValueError("sampling must be 'fixed' or 'adaptive'")
    tolerance = float(tolerance)
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a positive finite number")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    return sampling, tolerance, max_depth


def _evaluate_point(functions, value):
    result = []
    for func in functions:
        try:
            coordinate = float(func(float(value)))
            result.append(coordinate if np.isfinite(coordinate) else np.nan)
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            result.append(np.nan)
    return np.asarray(result, dtype=float)


def _adaptive_parametric(functions, bounds, tolerance, max_depth, seed_count=17):
    seeds = np.linspace(bounds[0], bounds[1], seed_count)
    output = []

    def refine(start, stop, first, last, depth):
        middle_t = (start + stop) / 2
        middle = _evaluate_point(functions, middle_t)
        all_finite = np.all(np.isfinite((first, middle, last)))
        error = np.linalg.norm(middle - (first + last) / 2) if all_finite else np.inf
        if depth >= max_depth or error <= tolerance:
            output.append((stop, last))
            return
        refine(start, middle_t, first, middle, depth + 1)
        refine(middle_t, stop, middle, last, depth + 1)

    first_point = _evaluate_point(functions, seeds[0])
    output.append((seeds[0], first_point))
    for start, stop in zip(seeds[:-1], seeds[1:]):
        first = output[-1][1]
        last = _evaluate_point(functions, stop)
        refine(start, stop, first, last, 0)
    points = np.asarray([point for _, point in output], dtype=float)
    return tuple(points[:, index] for index in range(points.shape[1]))


class Curve2D(IPlottable):
    """Base class for 2D curves that can be sampled and plotted."""

    dimensions = 2

    def __init__(self, samples: int = 500):
        self._samples = _validate_samples(samples)

    @property
    def samples(self) -> int:
        return self._samples

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def sample(self, samples: int = None):
        """Return ``(x_values, y_values)`` NumPy arrays."""

    def plot(self, show: bool = True, fig=None, ax=None, label: str = None, **style):
        from kiwicalc.plotting.plots import plot_curve_2d
        return plot_curve_2d(self, show=show, fig=fig, ax=ax, label=label, **style)

    def scatter(self, show: bool = True, fig=None, ax=None, label: str = None, **style):
        from kiwicalc.plotting.plots import scatter_curve_2d
        return scatter_curve_2d(self, show=show, fig=fig, ax=ax, label=label, **style)

    def transform(self, matrix):
        """Return a transformed copy using a 3x3 homogeneous matrix."""
        return TransformedCurve2D(self, matrix)

    def translate(self, x: float = 0, y: float = 0):
        matrix = np.asarray(((1, 0, x), (0, 1, y), (0, 0, 1)), dtype=float)
        return self.transform(matrix)

    def rotate(self, angle: float, center=(0, 0)):
        cx, cy = (float(value) for value in center)
        cosine, sine = cos(angle), sin(angle)
        matrix = np.asarray(
            ((cosine, -sine, cx - cosine * cx + sine * cy),
             (sine, cosine, cy - sine * cx - cosine * cy),
             (0, 0, 1)),
            dtype=float,
        )
        return self.transform(matrix)

    def scale(self, factor, y=None, center=(0, 0)):
        sx, sy = (float(factor), float(factor if y is None else y))
        cx, cy = (float(value) for value in center)
        matrix = np.asarray(((sx, 0, cx * (1 - sx)), (0, sy, cy * (1 - sy)), (0, 0, 1)), dtype=float)
        return self.transform(matrix)

    def reflect(self, axis="x"):
        normalized = str(axis).lower()
        matrices = {
            "x": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
            "y": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "origin": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            "y=x": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        }
        if normalized not in matrices:
            raise ValueError("axis must be 'x', 'y', 'origin', or 'y=x'")
        return self.transform(matrices[normalized])

    def point_at(self, t: float):
        """Return a point at normalized position ``t`` from 0 to 1."""
        from kiwicalc.geometry.points import Point2D
        return Point2D(*_sample_point(self.sample(), t))

    def tangent_at(self, t: float):
        return tuple(_unit(_derivatives(self.sample(), _position(t))[0]))

    def normal_at(self, t: float):
        x, y = self.tangent_at(t)
        return (-y, x)

    def curvature_at(self, t: float):
        first, second = _derivatives(self.sample(), _position(t))
        speed = np.linalg.norm(first)
        if speed == 0:
            raise ValueError("Curvature is undefined where the curve has zero speed")
        return float(abs(first[0] * second[1] - first[1] * second[0]) / speed ** 3)

    def arc_length(self, samples: int = None):
        return _arc_length(self.sample(samples=samples))

    @property
    def bounds(self):
        return _bounds(self.sample())

    def intersections(self, other, tolerance: float = 1e-6, samples: int = None):
        """Return approximate intersections with another 2D curve."""
        from kiwicalc.geometry.points import Point2D
        if not isinstance(other, Curve2D):
            raise TypeError("other must be a Curve2D")
        tolerance = float(tolerance)
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        first = np.column_stack(self.sample(samples=samples))
        second = np.column_stack(other.sample(samples=samples))
        found = []
        for p1, p2 in zip(first[:-1], first[1:]):
            if not np.all(np.isfinite((p1, p2))):
                continue
            r = p2 - p1
            for q1, q2 in zip(second[:-1], second[1:]):
                if not np.all(np.isfinite((q1, q2))):
                    continue
                s = q2 - q1
                denominator = r[0] * s[1] - r[1] * s[0]
                if abs(denominator) <= tolerance:
                    continue
                delta = q1 - p1
                t = (delta[0] * s[1] - delta[1] * s[0]) / denominator
                u = (delta[0] * r[1] - delta[1] * r[0]) / denominator
                if -tolerance <= t <= 1 + tolerance and -tolerance <= u <= 1 + tolerance:
                    point = p1 + t * r
                    if not any(np.linalg.norm(point - existing) <= tolerance * 10 for existing in found):
                        found.append(point)
        return [Point2D(*point) for point in found]

    def to_dict(self):
        from kiwicalc.serialization import curve_to_dict
        return curve_to_dict(self)

    def to_json(self, **kwargs):
        import json
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_dict(data):
        from kiwicalc.serialization import curve_from_dict
        return curve_from_dict(data)


class TransformedCurve2D(Curve2D):
    def __init__(self, curve: Curve2D, matrix):
        if not isinstance(curve, Curve2D):
            raise TypeError("curve must be a Curve2D")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("a 2D transformation matrix must be a finite 3x3 matrix")
        if isinstance(curve, TransformedCurve2D):
            self.source = curve.source
            self.matrix = matrix @ curve.matrix
        else:
            self.source = curve
            self.matrix = matrix
        super().__init__(curve.samples)

    def sample(self, samples: int = None):
        x, y = self.source.sample(samples=samples)
        points = np.vstack((x, y, np.ones_like(x)))
        transformed = self.matrix @ points
        return transformed[0], transformed[1]


class ParametricCurve2D(Curve2D):
    def __init__(self, x: ExpressionLike, y: ExpressionLike, t_range=(0, 2 * pi), samples: int = 500, sampling="fixed", tolerance=1e-3, max_depth=10):
        super().__init__(samples)
        self._x_source, self._y_source = x, y
        self._x = _as_callable(x, ("t",))
        self._y = _as_callable(y, ("t",))
        self._t_range = _validate_range(t_range, "t_range")
        self.sampling, self.tolerance, self.max_depth = _adaptive_options(sampling, tolerance, max_depth)

    @property
    def t_range(self):
        return self._t_range

    def sample(self, samples: int = None):
        if samples is None and self.sampling == "adaptive":
            return _adaptive_parametric((self._x, self._y), self.t_range, self.tolerance, self.max_depth)
        count = self.samples if samples is None else _validate_samples(samples)
        t = np.linspace(*self.t_range, count)
        return _evaluate_1d(self._x, t), _evaluate_1d(self._y, t)

    def adaptive(self, tolerance=1e-3, max_depth=10):
        return ParametricCurve2D(self._x, self._y, self.t_range, self.samples, "adaptive", tolerance, max_depth)


class PolarCurve2D(ParametricCurve2D):
    def __init__(self, radius: ExpressionLike, theta_range=(0, 2 * pi), samples: int = 500, sampling="fixed", tolerance=1e-3, max_depth=10):
        self._radius_source = radius
        radius_func = _as_callable(radius, ("theta",))
        self._radius = radius_func
        super().__init__(
            lambda theta: radius_func(theta) * cos(theta),
            lambda theta: radius_func(theta) * sin(theta),
            t_range=theta_range,
            samples=samples,
            sampling=sampling,
            tolerance=tolerance,
            max_depth=max_depth,
        )

    @property
    def theta_range(self):
        return self.t_range


class ImplicitCurve2D(Curve2D):
    def __init__(self, equation: ExpressionLike, x_range=(-10, 10), y_range=(-10, 10), resolution: int = 250, level: float = 0):
        super().__init__(resolution)
        self._equation_source = equation
        self._equation = _as_callable(equation, ("x", "y"), implicit=True)
        self.x_range = _validate_range(x_range, "x_range")
        self.y_range = _validate_range(y_range, "y_range")
        self.level = float(level)

    @property
    def resolution(self):
        return self.samples

    def sample(self, samples: int = None):
        count = self.resolution if samples is None else _validate_samples(samples)
        x_values = np.linspace(*self.x_range, count)
        y_values = np.linspace(*self.y_range, count)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.full(X.shape, np.nan, dtype=float)
        with np.errstate(all="ignore"):
            for index in np.ndindex(X.shape):
                try:
                    value = float(self._equation(float(X[index]), float(Y[index])))
                    Z[index] = value if np.isfinite(value) else np.nan
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    pass
        return X, Y, Z

    def plot(self, show: bool = True, fig=None, ax=None, label: str = None, **style):
        from kiwicalc.plotting.plots import plot_implicit_curve_2d
        return plot_implicit_curve_2d(self, show=show, fig=fig, ax=ax, label=label, **style)


class BezierCurve2D(Curve2D):
    def __init__(self, control_points, samples: int = 500):
        super().__init__(samples)
        self._control_points = _points_array(control_points, 2)

    @property
    def control_points(self):
        return self._control_points.copy()

    def sample(self, samples: int = None):
        count = self.samples if samples is None else _validate_samples(samples)
        t = np.linspace(0, 1, count)
        degree = len(self._control_points) - 1
        points = sum(
            comb(degree, index) * ((1 - t) ** (degree - index) * t ** index)[:, None] * point
            for index, point in enumerate(self._control_points)
        )
        return points[:, 0], points[:, 1]


def _catmull_rom(control_points: np.ndarray, count: int, closed: bool) -> np.ndarray:
    points = control_points
    if closed:
        points = np.vstack((points[-1], points, points[0], points[1]))
        segment_count = len(control_points)
    else:
        points = np.vstack((points[0], points, points[-1]))
        segment_count = len(control_points) - 1
    per_segment = max(2, int(np.ceil(count / segment_count)))
    pieces = []
    for index in range(segment_count):
        p0, p1, p2, p3 = points[index:index + 4]
        t = np.linspace(0, 1, per_segment, endpoint=index == segment_count - 1)[:, None]
        piece = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
        )
        pieces.append(piece)
    return np.vstack(pieces)


class CatmullRomSpline2D(Curve2D):
    def __init__(self, control_points, samples: int = 500, closed: bool = False):
        super().__init__(samples)
        self._control_points = _points_array(control_points, 2, minimum=3)
        self.closed = bool(closed)

    @property
    def control_points(self):
        return self._control_points.copy()

    def sample(self, samples: int = None):
        count = self.samples if samples is None else _validate_samples(samples)
        points = _catmull_rom(self._control_points, count, self.closed)
        return points[:, 0], points[:, 1]


class Ellipse(ParametricCurve2D):
    def __init__(self, radius_x: float = 2, radius_y: float = 1, center=(0, 0), rotation: float = 0, samples: int = 500):
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("ellipse radii must be positive")
        self.radius_x, self.radius_y = float(radius_x), float(radius_y)
        self.center = (float(center[0]), float(center[1]))
        self.rotation = float(rotation)
        super().__init__(
            lambda t: _rotate_2d(self.radius_x * cos(t), self.radius_y * sin(t), self.rotation, self.center)[0],
            lambda t: _rotate_2d(self.radius_x * cos(t), self.radius_y * sin(t), self.rotation, self.center)[1],
            samples=samples,
        )


class Arc(ParametricCurve2D):
    def __init__(self, radius: float = 1, center=(0, 0), start_angle: float = 0, end_angle: float = pi, samples: int = 250):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.center = (float(center[0]), float(center[1]))
        self.start_angle, self.end_angle = _validate_range((start_angle, end_angle), "angle range")
        super().__init__(
            lambda t: self.center[0] + self.radius * cos(t),
            lambda t: self.center[1] + self.radius * sin(t),
            t_range=(self.start_angle, self.end_angle),
            samples=samples,
        )


class Parabola(ParametricCurve2D):
    def __init__(self, focal_length: float = 1, vertex=(0, 0), rotation: float = pi / 2, t_range=(-3, 3), samples: int = 500):
        if focal_length == 0:
            raise ValueError("focal_length cannot be zero")
        self.focal_length = float(focal_length)
        self.vertex = (float(vertex[0]), float(vertex[1]))
        self.rotation = float(rotation)
        super().__init__(
            lambda t: _rotate_2d(self.focal_length * t * t, 2 * self.focal_length * t, self.rotation, self.vertex)[0],
            lambda t: _rotate_2d(self.focal_length * t * t, 2 * self.focal_length * t, self.rotation, self.vertex)[1],
            t_range=t_range,
            samples=samples,
        )


class Hyperbola(Curve2D):
    def __init__(self, semi_transverse: float = 1, semi_conjugate: float = 1, center=(0, 0), rotation: float = 0, t_range=(-2, 2), samples: int = 500):
        super().__init__(samples)
        if semi_transverse <= 0 or semi_conjugate <= 0:
            raise ValueError("hyperbola semi-axes must be positive")
        self.semi_transverse = float(semi_transverse)
        self.semi_conjugate = float(semi_conjugate)
        self.center = (float(center[0]), float(center[1]))
        self.rotation = float(rotation)
        self.t_range = _validate_range(t_range, "t_range")

    def sample(self, samples: int = None):
        count = self.samples if samples is None else _validate_samples(samples)
        t = np.linspace(*self.t_range, max(2, count // 2))
        x = self.semi_transverse * np.cosh(t)
        y = self.semi_conjugate * np.sinh(t)
        right = _rotate_2d(x, y, self.rotation, self.center)
        left = _rotate_2d(-x, y, self.rotation, self.center)
        return (
            np.concatenate((right[0], [np.nan], left[0])),
            np.concatenate((right[1], [np.nan], left[1])),
        )


class ArchimedeanSpiral(PolarCurve2D):
    def __init__(self, initial_radius: float = 0, growth: float = 0.2, theta_range=(0, 6 * pi), samples: int = 750):
        self.initial_radius, self.growth = float(initial_radius), float(growth)
        super().__init__(lambda theta: self.initial_radius + self.growth * theta, theta_range, samples)


class LogarithmicSpiral(PolarCurve2D):
    def __init__(self, initial_radius: float = 0.1, growth: float = 0.15, theta_range=(0, 6 * pi), samples: int = 750):
        if initial_radius <= 0:
            raise ValueError("initial_radius must be positive")
        self.initial_radius, self.growth = float(initial_radius), float(growth)
        super().__init__(lambda theta: self.initial_radius * exp(self.growth * theta), theta_range, samples)


class LissajousCurve2D(ParametricCurve2D):
    def __init__(self, amplitudes=(1, 1), frequencies=(3, 2), phase: float = pi / 2, t_range=(0, 2 * pi), samples: int = 750):
        self.amplitudes = tuple(float(value) for value in amplitudes)
        self.frequencies = tuple(float(value) for value in frequencies)
        if len(self.amplitudes) != 2 or len(self.frequencies) != 2:
            raise ValueError("amplitudes and frequencies must each contain two values")
        self.phase = float(phase)
        super().__init__(
            lambda t: self.amplitudes[0] * sin(self.frequencies[0] * t + self.phase),
            lambda t: self.amplitudes[1] * sin(self.frequencies[1] * t),
            t_range,
            samples,
        )


class Cardioid(PolarCurve2D):
    def __init__(self, scale: float = 1, center=(0, 0), theta_range=(0, 2 * pi), samples: int = 750):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 2:
            raise ValueError("center must contain two values")
        ParametricCurve2D.__init__(
            self,
            lambda t: self.center[0] + self.scale * (1 - cos(t)) * cos(t),
            lambda t: self.center[1] + self.scale * (1 - cos(t)) * sin(t),
            theta_range,
            samples,
        )


class RoseCurve(PolarCurve2D):
    def __init__(self, radius: float = 1, petals: int = 4, theta_range=(0, 2 * pi), samples: int = 1000):
        if radius <= 0 or not isinstance(petals, int) or isinstance(petals, bool) or petals <= 0:
            raise ValueError("radius and petals must be positive")
        self.radius, self.petals = float(radius), petals
        super().__init__(lambda theta: self.radius * cos(self.petals * theta), theta_range, samples)


class Cycloid(ParametricCurve2D):
    def __init__(self, radius: float = 1, turns: float = 2, start=(0, 0), samples: int = 750):
        if radius <= 0 or turns <= 0:
            raise ValueError("radius and turns must be positive")
        self.radius, self.turns = float(radius), float(turns)
        self.start = tuple(float(value) for value in start)
        if len(self.start) != 2:
            raise ValueError("start must contain two values")
        super().__init__(
            lambda t: self.start[0] + self.radius * (t - sin(t)),
            lambda t: self.start[1] + self.radius * (1 - cos(t)),
            (0, 2 * pi * self.turns),
            samples,
        )


class Epicycloid(ParametricCurve2D):
    def __init__(self, fixed_radius: float = 3, rolling_radius: float = 1, samples: int = 1000):
        if fixed_radius <= 0 or rolling_radius <= 0:
            raise ValueError("radii must be positive")
        self.fixed_radius, self.rolling_radius = float(fixed_radius), float(rolling_radius)
        ratio = (self.fixed_radius + self.rolling_radius) / self.rolling_radius
        super().__init__(
            lambda t: (self.fixed_radius + self.rolling_radius) * cos(t) - self.rolling_radius * cos(ratio * t),
            lambda t: (self.fixed_radius + self.rolling_radius) * sin(t) - self.rolling_radius * sin(ratio * t),
            samples=samples,
        )


class Hypocycloid(ParametricCurve2D):
    def __init__(self, fixed_radius: float = 4, rolling_radius: float = 1, samples: int = 1000):
        if fixed_radius <= 0 or rolling_radius <= 0 or rolling_radius >= fixed_radius:
            raise ValueError("radii must be positive and rolling_radius must be smaller than fixed_radius")
        self.fixed_radius, self.rolling_radius = float(fixed_radius), float(rolling_radius)
        ratio = (self.fixed_radius - self.rolling_radius) / self.rolling_radius
        super().__init__(
            lambda t: (self.fixed_radius - self.rolling_radius) * cos(t) + self.rolling_radius * cos(ratio * t),
            lambda t: (self.fixed_radius - self.rolling_radius) * sin(t) - self.rolling_radius * sin(ratio * t),
            samples=samples,
        )


class Superellipse(ParametricCurve2D):
    def __init__(self, radius_x: float = 2, radius_y: float = 1, exponent: float = 4, center=(0, 0), samples: int = 750):
        if radius_x <= 0 or radius_y <= 0 or exponent <= 0:
            raise ValueError("radii and exponent must be positive")
        self.radius_x, self.radius_y, self.exponent = float(radius_x), float(radius_y), float(exponent)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 2:
            raise ValueError("center must contain two values")
        power = 2 / self.exponent
        signed_power = lambda value: np.sign(value) * abs(value) ** power
        super().__init__(
            lambda t: self.center[0] + self.radius_x * signed_power(cos(t)),
            lambda t: self.center[1] + self.radius_y * signed_power(sin(t)),
            samples=samples,
        )


class Catenary(ParametricCurve2D):
    def __init__(self, scale: float = 1, t_range=(-3, 3), vertex=(0, 0), samples: int = 500):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        self.vertex = tuple(float(value) for value in vertex)
        if len(self.vertex) != 2:
            raise ValueError("vertex must contain two values")
        super().__init__(
            lambda t: self.vertex[0] + t,
            lambda t: self.vertex[1] + self.scale * (cosh(t / self.scale) - 1),
            t_range,
            samples,
        )


class Involute(ParametricCurve2D):
    def __init__(self, radius: float = 1, t_range=(0, 2 * pi), center=(0, 0), samples: int = 750):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 2:
            raise ValueError("center must contain two values")
        super().__init__(
            lambda t: self.center[0] + self.radius * (cos(t) + t * sin(t)),
            lambda t: self.center[1] + self.radius * (sin(t) - t * cos(t)),
            t_range,
            samples,
        )


class Curve3D(IPlottable3D):
    dimensions = 3

    def __init__(self, samples: int = 750):
        self._samples = _validate_samples(samples)

    @property
    def samples(self):
        return self._samples

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def sample(self, samples: int = None):
        """Return ``(x_values, y_values, z_values)`` NumPy arrays."""

    def plot3d(self, show: bool = True, fig=None, ax=None, label: str = None, **style):
        from kiwicalc.plotting.plots import plot_curve_3d
        return plot_curve_3d(self, show=show, fig=fig, ax=ax, label=label, **style)

    def plot(self, *args, **kwargs):
        return self.plot3d(*args, **kwargs)

    def transform(self, matrix):
        """Return a transformed copy using a 4x4 homogeneous matrix."""
        return TransformedCurve3D(self, matrix)

    def translate(self, x: float = 0, y: float = 0, z: float = 0):
        matrix = np.eye(4)
        matrix[:3, 3] = (x, y, z)
        return self.transform(matrix)

    def scale(self, factor, y=None, z=None, center=(0, 0, 0)):
        sx = float(factor)
        sy = sx if y is None else float(y)
        sz = sx if z is None else float(z)
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must contain three values")
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag((sx, sy, sz))
        matrix[:3, 3] = center * (1 - np.asarray((sx, sy, sz)))
        return self.transform(matrix)

    def rotate(self, angle: float, axis=(0, 0, 1), center=(0, 0, 0)):
        axis = np.asarray(axis, dtype=float)
        center = np.asarray(center, dtype=float)
        if axis.shape != (3,) or center.shape != (3,):
            raise ValueError("axis and center must contain three values")
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("rotation axis cannot be zero")
        x, y, z = axis / norm
        cosine, sine = cos(angle), sin(angle)
        one_minus = 1 - cosine
        rotation = np.asarray((
            (cosine + x*x*one_minus, x*y*one_minus - z*sine, x*z*one_minus + y*sine),
            (y*x*one_minus + z*sine, cosine + y*y*one_minus, y*z*one_minus - x*sine),
            (z*x*one_minus - y*sine, z*y*one_minus + x*sine, cosine + z*z*one_minus),
        ))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center - rotation @ center
        return self.transform(matrix)

    def rotate_x(self, angle: float, center=(0, 0, 0)):
        return self.rotate(angle, axis=(1, 0, 0), center=center)

    def rotate_y(self, angle: float, center=(0, 0, 0)):
        return self.rotate(angle, axis=(0, 1, 0), center=center)

    def rotate_z(self, angle: float, center=(0, 0, 0)):
        return self.rotate(angle, axis=(0, 0, 1), center=center)

    def reflect(self, plane="xy"):
        normalized = str(plane).lower()
        factors = {"xy": (1, 1, -1), "xz": (1, -1, 1), "yz": (-1, 1, 1), "origin": (-1, -1, -1)}
        if normalized not in factors:
            raise ValueError("plane must be 'xy', 'xz', 'yz', or 'origin'")
        return self.scale(*factors[normalized])

    def point_at(self, t: float):
        from kiwicalc.geometry.points import Point3D
        return Point3D(*_sample_point(self.sample(), t))

    def tangent_at(self, t: float):
        return tuple(_unit(_derivatives(self.sample(), _position(t))[0]))

    def normal_at(self, t: float):
        first, second = _derivatives(self.sample(), _position(t))
        tangent = _unit(first)
        normal_component = second - np.dot(second, tangent) * tangent
        return tuple(_unit(normal_component, "The curve has no defined normal at this point"))

    def curvature_at(self, t: float):
        first, second = _derivatives(self.sample(), _position(t))
        speed = np.linalg.norm(first)
        if speed == 0:
            raise ValueError("Curvature is undefined where the curve has zero speed")
        return float(np.linalg.norm(np.cross(first, second)) / speed ** 3)

    def arc_length(self, samples: int = None):
        return _arc_length(self.sample(samples=samples))

    @property
    def bounds(self):
        return _bounds(self.sample())

    def to_dict(self):
        from kiwicalc.serialization import curve_to_dict
        return curve_to_dict(self)

    def to_json(self, **kwargs):
        import json
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_dict(data):
        from kiwicalc.serialization import curve_from_dict
        return curve_from_dict(data)


class TransformedCurve3D(Curve3D):
    def __init__(self, curve: Curve3D, matrix):
        if not isinstance(curve, Curve3D):
            raise TypeError("curve must be a Curve3D")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("a 3D transformation matrix must be a finite 4x4 matrix")
        if isinstance(curve, TransformedCurve3D):
            self.source = curve.source
            self.matrix = matrix @ curve.matrix
        else:
            self.source = curve
            self.matrix = matrix
        super().__init__(curve.samples)

    def sample(self, samples: int = None):
        x, y, z = self.source.sample(samples=samples)
        points = np.vstack((x, y, z, np.ones_like(x)))
        transformed = self.matrix @ points
        return transformed[0], transformed[1], transformed[2]


class ParametricCurve3D(Curve3D):
    def __init__(self, x: ExpressionLike, y: ExpressionLike, z: ExpressionLike, t_range=(0, 2 * pi), samples: int = 750, sampling="fixed", tolerance=1e-3, max_depth=10):
        super().__init__(samples)
        self._coordinate_sources = (x, y, z)
        self._coordinates = tuple(_as_callable(value, ("t",)) for value in (x, y, z))
        self.t_range = _validate_range(t_range, "t_range")
        self.sampling, self.tolerance, self.max_depth = _adaptive_options(sampling, tolerance, max_depth)

    def sample(self, samples: int = None):
        if samples is None and self.sampling == "adaptive":
            return _adaptive_parametric(self._coordinates, self.t_range, self.tolerance, self.max_depth)
        count = self.samples if samples is None else _validate_samples(samples)
        t = np.linspace(*self.t_range, count)
        return tuple(_evaluate_1d(func, t) for func in self._coordinates)

    def adaptive(self, tolerance=1e-3, max_depth=10):
        return ParametricCurve3D(*self._coordinates, t_range=self.t_range, samples=self.samples, sampling="adaptive", tolerance=tolerance, max_depth=max_depth)


class BezierCurve3D(Curve3D):
    def __init__(self, control_points, samples: int = 750):
        super().__init__(samples)
        self._control_points = _points_array(control_points, 3)

    @property
    def control_points(self):
        return self._control_points.copy()

    def sample(self, samples: int = None):
        count = self.samples if samples is None else _validate_samples(samples)
        t = np.linspace(0, 1, count)
        degree = len(self._control_points) - 1
        points = sum(
            comb(degree, index) * ((1 - t) ** (degree - index) * t ** index)[:, None] * point
            for index, point in enumerate(self._control_points)
        )
        return points[:, 0], points[:, 1], points[:, 2]


class CatmullRomSpline3D(Curve3D):
    def __init__(self, control_points, samples: int = 750, closed: bool = False):
        super().__init__(samples)
        self._control_points = _points_array(control_points, 3, minimum=3)
        self.closed = bool(closed)

    @property
    def control_points(self):
        return self._control_points.copy()

    def sample(self, samples: int = None):
        count = self.samples if samples is None else _validate_samples(samples)
        points = _catmull_rom(self._control_points, count, self.closed)
        return points[:, 0], points[:, 1], points[:, 2]


class Line3D(ParametricCurve3D):
    def __init__(self, point=(0, 0, 0), direction=(1, 0, 0), t_range=(-5, 5), samples: int = 250):
        self.point = tuple(float(value) for value in point)
        self.direction = tuple(float(value) for value in direction)
        if len(self.point) != 3 or len(self.direction) != 3 or not any(self.direction):
            raise ValueError("point and non-zero direction must contain three values")
        super().__init__(*(lambda t, i=index: self.point[i] + self.direction[i] * t for index in range(3)), t_range=t_range, samples=samples)


class Helix(ParametricCurve3D):
    def __init__(self, radius: float = 1, pitch: float = 1, turns: float = 3, center=(0, 0, 0), samples: int = 750):
        if radius <= 0 or turns <= 0:
            raise ValueError("radius and turns must be positive")
        self.radius, self.pitch, self.turns = float(radius), float(pitch), float(turns)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda t: self.center[0] + self.radius * cos(t),
            lambda t: self.center[1] + self.radius * sin(t),
            lambda t: self.center[2] + self.pitch * t / (2 * pi),
            (0, 2 * pi * self.turns),
            samples,
        )


class LissajousCurve3D(ParametricCurve3D):
    def __init__(self, amplitudes=(1, 1, 1), frequencies=(3, 2, 5), phases=(pi / 2, 0, 0), t_range=(0, 2 * pi), samples: int = 1000):
        self.amplitudes = tuple(float(value) for value in amplitudes)
        self.frequencies = tuple(float(value) for value in frequencies)
        self.phases = tuple(float(value) for value in phases)
        if not all(len(values) == 3 for values in (self.amplitudes, self.frequencies, self.phases)):
            raise ValueError("amplitudes, frequencies, and phases must contain three values")
        coordinates = tuple(
            (lambda t, i=index: self.amplitudes[i] * sin(self.frequencies[i] * t + self.phases[i]))
            for index in range(3)
        )
        super().__init__(*coordinates, t_range=t_range, samples=samples)


class TorusKnot(ParametricCurve3D):
    def __init__(self, p: int = 2, q: int = 3, major_radius: float = 2, minor_radius: float = 1, samples: int = 1200):
        if not isinstance(p, int) or not isinstance(q, int) or p == 0 or q == 0:
            raise ValueError("p and q must be non-zero integers")
        if major_radius <= 0 or minor_radius <= 0:
            raise ValueError("torus radii must be positive")
        self.p, self.q = p, q
        self.major_radius, self.minor_radius = float(major_radius), float(minor_radius)
        radial = lambda t: self.major_radius + self.minor_radius * cos(self.q * t)
        super().__init__(
            lambda t: radial(t) * cos(self.p * t),
            lambda t: radial(t) * sin(self.p * t),
            lambda t: self.minor_radius * sin(self.q * t),
            samples=samples,
        )


class TrefoilKnot(TorusKnot):
    def __init__(self, major_radius: float = 2, minor_radius: float = 1, samples: int = 1200):
        super().__init__(2, 3, major_radius, minor_radius, samples)


class FigureEightKnot(ParametricCurve3D):
    def __init__(self, scale: float = 1, samples: int = 1200):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        super().__init__(
            lambda t: self.scale * (2 + cos(2 * t)) * cos(3 * t),
            lambda t: self.scale * (2 + cos(2 * t)) * sin(3 * t),
            lambda t: self.scale * sin(4 * t),
            samples=samples,
        )


__all__ = [
    "Curve2D", "TransformedCurve2D", "ParametricCurve2D", "PolarCurve2D", "ImplicitCurve2D",
    "BezierCurve2D", "CatmullRomSpline2D", "Ellipse", "Arc", "Parabola",
    "Hyperbola", "ArchimedeanSpiral", "LogarithmicSpiral", "LissajousCurve2D",
    "Cardioid", "RoseCurve", "Cycloid", "Epicycloid", "Hypocycloid",
    "Superellipse", "Catenary", "Involute",
    "Curve3D", "TransformedCurve3D", "ParametricCurve3D", "BezierCurve3D", "CatmullRomSpline3D",
    "Line3D", "Helix", "LissajousCurve3D", "TorusKnot", "TrefoilKnot", "FigureEightKnot",
]
