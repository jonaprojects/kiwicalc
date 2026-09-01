from __future__ import annotations

"""Sampleable 3D surfaces with convenient mathematical constructors."""

from abc import abstractmethod
from math import cos, pi, sin
from typing import Callable, Tuple, Union

import numpy as np

from kiwicalc.core.interfaces import IExpression, IPlottable3D
from kiwicalc.core.utils import to_lambda


ExpressionLike = Union[str, Callable, IExpression, int, float]


def _as_callable(value: ExpressionLike, variables: Tuple[str, ...]) -> Callable:
    if isinstance(value, IExpression):
        return value.to_lambda(variables=variables)
    if isinstance(value, str):
        if "=" in value:
            _, value = (part.strip() for part in value.split("=", 1))
        return to_lambda(value, variables)
    if callable(value):
        return value
    if isinstance(value, (int, float)):
        return lambda *args: value
    raise TypeError("Surface coordinates must be numbers, callables, strings, or KiwiCalc expressions")


def _range(bounds, name):
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    start, stop = float(bounds[0]), float(bounds[1])
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError(f"{name} must be a finite increasing range")
    return start, stop


def _resolution(value):
    if isinstance(value, int):
        value = (value, value)
    if len(value) != 2 or any(not isinstance(item, int) or isinstance(item, bool) or item < 2 for item in value):
        raise ValueError("resolution must be an integer or pair of integers of at least 2")
    return tuple(value)


def _evaluate_2d(func, first, second):
    result = np.full(first.shape, np.nan, dtype=float)
    with np.errstate(all="ignore"):
        for index in np.ndindex(first.shape):
            try:
                value = float(func(float(first[index]), float(second[index])))
                result[index] = value if np.isfinite(value) else np.nan
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                pass
    return result


class Surface3D(IPlottable3D):
    dimensions = 3

    def __init__(self, resolution=80):
        self._resolution = _resolution(resolution)

    @property
    def resolution(self):
        return self._resolution

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def sample(self, resolution=None):
        """Return ``(X, Y, Z)`` mesh arrays."""

    def plot3d(self, show: bool = True, fig=None, ax=None, label: str = None, wireframe: bool = False, **style):
        from kiwicalc.plotting.plots import plot_surface_3d
        return plot_surface_3d(self, show=show, fig=fig, ax=ax, label=label, wireframe=wireframe, **style)

    def plot(self, *args, **kwargs):
        return self.plot3d(*args, **kwargs)

    def to_dict(self):
        from kiwicalc.serialization import surface_to_dict
        return surface_to_dict(self)

    def to_json(self, **kwargs):
        import json
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_dict(data):
        from kiwicalc.serialization import surface_from_dict
        return surface_from_dict(data)


class ExplicitSurface3D(Surface3D):
    def __init__(self, z: ExpressionLike, x_range=(-5, 5), y_range=(-5, 5), resolution=80):
        super().__init__(resolution)
        self._z_source = z
        self._z = _as_callable(z, ("x", "y"))
        self.x_range = _range(x_range, "x_range")
        self.y_range = _range(y_range, "y_range")

    def sample(self, resolution=None):
        x_count, y_count = self.resolution if resolution is None else _resolution(resolution)
        x = np.linspace(*self.x_range, x_count)
        y = np.linspace(*self.y_range, y_count)
        X, Y = np.meshgrid(x, y)
        return X, Y, _evaluate_2d(self._z, X, Y)


class ParametricSurface3D(Surface3D):
    def __init__(self, x: ExpressionLike, y: ExpressionLike, z: ExpressionLike, u_range=(0, 2 * pi), v_range=(0, pi), resolution=80):
        super().__init__(resolution)
        self._coordinate_sources = (x, y, z)
        self._coordinates = tuple(_as_callable(value, ("u", "v")) for value in (x, y, z))
        self.u_range = _range(u_range, "u_range")
        self.v_range = _range(v_range, "v_range")

    def sample(self, resolution=None):
        u_count, v_count = self.resolution if resolution is None else _resolution(resolution)
        u = np.linspace(*self.u_range, u_count)
        v = np.linspace(*self.v_range, v_count)
        U, V = np.meshgrid(u, v)
        return tuple(_evaluate_2d(func, U, V) for func in self._coordinates)


class Sphere(ParametricSurface3D):
    def __init__(self, radius: float = 1, center=(0, 0, 0), resolution=80):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.radius * sin(v) * cos(u),
            lambda u, v: self.center[1] + self.radius * sin(v) * sin(u),
            lambda u, v: self.center[2] + self.radius * cos(v),
            resolution=resolution,
        )


class Ellipsoid(ParametricSurface3D):
    def __init__(self, radii=(2, 1, 1), center=(0, 0, 0), resolution=80):
        self.radii = tuple(float(value) for value in radii)
        self.center = tuple(float(value) for value in center)
        if len(self.radii) != 3 or any(value <= 0 for value in self.radii):
            raise ValueError("radii must contain three positive values")
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.radii[0] * sin(v) * cos(u),
            lambda u, v: self.center[1] + self.radii[1] * sin(v) * sin(u),
            lambda u, v: self.center[2] + self.radii[2] * cos(v),
            resolution=resolution,
        )


class Cylinder(ParametricSurface3D):
    def __init__(self, radius: float = 1, height: float = 2, center=(0, 0, 0), resolution=80):
        if radius <= 0 or height <= 0:
            raise ValueError("radius and height must be positive")
        self.radius, self.height = float(radius), float(height)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.radius * cos(u),
            lambda u, v: self.center[1] + self.radius * sin(u),
            lambda u, v: self.center[2] + v,
            v_range=(-self.height / 2, self.height / 2),
            resolution=resolution,
        )


class Cone(ParametricSurface3D):
    def __init__(self, radius: float = 1, height: float = 2, center=(0, 0, 0), resolution=80):
        if radius <= 0 or height <= 0:
            raise ValueError("radius and height must be positive")
        self.radius, self.height = float(radius), float(height)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.radius * (1 - v) * cos(u),
            lambda u, v: self.center[1] + self.radius * (1 - v) * sin(u),
            lambda u, v: self.center[2] + self.height * (v - 0.5),
            v_range=(0, 1),
            resolution=resolution,
        )


class Torus(ParametricSurface3D):
    def __init__(self, major_radius: float = 2, minor_radius: float = 0.5, center=(0, 0, 0), resolution=100):
        if major_radius <= 0 or minor_radius <= 0:
            raise ValueError("torus radii must be positive")
        self.major_radius, self.minor_radius = float(major_radius), float(minor_radius)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        radial = lambda v: self.major_radius + self.minor_radius * cos(v)
        super().__init__(
            lambda u, v: self.center[0] + radial(v) * cos(u),
            lambda u, v: self.center[1] + radial(v) * sin(u),
            lambda u, v: self.center[2] + self.minor_radius * sin(v),
            v_range=(0, 2 * pi),
            resolution=resolution,
        )


class Paraboloid(ParametricSurface3D):
    def __init__(self, scale_x: float = 1, scale_y: float = 1, radius: float = 3, center=(0, 0, 0), resolution=80):
        if scale_x <= 0 or scale_y <= 0 or radius <= 0:
            raise ValueError("scales and radius must be positive")
        self.scale_x, self.scale_y, self.radius = float(scale_x), float(scale_y), float(radius)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.scale_x * u * cos(v),
            lambda u, v: self.center[1] + self.scale_y * u * sin(v),
            lambda u, v: self.center[2] + u * u,
            u_range=(0, self.radius),
            v_range=(0, 2 * pi),
            resolution=resolution,
        )


class HyperbolicParaboloid(ExplicitSurface3D):
    def __init__(self, scale_x: float = 1, scale_y: float = 1, x_range=(-3, 3), y_range=(-3, 3), center=(0, 0, 0), resolution=80):
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("scales must be positive")
        self.scale_x, self.scale_y = float(scale_x), float(scale_y)
        self.center = tuple(float(value) for value in center)
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda x, y: self.center[2] + ((x - self.center[0]) / self.scale_x) ** 2 - ((y - self.center[1]) / self.scale_y) ** 2,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
        )


class Hyperboloid(ParametricSurface3D):
    def __init__(self, radii=(2, 2, 1), u_range=(-1.5, 1.5), center=(0, 0, 0), resolution=100):
        self.radii = tuple(float(value) for value in radii)
        self.center = tuple(float(value) for value in center)
        if len(self.radii) != 3 or any(value <= 0 for value in self.radii):
            raise ValueError("radii must contain three positive values")
        if len(self.center) != 3:
            raise ValueError("center must contain three values")
        super().__init__(
            lambda u, v: self.center[0] + self.radii[0] * np.cosh(u) * cos(v),
            lambda u, v: self.center[1] + self.radii[1] * np.cosh(u) * sin(v),
            lambda u, v: self.center[2] + self.radii[2] * np.sinh(u),
            u_range=u_range,
            v_range=(0, 2 * pi),
            resolution=resolution,
        )


__all__ = [
    "Surface3D", "ExplicitSurface3D", "ParametricSurface3D",
    "Sphere", "Ellipsoid", "Cylinder", "Cone", "Torus",
    "Paraboloid", "HyperbolicParaboloid", "Hyperboloid",
]
