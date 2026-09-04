"""Composable affine transformations for KiwiCalc geometry objects."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from numbers import Number
from typing import Any, Optional

import numpy as np

from kiwicalc.linalg.matrix import Matrix


@dataclass(frozen=True, init=False)
class AffineTransformation:
    """An immutable 2D or 3D transformation stored in homogeneous form.

    Use :meth:`then` or the chainable verbs to express operations in reading
    order. For example, ``rotation(...).translate(2, 0)`` rotates first and
    translates second.
    """

    _matrix: Matrix

    def __init__(self, matrix):
        source = matrix if isinstance(matrix, Matrix) else Matrix(matrix)
        values = source._as_numeric_array()
        if values.shape not in ((3, 3), (4, 4)):
            raise ValueError("An affine transformation requires a 3x3 (2D) or 4x4 (3D) matrix")
        expected_last_row = np.zeros(values.shape[1])
        expected_last_row[-1] = 1
        if not np.allclose(values[-1], expected_last_row, atol=1e-12, rtol=0):
            raise ValueError("The final homogeneous row must be [0, ..., 0, 1]")
        if np.iscomplexobj(values) and np.max(np.abs(np.imag(values))) > 1e-12:
            raise ValueError("Affine transformations require real matrix values")
        object.__setattr__(self, "_matrix", Matrix(np.real(values).tolist()))

    @property
    def matrix(self) -> Matrix:
        """Return an independent copy of the homogeneous matrix."""
        return self._matrix.copy()

    @property
    def dimension(self) -> int:
        return self._matrix.num_of_rows - 1

    @property
    def linear(self) -> Matrix:
        """Return the linear part of this transformation."""
        return Matrix(self._matrix.to_numpy()[:self.dimension, :self.dimension].tolist())

    @property
    def offset(self) -> tuple:
        """Return the translation component."""
        return tuple(self._matrix.to_numpy()[:self.dimension, self.dimension].tolist())

    @property
    def determinant(self) -> float:
        """Return the signed area or volume scale factor."""
        return float(np.linalg.det(self.linear.to_numpy(dtype=float)))

    @property
    def preserves_orientation(self) -> bool:
        return self.determinant > 0

    @property
    def is_rigid(self) -> bool:
        """Whether lengths and angles are preserved, allowing reflections."""
        linear = self.linear.to_numpy(dtype=float)
        return bool(np.allclose(linear.T @ linear, np.eye(self.dimension), atol=1e-10, rtol=0))

    @classmethod
    def from_matrix(cls, matrix) -> 'AffineTransformation':
        """Create a transformation from a homogeneous 3×3 or 4×4 matrix."""
        return cls(matrix if isinstance(matrix, Matrix) else Matrix(matrix))

    @classmethod
    def from_linear(cls, linear, translation=None) -> 'AffineTransformation':
        """Combine a 2×2 or 3×3 linear map with an optional translation."""
        source = linear if isinstance(linear, Matrix) else Matrix(linear)
        values = source._as_numeric_array()
        if values.shape not in ((2, 2), (3, 3)):
            raise ValueError("linear must be a 2x2 or 3x3 matrix")
        if np.iscomplexobj(values) and np.max(np.abs(np.imag(values))) > 1e-12:
            raise ValueError("Affine transformations require real matrix values")
        dimension = values.shape[0]
        offset = np.zeros(dimension) if translation is None else _coordinates(translation, dimension, "translation")
        homogeneous = np.eye(dimension + 1)
        homogeneous[:dimension, :dimension] = np.real(values)
        homogeneous[:dimension, dimension] = offset
        return cls(Matrix(homogeneous.tolist()))

    @classmethod
    def identity(cls, dimension=2) -> 'AffineTransformation':
        dimension = _dimension(dimension)
        return cls(Matrix.identity(dimension + 1))

    @classmethod
    def translation(cls, *offsets) -> 'AffineTransformation':
        values = _unpack_values(offsets, "translation")
        dimension = _dimension(len(values))
        return cls.from_linear(np.eye(dimension), values)

    @classmethod
    def scaling(cls, *factors, dimension=None, center=None) -> 'AffineTransformation':
        values = _unpack_values(factors, "scaling")
        if len(values) == 1:
            dimension = 2 if dimension is None else _dimension(dimension)
            values = values * dimension
        elif dimension is not None and _dimension(dimension) != len(values):
            raise ValueError("dimension must match the number of scale factors")
        dimension = _dimension(len(values))
        factors_array = _finite_real(values, "scale factors")
        center_array = np.zeros(dimension) if center is None else _coordinates(center, dimension, "center")
        linear = np.diag(factors_array)
        return cls.from_linear(linear, center_array - linear @ center_array)

    @classmethod
    def rotation(cls, angle, *, axis=None, center=None, degrees=False) -> 'AffineTransformation':
        angle = _angle(angle, degrees)
        if axis is None:
            center_array = np.zeros(2) if center is None else _coordinates(center, 2, "center")
            linear = np.asarray(((cos(angle), -sin(angle)), (sin(angle), cos(angle))))
            return cls.from_linear(linear, center_array - linear @ center_array)
        axis_array = _axis(axis)
        center_array = np.zeros(3) if center is None else _coordinates(center, 3, "center")
        x, y, z = axis_array
        cosine, sine, one_minus = cos(angle), sin(angle), 1 - cos(angle)
        linear = np.asarray((
            (cosine + x*x*one_minus, x*y*one_minus - z*sine, x*z*one_minus + y*sine),
            (y*x*one_minus + z*sine, cosine + y*y*one_minus, y*z*one_minus - x*sine),
            (z*x*one_minus - y*sine, z*y*one_minus + x*sine, cosine + z*z*one_minus),
        ))
        return cls.from_linear(linear, center_array - linear @ center_array)

    @classmethod
    def shearing(cls, *, x=0, y=0) -> 'AffineTransformation':
        """Create a 2D shear where ``x`` and ``y`` are the cross-axis factors."""
        x, y = _finite_real((x, y), "shear factors")
        return cls.from_linear(((1, x), (y, 1)))

    @classmethod
    def reflection(cls, axis="x", *, point=None, dimension=None) -> 'AffineTransformation':
        """Reflect across a named axis/plane or a hyperplane with a normal vector."""
        if isinstance(axis, str):
            normalized = axis.lower().replace(" ", "")
            named_normals = {
                "x": (0, 1), "y": (1, 0), "y=x": (1, -1), "y=-x": (1, 1),
                "xy": (0, 0, 1), "xz": (0, 1, 0), "yz": (1, 0, 0),
            }
            if normalized == "origin":
                actual_dimension = 2 if dimension is None else _dimension(dimension)
                center = np.zeros(actual_dimension) if point is None else _coordinates(point, actual_dimension, "point")
                return cls.from_linear(-np.eye(actual_dimension), 2 * center)
            if normalized not in named_normals:
                raise ValueError("axis must name x, y, y=x, y=-x, xy, xz, yz, origin, or be a normal vector")
            normal = np.asarray(named_normals[normalized], dtype=float)
        else:
            normal = _finite_real(_unpack_values((axis,), "normal"), "normal")
            _dimension(len(normal))
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            raise ValueError("reflection normal cannot be zero")
        normal /= normal_length
        actual_dimension = len(normal)
        anchor = np.zeros(actual_dimension) if point is None else _coordinates(point, actual_dimension, "point")
        linear = np.eye(actual_dimension) - 2 * np.outer(normal, normal)
        return cls.from_linear(linear, anchor - linear @ anchor)

    def then(self, other: 'AffineTransformation') -> 'AffineTransformation':
        """Return a transform that applies ``self`` first and ``other`` second."""
        if not isinstance(other, AffineTransformation):
            raise TypeError("then() expects an AffineTransformation")
        if self.dimension != other.dimension:
            raise ValueError("Cannot compose transformations with different dimensions")
        return AffineTransformation(other._matrix @ self._matrix)

    def translate(self, *offsets) -> 'AffineTransformation':
        transform = type(self).translation(*offsets)
        return self.then(_require_dimension(transform, self.dimension))

    def scale(self, *factors, center=None) -> 'AffineTransformation':
        if len(_unpack_values(factors, "scaling")) == 1:
            transform = type(self).scaling(*factors, dimension=self.dimension, center=center)
        else:
            transform = type(self).scaling(*factors, center=center)
        return self.then(_require_dimension(transform, self.dimension))

    def rotate(self, angle, *, axis=None, center=None, degrees=False) -> 'AffineTransformation':
        if self.dimension == 3 and axis is None:
            axis = "z"
        transform = type(self).rotation(angle, axis=axis, center=center, degrees=degrees)
        return self.then(_require_dimension(transform, self.dimension))

    def shear(self, *, x=0, y=0) -> 'AffineTransformation':
        if self.dimension != 2:
            raise ValueError("shear() currently supports 2D transformations")
        return self.then(type(self).shearing(x=x, y=y))

    def reflect(self, axis="x", *, point=None) -> 'AffineTransformation':
        transform = type(self).reflection(axis, point=point, dimension=self.dimension)
        return self.then(_require_dimension(transform, self.dimension))

    def inverse(self) -> 'AffineTransformation':
        """Return the inverse affine transformation."""
        try:
            inverse = np.linalg.inv(self._matrix.to_numpy(dtype=float))
        except np.linalg.LinAlgError as exc:
            raise ValueError("This affine transformation is not invertible") from exc
        return AffineTransformation(Matrix(np.real_if_close(inverse).tolist()))

    def apply(self, value):
        """Transform a point, vector, point collection, curve, or coordinate data."""
        from kiwicalc.geometry.curves import Curve2D, Curve3D
        from kiwicalc.geometry.point_collections import PointCollection
        from kiwicalc.geometry.points import Point
        from kiwicalc.geometry.vectors import Vector

        if isinstance(value, Point):
            coordinates = self._apply_point(value.coordinates)
            return _point_like(value, coordinates)
        if isinstance(value, Vector):
            start = self._apply_point(value.start_coordinate)
            end = self._apply_point(value.end_coordinate)
            return _vector_like(value, start, end)
        if isinstance(value, PointCollection):
            points = [self.apply(point) for point in value.points]
            return type(value)(points)
        if isinstance(value, (Curve2D, Curve3D)):
            if value.dimensions != self.dimension:
                raise ValueError(f"Cannot apply a {self.dimension}D transformation to a {value.dimensions}D curve")
            return value.transform(self._matrix.to_numpy(dtype=float))
        if isinstance(value, np.ndarray):
            return self._apply_array(value)
        if isinstance(value, (str, bytes)):
            raise TypeError("apply() expects geometry or numeric coordinates")
        try:
            items = list(value)
        except TypeError as exc:
            raise TypeError("apply() expects geometry or numeric coordinates") from exc
        if len(items) == self.dimension and all(isinstance(item, Number) for item in items):
            return tuple(self._apply_point(items).tolist())
        return [tuple(row) for row in self._apply_array(np.asarray(items)).tolist()]

    __call__ = apply

    def __matmul__(self, other):
        """Compose transforms or apply this transform to geometry."""
        if isinstance(other, AffineTransformation):
            if self.dimension != other.dimension:
                raise ValueError("Cannot compose transformations with different dimensions")
            return AffineTransformation(self._matrix @ other._matrix)
        return self.apply(other)

    def _apply_point(self, coordinates):
        point = _coordinates(coordinates, self.dimension, "coordinates")
        homogeneous = np.append(point, 1)
        return self._matrix.to_numpy(dtype=float).dot(homogeneous)[:self.dimension]

    def _apply_array(self, values):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("coordinates must contain numeric values") from exc
        if array.ndim == 1:
            return self._apply_point(array)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(f"coordinate data must have shape ({self.dimension},) or (n, {self.dimension})")
        if np.any(~np.isfinite(array)):
            raise ValueError("coordinates must contain finite values")
        ones = np.ones((len(array), 1))
        return (self._matrix.to_numpy(dtype=float) @ np.hstack((array, ones)).T).T[:, :self.dimension]


def _dimension(value):
    if isinstance(value, bool) or value not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    return int(value)


def _unpack_values(values, name):
    if len(values) == 1 and not isinstance(values[0], (Number, str, bytes)):
        try:
            values = tuple(values[0])
        except TypeError as exc:
            raise TypeError(f"{name} values must be numeric") from exc
    if not values:
        raise ValueError(f"{name} requires coordinate values")
    return tuple(values)


def _finite_real(values, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _coordinates(values, dimension, name):
    values = getattr(values, "coordinates", values)
    array = _finite_real(values, name)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must contain exactly {dimension} values")
    return array


def _angle(value, degrees):
    if isinstance(value, bool) or not isinstance(value, Number):
        raise TypeError("angle must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("angle must be finite")
    return result * pi / 180 if degrees else result


def _axis(value):
    if isinstance(value, str):
        axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        normalized = value.lower()
        if normalized not in axes:
            raise ValueError("3D rotation axis must be x, y, z, or a three-value vector")
        value = axes[normalized]
    axis = _coordinates(value, 3, "axis")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis cannot be zero")
    return axis / norm


def _require_dimension(transform, dimension):
    if transform.dimension != dimension:
        raise ValueError(f"Expected a {dimension}D transformation")
    return transform


def _point_like(source, coordinates):
    from kiwicalc.geometry.points import Point, Point2D, Point3D
    if type(source) is Point2D:
        return Point2D(*coordinates)
    if type(source) is Point3D:
        return Point3D(*coordinates)
    return Point(coordinates)


def _vector_like(source, start, end):
    from kiwicalc.geometry.vectors import Vector, Vector2D, Vector3D
    direction = np.asarray(end) - np.asarray(start)
    if type(source) is Vector2D:
        return Vector2D(*direction, start_coordinate=start)
    if type(source) is Vector3D:
        return Vector3D(*direction, start_coordinate=start)
    return Vector(direction_vector=direction, start_coordinate=start)


__all__ = ["AffineTransformation"]
