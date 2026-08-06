from __future__ import annotations
import math
from math import sqrt, pi
import cmath
import warnings
from functools import reduce
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.interfaces import IPlottable, IScatterable, IExpression
from kiwicalc.core.utils import (
    is_number, round_decimal, _format_minus, decimal_range,
    format_coefficient, format_free_number
)
from kiwicalc.expressions.poly import Poly
from kiwicalc.expressions.mono import Mono
from kiwicalc.expressions.sum import ExpressionSum
from kiwicalc.expressions.var import Var
from kiwicalc.expressions.roots import Sqrt
from kiwicalc.expressions.special import Abs

class Point:

    def __init__(self, coordinates: Union[Iterable, int, float]):
        if isinstance(coordinates, Iterable):
            self._coordinates = [coordinate for coordinate in coordinates]
            for index, coordinate in enumerate(self._coordinates):
                if isinstance(coordinate, IExpression):
                    self._coordinates[index] = coordinate.__copy__()
        elif isinstance(coordinates, (int, float)):
            self._coordinates = [coordinates]
        else:
            raise TypeError(f'Invalid type {type(coordinates)} for creating a new Point object')

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Iterable):
        self._coordinates = coordinates

    @property
    def dimensions(self):
        return len(self._coordinates)

    def plot(self):
        self.scatter()

    def scatter(self, show=True):
        if len(self._coordinates) == 1:
            plt.scatter(self._coordinates[0], 0)
        if len(self._coordinates) == 2:
            plt.scatter(self._coordinates[0], self._coordinates[1])
        elif len(self._coordinates) == 3:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter2d(self._coordinates[0], self._coordinates[1], self._coordinates[2])
        if show:
            plt.show()

    def __iadd__(self, other: 'Union[Iterable, Point]'):
        if isinstance(other, Iterable):
            self._coordinates = [coord1 + coord2 for coord1, coord2 in zip(self._coordinates, other)]
            return self
        elif isinstance(other, Point):
            self._coordinates = [coord1 + coord2 for coord1, coord2 in zip(self._coordinates, other._coordinates)]
            return self
        else:
            raise TypeError(f'Encountered unexpected type {type(other)} while attempting to add points. Expected typesIterable or Point')

    def __isub__(self, other: 'Union[Iterable, Point]'):
        if isinstance(other, Point):
            self._coordinates = [coord1 - coord2 for coord1, coord2 in zip(self._coordinates, other._coordinates)]
            return self
        elif isinstance(other, Iterable):
            self._coordinates = [coord1 - coord2 for coord1, coord2 in zip(self._coordinates, other)]
            return self
        else:
            raise TypeError(f'Encountered unexpected type {type(other)} while attempting to subtract points. Expectedtypes Iterable or Point')

    def __add__(self, other: 'Union[Iterable, Point]'):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: 'Union[Iterable, Point]'):
        if isinstance(other, Iterable):
            other = Point(Iterable)
        if isinstance(other, Point):
            return other.__add__(self)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: 'Union[Point,PointCollection]'):
        if isinstance(other, Iterable):
            other = Point(Iterable)
        if isinstance(other, Point):
            return other.__sub__(self)
        else:
            raise NotImplementedError

    def __imul__(self, other: 'Union[int, float, Point, PointCollection, IExpression]'):
        if isinstance(other, (int, float, IExpression)):
            if isinstance(other, IExpression):
                other_evaluation = other.try_evaluate()
                if other_evaluation is not None:
                    other = other_evaluation
            for index in range(len(self._coordinates)):
                self._coordinates[index] *= other
                return self
        elif isinstance(other, Point):
            return reduce(lambda tuple1, tuple2: tuple1[0] * tuple2[0] + tuple1[1] * tuple2[1], zip(self._coordinates, other._coordinates))
        elif isinstance(other, PointCollection):
            raise NotImplementedError("This feature isn't implemented yet in this version")

    def __mul__(self, other: 'Union[int, float, Point, PointCollection, IExpression]'):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: 'Union[int, float, Point, PointCollection, IExpression]'):
        return self.__copy__().__imul__(other)

    def coord_at(self, index: int):
        return self._coordinates[index]

    def max_coord(self):
        return max(self._coordinates)

    def min_coord(self):
        return min(self._coordinates)

    def sum(self):
        return sum(self._coordinates)

    def distance(self, other_point: 'Point'):
        if len(self.coordinates) != len(other_point.coordinates):
            raise ValueError(f'Cannot calculate distance between points in different dimensions.')
        return sqrt(sum(((coord1 - coord2) ** 2 for coord1, coord2 in zip(self.coordinates, other_point.coordinates))))

    def __eq__(self, other: 'Union[Point, PointCollection]'):
        if other is None:
            return False
        if isinstance(other, PointCollection):
            if len(other.points) != 1:
                return False
            other = other.points[0]
        if isinstance(other, Point):
            return self._coordinates == other._coordinates
        else:
            raise TypeError(f'Invalid type {type(other)} for comparing with a Point object')

    def __ne__(self, other: 'Union[Point, PointCollection]'):
        return not self.__eq__(other)

    def __neg__(self):
        return Point(coordinates=[-coordinate for coordinate in self._coordinates])

    def __repr__(self):
        return f'Point({self._coordinates})'

    def __str__(self):
        if all((isinstance(coordinate, (int, float)) for coordinate in self._coordinates)):
            return f'({','.join((str(round(coordinate, 3)) for coordinate in self._coordinates))})'
        return f'({','.join((coordinate.__str__() for coordinate in self._coordinates))})'

    def __copy__(self):
        return Point(self._coordinates)

    def __len__(self):
        return len(self._coordinates)

class Point1D(Point, IPlottable):

    def __init__(self, x: Union[int, float, IExpression]):
        super(Point1D, self).__init__((x,))

    @property
    def x(self):
        return self._coordinates[0]

class Point2D(Point, IPlottable):

    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression]):
        super(Point2D, self).__init__((x, y))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

class Point3D(Point):

    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression], z: Union[int, float, IExpression]):
        super(Point3D, self).__init__((x, y, z))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]

class Point4D(Point):

    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression], z: Union[int, float, IExpression], c: Union[int, float, IExpression]):
        super(Point4D, self).__init__((x, y, z, c))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]

    @property
    def c(self):
        return self._coordinates[3]

class Line2D(IPlottable):

    def __init__(self, point1: Union[Point2D, Iterable], point2: Union[Point2D, Iterable], gen_copies=True):
        if isinstance(point1, Point2D):
            self._point1 = point1.__copy__() if gen_copies else point1
        elif isinstance(point1, Iterable):
            x, y = point1
            self._point1 = Point2D(x, y)
        else:
            raise TypeError(f"Invalid type for param 'point1' when creating a Line object.")
        if isinstance(point2, Point2D):
            self._point2 = point2.__copy__() if gen_copies else point2
        elif isinstance(point2, Iterable):
            x, y = point2
            self._point2 = Point2D(x, y)
        else:
            raise TypeError(f"Invalid type for param 'point2' when creating a Line object.")

    def middle(self):
        return Point2D((self._point1.x + self._point2) / 2, (self._point1.y + self._point2.y) / 2)

    def length(self):
        inside_root = (self._point1.x - self._point2.x) ** 2 + (self._point1.y - self._point2.y) ** 2
        if isinstance(inside_root, (int, float)):
            return sqrt(inside_root)

    @property
    def slope(self):
        x1, x2 = (self._point1.x, self._point2.x)
        y1, y2 = (self._point1.y, self._point2.y)
        numerator, denominator = (y2 - y1, x2 - x1)
        if denominator is None:
            warnings.warn("There's no slope for a single x value with two y values.")
            return None
        return numerator / denominator

    @property
    def free_number(self):
        m = self.slope
        if m is None:
            warnings.warn("There's no free number for a single x value with two y values.")
            return None
        return self._point1.y - self._point1.x * m

    def equation(self):
        m = self.slope
        if m is None:
            warnings.warn("There's no slope for a single x value with two y values.")
            return None
        b = self._point1.y - self._point1.x * m
        m_str = format_coefficient(m)
        b_str = format_free_number(b)
        return f'{m_str}x{b_str}'

    def to_lambda(self):
        m = self.slope
        if m is None:
            warnings.warn('Cannot generate a lambda expression for a single x value with two y values.')
            return None
        b = self._point1.y - self._point1.x * m
        return lambda x: m * x + b

    def intersection(self):
        pass

    def plot(self, start: float=-6, stop: float=6, step: float=0.3, ymin: float=-10, ymax: float=10, title: str=None, formatText: bool=False, show_axis: bool=True, show: bool=True, fig=None, ax=None, values=None):
        from kiwicalc.plotting.plots import plot_function, scatter_function
        my_lambda = self.to_lambda()
        if my_lambda is None:
            pass
        plot_function(my_lambda, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title, show_axis=show_axis, show=show, fig=fig, formatText=formatText, ax=ax, values=values)

    def scatter(self, start: float=-10, stop: float=10, step: float=0.05, ymin: float=-10, ymax: float=10, title=None, show_axis=True, show=True, fig=None, ax=None, formatText=True, values=None):
        lambda_expression = self.to_lambda()
        if not lambda_expression:
            pass
        if title is None:
            title = self.__str__()
        scatter_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title, show_axis=show_axis, show=show, fig=fig, ax=ax, values=values)

class Circle(IPlottable):

    def __init__(self, radius: Union[float, int, IExpression], center: Union[Iterable[Union[int, float, IExpression]], Point]=(0, 0), gen_copies=False):
        if isinstance(radius, (int, float)):
            self._radius = Mono(radius)
        elif isinstance(radius, IExpression):
            if gen_copies:
                self._radius = radius.__copy__()
            else:
                self._radius = radius
        else:
            raise TypeError(f'Invalid type {type(radius)} for radius when creating a Circle object')
        if isinstance(center, Iterable) and (not isinstance(center, Point)):
            center_list = [coordinate for coordinate in center]
            if any((not isinstance(coordinate, (IExpression, int, float)) for coordinate in center_list)):
                raise TypeError(f'Invalid types of coordinates when creating a Circle object')
            for index, coordinate in enumerate(center_list):
                if isinstance(coordinate, (int, float)):
                    center_list[index] = Mono(coordinate)
            center = Point(center_list)
        if isinstance(center, Point):
            if center.dimensions != 2:
                raise ValueError(f'Circle object can only contain a 2D Point as a center ( Got {center.dimensions}D')
            self._center = center.__copy__() if gen_copies else center
        else:
            raise TypeError(f'Invalid type {type(center)} for the center point when creating a Circle object')

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def center(self) -> Point:
        return self._center

    @property
    def left_edge(self):
        return Point((-self._radius + self.center_x, self.center_y))

    @property
    def right_edge(self):
        return Point((self._radius + self.center_x, self.center_y))

    @property
    def top_edge(self):
        return Point((self.center_x, self._radius + self.center_y))

    @property
    def bottom_edge(self):
        return Point((self.center_x, -self._radius + self.center_y))

    @property
    def center_x(self):
        return self._center.coordinates[0]

    @property
    def center_y(self):
        return self._center.coordinates[1]

    def area(self):
        result = self._radius ** 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def perimeter(self):
        result = self._radius * 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def point_inside(self, point: Union[Point, Iterable], already_evaluated: Tuple[float, float, float]=None) -> bool:
        """
        Checks whether a 2D point is inside the circle

        :param point: the point
        :param already_evaluated: Evaluations of the radius and center point of the circle as floats.
        :return: Returns True if the point is indeed inside the circle or touches it from the inside, otherwise False.
        """
        if isinstance(point, Point):
            x, y = (point.coordinates[0], point.coordinates[1])
        elif isinstance(point, Iterable):
            coordinates = [coord for coord in point]
            if len(coordinates) != 2:
                raise ValueError('Can only accept points with 2 dimensions')
            x, y = (coordinates[0], coordinates[1])
        else:
            raise ValueError(f'Invalid type {type(point)} for this method.')
        if already_evaluated is not None:
            radius_eval, center_x_eval, center_y_eval = already_evaluated
        else:
            radius_eval = self._radius.try_evaluate()
            center_x_eval = self.center_x.try_evaluate()
            center_y_eval = self.center_y.try_evaluate()
        if None not in (radius_eval, center_x_eval, center_y_eval):
            if x > center_x_eval + radius_eval:
                return False
            if x < center_x_eval - radius_eval:
                return False
            if y > center_y_eval + radius_eval:
                return False
            if y < center_y_eval - radius_eval:
                return False
            return True
        else:
            raise ValueError('This feature is only supported for Circles without any additional parameters')

    def is_inside(self, other_circle: 'Circle') -> bool:
        if not isinstance(other_circle, Circle):
            raise TypeError(f"Invalid type '{type(other_circle)}'. Expected type 'circle'. ")
        my_radius_eval = self._radius.try_evaluate()
        my_center_x_eval = self.center_x.try_evaluate()
        my_center_y_eval = self.center_y.try_evaluate()
        other_radius_eval = other_circle._radius.try_evaluate()
        other_center_x_eval = other_circle.center_x.try_evaluate()
        other_center_y_eval = other_circle.center_y.try_evaluate()
        if None not in (my_radius_eval, my_center_x_eval, my_center_y_eval, other_radius_eval, other_center_x_eval, other_center_y_eval):
            if not other_circle.point_inside(self.top_edge, already_evaluated=(other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.bottom_edge, already_evaluated=(other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.right_edge, already_evaluated=(other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.left_edge, already_evaluated=(other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            return True
        else:
            raise ValueError("Can't determine whether a circle is inside another, when one or more of them are expressed via parameters")

    def plot(self, fig=None, ax=None):
        radius_eval = self._radius.try_evaluate()
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        if None in (radius_eval, center_x_eval, center_y_eval):
            raise ValueError('Can only plot circles with real numbers (and not algebraic expressions)')
        circle1 = plt.Circle((center_x_eval, center_y_eval), radius_eval, color='r', fill=False)
        if None in (fig, ax):
            fig, ax = plt.subplots()
        ax.add_patch(circle1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()
        plt.show()

    def to_lambda(self):
        warnings.warn('This is an experimental feature!')
        radius_evaluation = self._radius.try_evaluate()
        center_x_evaluation = self.center_x.try_evaluate()
        center_y_evaluation = self.center_y.try_evaluate()
        if None not in (radius_evaluation, center_x_evaluation, center_y_evaluation):
            return lambda x: (sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation, -sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation)
        return lambda x: (Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y, -Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y)

    @property
    def equation(self) -> str:
        x_part = _format_minus('x', self.center_x)
        if self.center_y == 0:
            y_part = 'y^2'
        elif '+' in self.center_y.__str__() or '-' in self.center_y.__str__():
            y_part = f'(y-({self.center_y}))^2'
        else:
            y_part = f'(y-{self.center_y})^2'
        radius_eval = self._radius.try_evaluate()
        radius_part = f'{self._radius}^2' if radius_eval is None or (radius_eval is not None and radius_eval > 100) else f'{radius_eval ** 2}'
        return f'{x_part} + {y_part} = {radius_part}'

    def x_intersection(self):
        pass

    def _expression(self):
        x = Var('x')
        y = Var('y')
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 - self._radius ** 2

    def intersection(self, other):
        from kiwicalc.equations.system import solve_poly_system
        if isinstance(other, Circle):
            if self.has_parameters() or other.has_parameters():
                raise ValueError("This feature hasn't been implemented yet for Circle equations with additionalparameters")
            else:
                initial_x = (self.center_x + other.center_x) / 2
                initial_y = (self.center_y + other.center_y) / 2
                intersections = solve_poly_system([self._expression(), other._expression()], initial_vals={'x': initial_x, 'y': initial_y})
                return intersections

    def has_parameters(self) -> bool:
        coefficient_eval = self._radius.try_evaluate()
        if coefficient_eval is None:
            return True
        center_x_eval = self.center_x.try_evaluate()
        if center_x_eval is None:
            return True
        center_y_eval = self.center_y.try_evaluate()
        if center_y_eval is None:
            return True
        return False

    def y_intersection(self, get_complex=False):
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        radius_eval = self._radius.try_evaluate()
        if None not in (center_x_eval, radius_eval):
            if abs(center_x_eval) > abs(radius_eval):
                if get_complex:
                    warnings.warn('Solving the intersections with complex numbers is still experimental...The issue will be resolved in later versions. Sorry!')
                    val = cmath.sqrt(radius_eval ** 2 - center_x_eval ** 2)
                    if center_y_eval is not None:
                        y1, y2 = (val + center_y_eval, -val + center_y_eval)
                    else:
                        y1, y2 = (val + self.center_y, -val + self.center_y)
                    return (Point((0, y1)), Point((0, y2)))
                return None
            else:
                val = sqrt(radius_eval ** 2 - center_x_eval ** 2)
                if val == 0:
                    if center_y_eval is not None:
                        return Point((0, center_y_eval))
                    else:
                        return Point((0, self.center_y))
                else:
                    if center_y_eval is not None:
                        y1, y2 = (val + center_y_eval, -val + center_y_eval)
                    else:
                        y1, y2 = (val + self.center_y, -val + self.center_y)
                    return (Point((0, y1)), Point((0, y2)))
        else:
            my_root = f'sqrt({_format_minus(self._radius, 0)} - {_format_minus(self.center_x, 0)})'

    def assign(self, **kwargs):
        self._radius.assign(**kwargs)
        self._center.coordinates[0].assign(**kwargs)
        self._center.coordinates[1].assign(**kwargs)

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    def __copy__(self):
        return Circle(radius=self._radius.__copy__(), center=self.center.__copy__())

    def __call__(self, x: Union[int, float, IExpression], **kwargs):
        pass

    def __repr__(self):
        return f'Circle(radius={self._radius}, center={self._center})'

    def __str__(self):
        return f'Circle(radius={self._radius}, center={self._center})'

def process_to_points(func: Union[Callable, str], start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, values=None):
    from kiwicalc.functions.function import Function
    if isinstance(func, str):
        func = Function(func)
    if values is None:
        values = list(decimal_range(start, stop, step)) if values is None else values
    results = []
    for index, value in enumerate(values):
        try:
            current_result = func(value)
            results.append(current_result)
        except ValueError:
            results.append(None)
    return (values, results)

from kiwicalc.geometry.point_collections import PointCollection
