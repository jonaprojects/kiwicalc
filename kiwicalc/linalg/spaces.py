from __future__ import annotations
import numpy as np
import warnings
from contextlib import contextmanager
from math import sqrt
from typing import Union, Tuple, List, Optional, Any, Callable, Iterable
from kiwicalc.core.interfaces import IPlottable3D, IScatterable3D
from kiwicalc.core.utils import to_lambda, decimal_range
from kiwicalc.geometry.points import Point
from kiwicalc.geometry.vectors import Vector
from kiwicalc.equations.single import LinearEquation
from kiwicalc.parsing.parse_expression import surface_from_str
from kiwicalc.geometry.surfaces import Surface3D

class Surface(Surface3D):
    """
    represents a surface of the equation ax+by+cz+d = 0, where (a,b,c) is the perpendicular of the surface, and d
    is a free number.
    """

    def __init__(self, coefs, resolution=80):
        super().__init__(resolution=resolution)
        if isinstance(coefs, str):
            self.__a, self.__b, self.__c, self.__d = surface_from_str(coefs, get_coefficients=True)
        elif isinstance(coefs, Iterable):
            coefficients = [coef for coef in coefs]
            if len(coefficients) == 4:
                self.__a, self.__b, self.__c, self.__d = (coefficients[0], coefficients[1], coefficients[2], coefficients[3])
            elif len(coefficients) == 3:
                self.__a, self.__b, self.__c, self.__d = (coefficients[0], coefficients[1], coefficients[2], 0)
            else:
                raise ValueError(f'Invalid number of coefficients in coefficients of surface. Got {len(coefficients)}, expected 4 or 3')

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def d(self):
        return self.__d

    def intersection(self, vector: Vector, get_point=False):
        """
        Finds the intersection between a surface and a vector

        :param get_point: If set to True, a point that represents the intersection will be returned instead of a list that represents the coordinates of the intersection. Default value is false.

        :param vector: An object of type Vector.
        :return: Returns a list of the coordinates of the intersection. If get_point = True, returns corresponding
        point object.
        """
        general_point = vector.general_point('t')
        expression = (
            self.__a * general_point[0]
            + self.__b * general_point[1]
            + self.__c * general_point[2]
            + self.__d
        )
        t_solution = LinearEquation(f'{expression} = 0', variables=('t',), calc_now=True).solution
        for polynomial in general_point:
            polynomial.assign(t=t_solution)
        if get_point:
            return Point((polynomial.expressions[0].coefficient for polynomial in general_point))
        return [polynomial.expressions[0].coefficient for polynomial in general_point]

    def __str__(self) -> str:
        """Getting the string representation of the algebraic formula of the surface. ax + by + cz + d = 0"""
        accumulator = f'{self.__a}'
        return accumulator + ''.join((f'+{val}{var}' if val > 0 else f'-{val}{var}' for val, var in zip((self.__b, self.__c, self.__d), ('x', 'y', 'z', '')) if val != 0))

    def __repr__(self):
        return f'Surface("{self.__str__()}")'

    def to_lambda(self):
        if self.__c == 0:
            warnings.warn('c = 0 might lead to unexpected behaviors in this version.')
            return lambda x, y: 0
        return lambda x, y: (-self.__a * x - self.__b * y - self.__d) / self.__c

    def _sample_plane(self, start=-3, stop=3, resolution=None):
        counts = self.resolution if resolution is None else ((resolution, resolution) if isinstance(resolution, int) else resolution)
        normal = np.asarray((self.__a, self.__b, self.__c), dtype=float)
        norm_squared = float(np.dot(normal, normal))
        if norm_squared == 0:
            raise ValueError('A plane must have at least one non-zero normal coefficient')
        origin = -float(self.__d) * normal / norm_squared
        reference = np.asarray((1.0, 0.0, 0.0)) if abs(normal[0]) < abs(normal[1]) else np.asarray((0.0, 1.0, 0.0))
        first_basis = np.cross(normal, reference)
        first_basis /= np.linalg.norm(first_basis)
        second_basis = np.cross(normal, first_basis)
        second_basis /= np.linalg.norm(second_basis)
        u = np.linspace(start, stop, counts[0])
        v = np.linspace(start, stop, counts[1])
        U, V = np.meshgrid(u, v)
        points = origin[:, None, None] + first_basis[:, None, None] * U + second_basis[:, None, None] * V
        return points[0], points[1], points[2]

    def sample(self, resolution=None):
        return self._sample_plane(resolution=resolution)

    def plot(self, start: float=-3, stop: float=3, step: float=0.3, xlabel: str='X Values', ylabel: str='Y Values', zlabel: str='Z Values', show=True, fig=None, ax=None, write_labels=True, meshgrid=None, wireframe=False, **style):
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        if meshgrid is not None and self.__c != 0:
            X, Y = meshgrid
            Z = (-self.__a * X - self.__b * Y - self.__d) / self.__c
        else:
            count = max(2, int(np.ceil((stop - start) / step)))
            X, Y, Z = self._sample_plane(start, stop, count)
        artist = ax.plot_wireframe(X, Y, Z, **style) if wireframe else ax.plot_surface(X, Y, Z, **style)
        if write_labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if show:
            plt.show()
        return artist

    def plot3d(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def __eq__(self, other):
        """Equating between surfaces. Surfaces are equal if they have the same a,b,c,d coefficients """
        if other is None:
            return False
        if isinstance(other, Surface):
            return (self.__a, self.__b, self.__c, self.__d) == (other.__a, other.__b, other.__c, other.__d)
        if isinstance(other, list):
            return [self.__a, self.__b, self.__c, self.__d] == other
        if isinstance(other, tuple):
            return (self.__a, self.__b, self.__c, self.__d) == other
        if isinstance(other, set):
            return {self.__a, self.__b, self.__c, self.__d} == other
        raise TypeError(f"Invalid type '{type(other)}' for checking equality with object of instance of class Surface.Expected types 'Surface', 'list', 'tuple', 'set'. ")

    def __ne__(self, other):
        return not self.__eq__(other)

def mav(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ Mean absolute value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += abs(func1(value) - func2(value))
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError('Cannot process 0 points')
    return my_sum / num_of_points

def msv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """mean square value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError('Cannot process 0 points')
    return my_sum / num_of_points

def mrv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ mean root value """
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError('Cannot process 0 points')
    return sqrt(my_sum / num_of_points)

@contextmanager
def copy(expression):
    try:
        copy_method = getattr(expression, '__copy__', None)
        if callable(copy_method):
            copy_of_expression = expression.__copy__()
            yield copy_of_expression
        else:
            copy_method = getattr(expression, 'copy', None)
            if callable(copy_method):
                copy_of_expression = expression.copy()
                yield copy_of_expression
    finally:
        del copy_of_expression
