from __future__ import annotations
import math
from math import sqrt
import warnings
import operator
import random
from functools import reduce
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.interfaces import IPlottable, IScatterable, IExpression
from kiwicalc.core.utils import round_decimal, copy_expression
from kiwicalc.expressions.var import Var
from kiwicalc.geometry.points import Point, Point2D, Point3D

class Vector:

    def __init__(self, direction_vector=None, start_coordinate=None, end_coordinate=None):
        """
        Creates a new Vector object.


        :param direction_vector: For example, the direction vector of a vector that starts from ( 1,1,1 ) and ends with (4,4,4) is (3,3,3)

        :param start_coordinate: The coordinate that represents the origin of the vector on an axis system.
        :param end_coordinate: The coordinate that represents the end of the vector on an axis system.
        """
        if direction_vector is None and (start_coordinate is None or end_coordinate is None):
            raise ValueError('A vector requires a direction, or both a start and end coordinate.')
        if start_coordinate is not None and end_coordinate is not None:
            if len(start_coordinate) != len(end_coordinate):
                raise ValueError('Cannot handle with vectors with different dimensions in this version.')
            try:
                self._start_coordinate = list(start_coordinate)
            except TypeError:
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list.expected types were tuple,list, set, and dict.")
            try:
                self._end_coordinate = list(end_coordinate)
            except TypeError:
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list.expected types were tuple,list, set, and dict.")
            self._direction_vector = list([end_coordinate[i] - start_coordinate[i] for i in range(len(start_coordinate))])
        elif direction_vector is not None and start_coordinate is not None:
            if len(direction_vector) != len(start_coordinate):
                raise ValueError('Direction and start coordinate must have the same dimensions.')
            self._start_coordinate = list(start_coordinate)
            self._direction_vector = list(direction_vector)
            self._end_coordinate = [self._start_coordinate[i] + self._direction_vector[i] for i in range(len(self._start_coordinate))]
        elif direction_vector is not None and end_coordinate is not None:
            if len(direction_vector) != len(end_coordinate):
                raise ValueError('Direction and end coordinate must have the same dimensions.')
            self._end_coordinate = list(end_coordinate)
            self._direction_vector = list(direction_vector)
            self._start_coordinate = [self._end_coordinate[i] - self._direction_vector[i] for i in range(len(self._end_coordinate))]
        elif direction_vector is not None:
            self._direction_vector = list(direction_vector)
            self._end_coordinate = self._direction_vector.copy()
            self._start_coordinate = [0 for _ in range(len(self._end_coordinate))]

    @property
    def start_coordinate(self):
        return self._start_coordinate

    @property
    def end_coordinate(self):
        return self._end_coordinate

    @property
    def direction(self):
        return self._direction_vector

    def plot(self, show=True, arrow_length_ratio: float=0.05, fig=None, ax=None):
        from kiwicalc.plotting.plots import plot_vector_2d, plot_vector_3d
        start_length, end_length = (len(self._start_coordinate), len(self._end_coordinate))
        if start_length == end_length == 2:
            plot_vector_2d(self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0], self._direction_vector[1], show=show, fig=fig, ax=ax)
        elif start_length == end_length == 3:
            u, v, w = (self._direction_vector[0], self._direction_vector[1], self._direction_vector[2])
            start_x, start_y, start_z = (self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[2])
            plot_vector_3d((start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig, ax=ax)
        else:
            raise ValueError(f'Cannot plot a vector with {start_length} dimensions. (Only 2D and 3D plotting is supported')

    def length(self):
        return round_decimal(sqrt(sum(item ** 2 for item in self._direction_vector)))

    def multiply(self, other: 'Union[int, float, IExpression, Iterable, Vector, VectorCollection, ]'):
        if isinstance(other, (Vector, Iterable)) and (not isinstance(other, (VectorCollection, IExpression))):
            return self.scalar_product(other)
        elif isinstance(other, (int, float, IExpression)):
            return self.multiply_all(other)
        else:
            raise TypeError(f'Vector.multiply(): expected types Vector/tuple/list/int/float but got {type(other)}')

    def multiply_all(self, number: Union[int, float, IExpression]):
        """ Multiplies the vector by the given expression, and returns the current vector ( Which was not copied ) """
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] *= number
        self.__update_end()
        return self

    def scalar_product(self, other: Iterable):
        """

        :param other: other vector
        :return: returns the scalar multiplication of two vectors
        :raise: raises an Exception when the type of other isn't tuple or Vector
        """
        if isinstance(other, Iterable):
            other = Vector(other)
        if isinstance(other, Vector):
            if len(self._direction_vector) != len(other._direction_vector):
                raise ValueError('Cannot calculate a scalar product for vectors with different dimensions.')
            scalar_result = 0
            for a, b in zip(self._direction_vector, other._direction_vector):
                scalar_result += a * b
            return scalar_result
        else:
            raise TypeError(f'Vector.scalar_product(): expected type Vector or tuple, but got {type(other)}')

    def equal_direction_ratio(self, other):
        """

        :param other: another vector
        :return: True if the two vectors have the same ratio of directions, else False
        """
        try:
            if len(self._direction_vector) != len(other._direction_vector):
                return False
            if len(self._direction_vector) == 0:
                return False
            elif len(self._direction_vector) == 1:
                return self._direction_vector[0] == other._direction_vector[0]
            else:
                ratios = []
                for a, b in zip(self._direction_vector, other._direction_vector):
                    if a == 0 and b != 0 or (b == 0 and a != 0):
                        return False
                    if not a == b == 0:
                        ratios.append(a / b)
                if ratios:
                    return all((x == ratios[0] for x in ratios))
                return True
        except ZeroDivisionError:
            warnings.warn("Cannot check whether the vectors' directions are equal because of a ZeroDivisionError")

    @classmethod
    def random_vector(cls, numbers_range: Tuple[int, int], num_of_dimensions: int=None):
        """
        Generate a random vector object.

        :param numbers_range: the range of possible values
        :param num_of_dimensions: the number of dimensions of the vector. If not set, a number will be chosen
        :return: Returns a Vector object, (or Vector2D or Vector3D objects).
        """
        if cls is Vector2D:
            num_of_dimensions = 2
        elif cls is Vector3D:
            num_of_dimensions = 3
        if num_of_dimensions is None:
            num_of_dimensions = random.randint(2, 9)
        direction = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]
        start = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]
        if cls in (Vector2D, Vector3D):
            return cls(*direction, start_coordinate=start)
        return cls(direction_vector=direction, start_coordinate=start)

    def general_point(self, var_name: str='x'):
        """
        Generate an algebraic expression that represents any dot on the vector

        :param var_name: the name of the variable (str)
        :return: returns a list of algebraic expressions
        """
        variable = Var(var_name)
        lst = [start + variable * direction for start, direction in zip(self._start_coordinate, self._direction_vector)]
        return lst

    def intersection(self, other: 'Union[Vector, VectorCollection, Surface]', get_points=False):
        from kiwicalc.equations.system import LinearSystem
        from kiwicalc.linalg.spaces import Surface

        if isinstance(other, Vector):
            if self._direction_vector == other._direction_vector:
                print('The vectors have the same directions, unhandled case for now')
                return
            my_general, other_general = (self.general_point('t'), other.general_point('s'))
            try:
                solutions_dict = LinearSystem(
                    (f'{expr1}={expr2}' for expr1, expr2 in zip(my_general, other_general))
                ).get_solutions()
            except ValueError:
                return None
            t, s = (solutions_dict['t'], solutions_dict['s'])
            for expression in my_general:
                expression.assign(t=t)
            coordinates = [expression.try_evaluate() for expression in my_general]
            if get_points:
                return Point(coordinates)
            return coordinates
        elif isinstance(other, VectorCollection):
            return any((self.intersection(other_vector) for other_vector in other.vectors))
        elif isinstance(other, Surface):
            return other.intersection(self)
        else:
            raise TypeError(f'Invalid type {type(other)} for searching intersections with a vector. Expected types: Vector, VectorCollection, Surface.')

    def equal_lengths(self, other):
        """

        :param other: another vector
        :return: True if self and other have the same lengths, else otherwise
        """
        return self.length() == other.length()

    @staticmethod
    def fill(dimension: int, value) -> 'Vector':
        return Vector(direction_vector=[value for _ in range(dimension)])

    @staticmethod
    def fill_zeros(dimension: int) -> 'Vector':
        return Vector.fill(dimension, 0)

    @staticmethod
    def fill_ones(dimension: int) -> 'Vector':
        return Vector.fill(dimension, 1)

    def __copy__(self):
        return Vector(start_coordinate=self._start_coordinate, end_coordinate=self._end_coordinate)

    def __eq__(self, other: 'Union[Vector, VectorCollection]'):
        """ Returns whether the two vectors have the same starting position, length, and ending position."""
        if other is None:
            return False
        if isinstance(other, Vector):
            return self._direction_vector == other._direction_vector
        elif isinstance(other, VectorCollection):
            if other.num_of_vectors != 1:
                return False
            return self == other.vectors[0]
        else:
            raise TypeError(f'Invalid type {type(other)} for equating vectors.')

    def __ne__(self, other):
        return not self.__eq__(other)

    def __update_end(self):
        self._end_coordinate = [start + direction for start, direction in zip(self._start_coordinate, self._direction_vector)]

    def __imul__(self, other: 'Union[IExpression, int, float, Vector, VectorCollection, Surface]'):
        return self.multiply(other)

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def power_by_vector(self, other: 'Union[Iterable, Vector]'):
        from kiwicalc.linalg.matrix import Matrix

        if not isinstance(other, (Iterable, Vector)):
            raise TypeError(f"Invalid type '{type(other)} to raise a vector by another vector ( vector1 ** vector2 )'")
        if isinstance(other, Iterator):
            other = list(other)
        other_items = other._direction_vector if isinstance(other, Vector) else other
        return Matrix(matrix=[[my_item ** other_item for other_item in other_items] for my_item in self._direction_vector])

    def power_by_expression(self, expression: Union[int, float, IExpression]):
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] **= expression
        self.__update_end()
        return self

    def power_by(self, other: 'Union[int, float, IExpression, Iterable, Vector, VectorCollection]'):
        return self.__ipow__(other)

    def __ipow__(self, other: 'Union[int, float, IExpression, Iterable, Vector]'):
        if isinstance(other, (int, float, IExpression)):
            return self.power_by_expression(other)
        elif isinstance(other, (Vector, Iterable)) and (not isinstance(other, (IExpression, VectorCollection))):
            return self.power_by_vector(other)
        else:
            raise TypeError(f"Invalid type '{type(other)}' for raising a Vector by a power.")

    def __pow__(self, other: float):
        return self.__copy__().__ipow__(other)

    def __iadd__(self, other: 'Union[Vector, VectorCollection, Surface, IExpression, int, float]'):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] += other_coordinate
            self.__update_end()
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] += other
            self.__update_end()
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(self)
            return other_copy
        else:
            raise TypeError(f'Invalid type {type(other)} for adding vectors')

    def __isub__(self, other: 'Union[Vector, VectorCollection]'):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] -= other_coordinate
            self._end_coordinate = [start + direction for start, direction in zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] -= other
            self._end_coordinate = [start + direction for start, direction in zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(-self)
            return other_copy
        else:
            raise TypeError(f'Invalid type {type(other)} for adding vectors')

    def __sub__(self, other: 'Union[Vector, VectorCollection]'):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: 'Union[Vector, VectorCollection]'):
        return self.__neg__().__iadd__(other)

    def __add__(self, other: 'Union[Vector, VectorCollection]'):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: 'Union[Vector, VectorCollection]'):
        return self.__copy__().__iadd__(other)

    def __neg__(self):
        return Vector(direction_vector=[-x for x in self._direction_vector], start_coordinate=self._end_coordinate, end_coordinate=self._start_coordinate)

    def __str__(self):
        """
        :return: string representation of the vector
        """
        return f'start: {self._start_coordinate} end: {self._end_coordinate} direction: {self._direction_vector} '

    def __repr__(self):
        """

        :return: returns a string representation of the object's constructor
        """
        return f'Vector(start_coordinate={self._start_coordinate},end_coordinate={self._end_coordinate})'

    def __abs__(self):
        """
        :return: returns a vector with absolute values, preserves the starting coordinate but changes the ending point
        """
        return Vector(direction_vector=[abs(x) for x in self._direction_vector], start_coordinate=self._start_coordinate)

    def __len__(self):
        return self.length()

class Vector2D(Vector, IPlottable):

    def __init__(self, x, y, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 2:
                raise ValueError(f"Vector2D object can only receive 2D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 2:
                raise ValueError(f"Vector2D object can only receive 2D coordinates: got wrong 'end_coordinate' param")
        super().__init__(direction_vector=(x, y), start_coordinate=start_coordinate, end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    def plot(self, show=True, arrow_length_ratio: float=0.05, fig=None, ax=None):
        from kiwicalc.plotting.plots import plot_vector_2d
        return plot_vector_2d(self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0], self._direction_vector[1], show=show, fig=fig, ax=ax)

class Vector3D(Vector, IPlottable):

    def __init__(self, x, y, z, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 3:
                raise ValueError(f"Vector3D object can only receive 3D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 3:
                raise ValueError(f"Vector3D object can only receive 3D coordinates: got wrong 'end_coordinate' param")
        super(Vector3D, self).__init__(direction_vector=(x, y, z), start_coordinate=start_coordinate, end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    @property
    def z_step(self):
        return self._direction_vector[2]

    def plot(self, show=True, arrow_length_ratio: float=0.05, fig=None, ax=None):
        from kiwicalc.plotting.plots import plot_vector_3d
        u, v, w = (self._direction_vector[0], self._direction_vector[1], self._direction_vector[2])
        start_x, start_y, start_z = (self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[2])
        plot_vector_3d((start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig, ax=ax)

class VectorCollection:

    def __init__(self, *vectors):
        self.__vectors = []
        for vector in vectors:
            if isinstance(vector, Vector):
                self.__vectors.append(vector)
            elif isinstance(vector, Iterable):
                self.__vectors.append(Vector(vector))
            else:
                raise TypeError(f'Encountered invalid type {type(vector)} while building a vector collection.')

    @property
    def vectors(self):
        return self.__vectors

    @property
    def num_of_vectors(self):
        return len(self.__vectors)

    @vectors.setter
    def vectors(self, vectors):
        if isinstance(vectors, Vector):
            vectors = VectorCollection(vectors)
        elif isinstance(vectors, (list, set, tuple)):
            vectors = VectorCollection(*vectors)
        if isinstance(vectors, VectorCollection):
            self.__vectors = list(vectors.vectors)
        else:
            raise TypeError(f'Unexpected type {type(vectors)} in the setter property of vectors in class VectorCollection.\nExpected types VectorCollection, Vector, tuple, list, set')

    def append(self, vector: 'Union[Vector, Iterable[Union[Vector, IExpression, VectorCollection, int, float, Iterable]], VectorCollection]'):
        """ Append vectors to the collection of vectors """
        if isinstance(vector, Vector):
            self.__vectors.append(vector)
        elif isinstance(vector, VectorCollection):
            self.__vectors.extend(vector)
        elif isinstance(vector, Iterable) and (not isinstance(vector, IExpression)):
            for item in vector:
                if isinstance(item, Vector):
                    self.__vectors.append(item)
                elif isinstance(item, VectorCollection):
                    self.__vectors.extend(item)
                elif isinstance(item, Iterable):
                    self.__vectors.append(Vector(item))
                else:
                    raise TypeError(f'Invalid type {type(vector)} for appending into a VectorCollection')
        else:
            raise TypeError(f'Invalid type {type(vector)} for appending into a VectorCollection')

    def plot(self):
        from kiwicalc.plotting.plots import plot_vector_3d

        num_of_vectors = len(self.__vectors)
        if num_of_vectors > 0:
            if len(self.__vectors[0].start_coordinate) == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for vector in self.__vectors:
                    start = (vector.start_coordinate[0], vector.start_coordinate[1], vector.start_coordinate[2])
                    end = (vector.end_coordinate[0], vector.end_coordinate[1], vector.end_coordinate[2])
                    plot_vector_3d(start, end, fig=fig, ax=ax, show=False)
                min_x, max_x, min_y, max_y, min_z, max_z = _get_limits_vectors_3d(self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
                ax.set_zlim([min_z, max_z])
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                for vector in self.__vectors:
                    vector.plot(show=False, fig=fig, ax=ax)
                min_x, max_x, min_y, max_y = _get_limits_vectors_2d(self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
        plt.show()

    def filter(self, predicate: Callable[[Any], bool]=lambda x: bool(x)):
        return filter(predicate, self.__vectors)

    def map(self, func: Callable):
        return map(func, self.__vectors)

    def longest(self, get_index=False, remove=False):
        """
        returns the longest vector in the collection

        :param get_index: if True, returns a tuple: (index,longest_vector)
        :param remove: if True, removes the longest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        longest_vector = max(self.__vectors, key=lambda vector: vector.length())
        if not (get_index or remove):
            return longest_vector
        index = self.__vectors.index(longest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return (index, longest_vector)
        return longest_vector

    def shortest(self, get_index=False, remove=False):
        """
        returns the shortest vector in the collection

        :param get_index: if True, returns a tuple: (index,shortest_vector)
        :param remove: if True, removes the shortest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        shortest_vector = min(self.__vectors, key=lambda vector: vector.length())
        if not (get_index or remove):
            return shortest_vector
        index = self.__vectors.index(shortest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return (index, shortest_vector)
        return shortest_vector

    def find(self, vec: Vector):
        for index, vector in enumerate(self.__vectors):
            if vector == vec or vector is vec:
                return index
        return -1

    def nlongest(self, n: int):
        """returns the n longest vector for an integer n"""
        if not 0 <= n <= len(self.__vectors):
            raise ValueError(f'n must be between 0 and {len(self.__vectors)}')
        return sorted(self.__vectors, key=lambda vector: vector.length(), reverse=True)[:n]

    def nshortest(self, n: int):
        """returns the n shortest vector for an integer n"""
        if not 0 <= n <= len(self.__vectors):
            raise ValueError(f'n must be between 0 and {len(self.__vectors)}')
        return sorted(self.__vectors, key=lambda vector: vector.length())[:n]

    def sort_by_length(self, reverse=False):
        self.__vectors.sort(key=lambda vector: vector.length(), reverse=reverse)

    def pop(self, index: int=-1):
        return self.__vectors.pop(index)

    def __iadd__(self, other: 'Union[Vector, VectorCollection, Iterable]'):
        self.append(other)
        return self

    def __add__(self, other: 'Union[Vector, VectorCollection, Iterable]'):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other):
        return self.__copy__().__iadd__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float, IExpression)):
            for index in range(len(self.__vectors)):
                self.__vectors[index] *= other
            return self
        raise TypeError(f'Invalid type {type(other)} for multiplying a VectorCollection object.')

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f'Invalid type for dividing a VectorCollection object: {type(other)}. Expected a numberor an algebraic expression.')
        if other == 0:
            raise ValueError('Cannot divide a VectorCollection object by 0')
        for i in range(len(self.__vectors)):
            self.__vectors[i] *= 1 / other
        return self

    def __truediv__(self, other):
        return self.__copy__().__itruediv__(other)

    def __bool__(self):
        return bool(self.__vectors)

    def to_matrix(self):
        from kiwicalc.linalg.matrix import Matrix
        return Matrix([[copy_expression(expression) for expression in vector.direction] for vector in self.__vectors])

    def __eq__(self, other: 'Union[Vector, VectorCollection, Iterable]'):
        if other is None:
            return False
        if not isinstance(other, (Vector, VectorCollection)):
            if isinstance(other, Iterable):
                if isinstance(other[0], Iterable):
                    try:
                        other = VectorCollection(*other)
                    except (ValueError, TypeError):
                        try:
                            other = Vector(other)
                        except (ValueError, TypeError):
                            raise ValueError('Invalid value for equating VectorCollection objects.')
                else:
                    try:
                        other = Vector(other)
                    except (ValueError, TypeError):
                        raise ValueError('Invalid value for equating VectorCollection objects.')
            else:
                raise TypeError(f"Invalid type '{type(other)}' for equating VectorCollection objects.")
        if isinstance(other, Vector) and len(self.__vectors) == 1:
            return self.__vectors[0] == other
        elif isinstance(other, VectorCollection):
            if len(self.__vectors) != len(other.__vectors):
                return False
            for vec in self.__vectors:
                if other.__vectors.count(vec) != self.__vectors.count(vec):
                    return False
            return True
        else:
            raise TypeError(f'Invalid type {type(other)} for equating VectorCollection objects.')

    def __ne__(self, other: 'Union[Vector, VectorCollection, Iterable]'):
        return not self.__eq__(other)

    def __getitem__(self, item):
        return self.__vectors.__getitem__(item)

    def __setitem__(self, key, value):
        return self.__vectors.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__vectors.__delitem__(key)

    def __copy__(self):
        return VectorCollection(*(vector.__copy__() for vector in self.__vectors))

    def __contains__(self, other: Vector):
        if isinstance(other, Vector):
            return bool([vector for vector in self.__vectors if vector.__eq__(other)])

    def __iter__(self):
        self.__current_index = 0
        return self

    def __next__(self):
        if self.__current_index < len(self.__vectors):
            x = self.__vectors[self.__current_index]
            self.__current_index += 1
            return x
        else:
            raise StopIteration

    def __len__(self):
        """ number of vectors that the collection contains"""
        return len(self.__vectors)

    def total_number_of_items(self):
        """ Total number of items in all of the vectors. """
        return sum((len(vector) for vector in self.__vectors))

def _get_limits_vectors_2d(vectors):
    """Internal method: find the edge values for the scope of the 2d frame"""
    min_x = min((min(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)) * 1.05
    max_x = max((max(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)) * 1.05
    min_y = min((min(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)) * 1.05
    max_y = max((max(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)) * 1.05
    return (min_x, max_x, min_y, max_y)

def _get_limits_vectors_3d(vectors):
    """Internal method: find the edge values for the scope of the 3d frame"""
    min_x = min((min(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors))
    max_x = max((max(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors))
    min_y = min((min(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors))
    max_y = max((max(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors))
    min_z = min((min(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors))
    max_z = max((max(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors))
    return (min_x, max_x, min_y, max_y, min_z, max_z)
