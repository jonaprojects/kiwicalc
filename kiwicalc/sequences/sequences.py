from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from functools import reduce
from math import log
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable
import matplotlib.pyplot as plt
from kiwicalc.core.interfaces import IPlottable
from kiwicalc.core.utils import is_lambda, lambda_from_recursive

class Sequence(ABC):

    @property
    @abstractmethod
    def first(self):
        pass

    @abstractmethod
    def in_index(self, index: int) -> float:
        pass

    @abstractmethod
    def index_of(self, item: float) -> float:
        pass

    @abstractmethod
    def sum_first_n(self, n: int) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def range(self, start: int, stop: int):
        return (self.in_index(current_index) for current_index in range(start, stop))

    def product_in_range(self, start: int, end: int):
        return reduce(lambda a, b: a * b, (self.in_index(i) for i in range(start, end)))

    def product_first_n(self, end: int):
        return self.product_in_range(1, end + 1)

    def sum_in_range(self, start: int, end: int):
        return sum((self.in_index(current_index) for current_index in range(start, end)))

    def __contains__(self, item: float):
        index = self.index_of(item)
        return index > 0

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start == 0:
                warnings.warn('Sequence indices start from 1 and not from 0, skipped to 1')
                start = 1
            else:
                start = item.start
            step = 1 if item.step is None else item.step
            return [self.in_index(i) for i in range(start, item.stop + 1, step)]
        elif isinstance(item, int):
            return self.in_index(item)

    def __generate_data(self, start: int, stop: int, step: int):
        return (list(range(start, stop, step)), [self.in_index(index) for index in range(start, stop, step)])

    def plot(self, start: int, stop: int, step: int=1, show=True):
        axes, y_values = self.__generate_data(start, stop, step)
        plt.plot(axes, y_values)
        if show:
            plt.show()

class GeometricSeq(Sequence, IPlottable):
    """
    A class that represents a geometric sequence, namely, a sequence in which every item can be
    multiplied by a constant (the ratio of the sequence) to reach the next item.
    """

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], ratio: float=None):
        """Create a new GeometricSeq object"""
        if isinstance(first_numbers, str):
            if ',' in first_numbers:
                first_numbers = [float(i) for i in tuple(first_numbers.split(','))]
            else:
                first_numbers = [float(i) for i in tuple(first_numbers.split(' '))]
        elif isinstance(first_numbers, (int, float)):
            first_numbers = [first_numbers]
        if isinstance(first_numbers, (tuple, list, set)):
            if not first_numbers:
                raise ValueError("GeometricSeq.__init__(): Cannot accept an empty collection for parameter 'first_numbers'")
            if any((number == 0 for number in first_numbers)):
                raise ValueError("GeometricSeq.__init__(): Zeroes aren't allowed in geometric sequences")
            self.__first = first_numbers[0]
            if ratio is not None:
                self.__ratio = ratio
                return
            if len(first_numbers) == 1:
                raise ValueError('GeometricSeq.__init__(): Please Enter more initial values, or specify the ratio of the sequence.')
            self.__ratio = first_numbers[1] / first_numbers[0]
        else:
            raise TypeError(f"GeometricSeq.__init__():Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__first

    @property
    def ratio(self):
        return self.__ratio

    def in_index(self, index: int) -> float:
        return self.__first * pow(self.__ratio, index - 1)

    def index_of(self, item: float) -> float:
        result = log(item / self.__first, self.__ratio) + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        if self.__ratio == 1:
            return self.__first * n
        return self.__first * (self.__ratio ** n - 1) / (self.__ratio - 1)

    def __repr__(self):
        return f'Sequence(first_numbers=({self.__first},),ratio={self.__ratio})'

    def __str__(self):
        return f'{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (ratio = {self.__ratio})'

class ArithmeticProg(Sequence):
    """A class for representing arithmetic progressions. for instance: 2, 4, 6, 8, 10 ..."""

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], difference: float=None):
        if isinstance(first_numbers, str):
            if ',' in first_numbers:
                first_numbers = [float(i) for i in tuple(first_numbers.split(','))]
            else:
                first_numbers = [float(i) for i in tuple(first_numbers.split(' '))]
        elif isinstance(first_numbers, (int, float)):
            first_numbers = [first_numbers]
        if isinstance(first_numbers, (tuple, list, set)):
            if not first_numbers:
                raise ValueError("ArithmeticProg.__init__(): Cannot accept an empty collection for parameter 'first_numbers'")
            self.__first = first_numbers[0]
            if difference is not None:
                self.__difference = difference
                return
            if len(first_numbers) == 1:
                raise ValueError('ArithmeticProg.__init__(): Please Enter more initial values, or specify the difference of the sequence.')
            self.__difference = first_numbers[1] - first_numbers[0]
        else:
            raise TypeError(f"ArithmeticProg.__init__():Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__first

    @property
    def difference(self):
        return self.__difference

    def in_index(self, index: int) -> float:
        return self.__first + self.__difference * (index - 1)

    def index_of(self, item: float) -> float:
        result = (item - self.__first) / self.__difference + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        return 0.5 * n * (2 * self.__first + (n - 1) * self.__difference)

    def __str__(self):
        return f'{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (difference = {self.__difference})'

    def __repr__(self):
        return f'Sequence(first_numbers=({self.__first},),difference={self.__difference})'

class RecursiveSeq(Sequence):

    def __init__(self, recursive_function: str, first_values: Iterable):
        """
        Create a new instance of a recursive sequence.

        :param recursive_function:
        :param first_values:
        """
        self.__first_values = {index: value for index, value in enumerate(first_values)}
        self.__recursive_string = recursive_function.strip()
        self.__lambda, self.__indices = lambda_from_recursive(self.__recursive_string)

    @property
    def first(self):
        return self.__first_values[0]

    def in_index(self, n: int, accumulate=True):
        return self.at_n(n, accumulate)

    def index_of(self, item):
        raise NotImplementedError

    def sum_first_n(self, n: int):
        raise NotImplementedError

    def at_n(self, n: int, accumulate=True):
        if n == 0:
            raise ValueError('Sequence indices start from 1, not from 0 - a1,a2,a3....')
        return self.__at_n(n - 1, accumulate)

    def __at_n(self, n: int, accumulate=True):
        """
        Get the nth element in the series.

        :param n: The place of the desired element. Must be an integer and greater than zero.
        :param accumulate: If set to true, results of computations will be saved to shorten execution time ( on the expense of the allocated memory).
        :return: Returns the nth element of the series.
        """
        if len(self.__indices) > len(self.__first_values):
            raise ValueError(f'Not enough initial values were entered for the series, got {len(self.__first_values)}, expected at least {len(self.__indices)} values')
        if n in self.__first_values:
            return self.__first_values[n]
        new_indices = [int(eval(old_index.replace('k', str(n)))) for old_index in self.__indices]
        pre_defined_values, undefined_indices = ([], [])
        for new_index in new_indices:
            if new_index in self.__first_values:
                pre_defined_values.append(self.__first_values[new_index])
            else:
                undefined_indices.append(new_index)
        if undefined_indices:
            pre_defined_values.extend([self.__at_n(index, accumulate) for index in undefined_indices])
        pre_defined_values.append(n + 1)
        result = self.__lambda(*pre_defined_values)
        if accumulate:
            self.__first_values[n] = result
        return result

    def place_already_found(self, n: int) -> bool:
        """
        Checks if the value in the specified place in the recursive series has already been computed.

        :param n: The place on the series, starting from 1. Must be an integer.
        :return: Returns True if the value has been computed, otherwise, False
        """
        return n in self.__first_values.keys()

    def __str__(self):
        return f'{self.__recursive_string}'
