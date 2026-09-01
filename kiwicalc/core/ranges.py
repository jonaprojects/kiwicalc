from __future__ import annotations
import math
import re
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable, Set
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.utils import decimal_range, is_number, round_decimal
from kiwicalc.core.operators import (
    Operator, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual,
    GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL,
    range_operator_from_string
)
from kiwicalc.core.interfaces import IExpression

class Range:
    __slots__ = ['__expression', '__minimum', '__maximum', '__min_operator', '__max_operator']

    def __init__(self, expression: 'Union[str,IExpression, Function, int, float]', limits: Union[set, list, tuple]=None, operators: Union[set, list, tuple]=None, dtype='poly', copy: bool=True):
        from kiwicalc.expressions.mono import Mono

        if isinstance(expression, str):
            expression, limits, operators = create_range(expression, get_tuple=True)
        if isinstance(expression, IExpression):
            self.__expression = expression.__copy__() if copy else expression
        elif isinstance(expression, (int, float)):
            self.__expression = Mono(expression)
        else:
            raise TypeError(f"Range.__init__(): Invalid type of expression: {type(expression)}.Expected types 'IExpression', 'Function', or str.")
        if not isinstance(limits, (set, list, tuple)):
            raise TypeError(f"Range.__init__(): Invalid type of limits: {type(limits)}. Expected types 'list', 'tuple', 'set'.")
        if len(limits) != 2:
            raise ValueError('The length')
        if limits[0] in (np.inf, -np.inf):
            self.__minimum = limits[0]
        elif isinstance(limits[0], (int, float)):
            self.__minimum = Mono(limits[0])
        elif isinstance(limits[0], IExpression):
            self.__minimum = limits[0].__copy__() if copy else limits[0]
        elif limits[0] is None:
            self.__minimum = -np.inf
        else:
            raise TypeError("Minimum of the range must be of type 'IExpression', 'Function', None, and inf ")
        if limits[1] in (np.inf, -np.inf):
            self.__maximum = limits[1]
        elif isinstance(limits[1], (int, float)):
            self.__maximum = Mono(limits[1])
        elif isinstance(limits[1], IExpression):
            self.__maximum = limits[1].__copy__() if copy else limits[1]
        elif limits[1] is None:
            self.__maximum = np.inf
        else:
            raise TypeError("Maximum of the range must be of type 'IExpression', 'Function', None, and inf ")
        if not isinstance(operators, (list, set, tuple)):
            raise TypeError(f"Range.__init__(): Invalid type of operators: {type(limits)}. Expected types 'list', 'tuple', 'set'.")
        if not len(operators) == 2:
            raise ValueError(f'Range.__init__(): The length of the operators must be 2.')
        if copy:
            self.__min_operator = operators[0].__copy__() if hasattr(operators[0], '__copy__') else operators[0]
            self.__max_operator = operators[1].__copy__() if hasattr(operators[1], '__copy__') else operators[1]
        else:
            self.__min_operator, self.__max_operator = operators

    @property
    def expression(self):
        return self.__expression

    @property
    def min_limit(self):
        return self.__minimum

    @property
    def max_limit(self):
        return self.__maximum

    @property
    def min_operator(self):
        return self.__min_operator

    @property
    def max_operator(self):
        return self.__max_operator

    def try_evaluate(self):
        return self.__evaluate()

    def evaluate_when(self, **kwargs):
        if isinstance(self.__minimum, IExpression):
            min_eval = self.__minimum.when(**kwargs).try_evaluate()
        else:
            min_eval = None
        expression_eval = self.__expression.when(**kwargs).try_evaluate()
        if isinstance(self.__maximum, IExpression):
            max_eval = self.__maximum.when(**kwargs).try_evaluate()
        else:
            max_eval = None
        return self.__evaluate(min_eval, expression_eval, max_eval)

    def __evaluate(self, min_eval: float=None, expression_eval: float=None, max_eval: float=None) -> Optional[bool]:
        if self.__minimum == np.inf or self.__maximum == -np.inf:
            return False
        expression_eval = self.__expression.try_evaluate() if expression_eval is None else expression_eval
        if self.__minimum == -np.inf and self.__maximum == np.inf:
            return True
        if self.__minimum != -np.inf:
            minimum_evaluation = self.__minimum.try_evaluate() if min_eval is None else min_eval
            if self.__maximum != np.inf:
                maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
                if None not in (minimum_evaluation, maximum_evaluation):
                    if maximum_evaluation < minimum_evaluation:
                        return False
                if None not in (maximum_evaluation, expression_eval):
                    if not self.__max_operator.method(expression_eval, maximum_evaluation):
                        return False
            if None not in (minimum_evaluation, expression_eval):
                return self.__min_operator.method(minimum_evaluation, expression_eval)
            return None
        else:
            maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
            if None not in (maximum_evaluation, expression_eval):
                return self.__max_operator.method(expression_eval, maximum_evaluation)
            return None

    def __str__(self):
        if self.__minimum == -np.inf and self.__maximum == np.inf:
            return f'-∞{self.__min_operator}{self.__expression}{self.__max_operator}∞'
        if self.__minimum == -np.inf:
            minimum_str = ''
        else:
            minimum_str = f'{self.__minimum}{self.__min_operator}'
        if self.__maximum == np.inf:
            maximum_str = ''
        else:
            maximum_str = f'{self.__max_operator}{self.__maximum}'
        return f'{minimum_str}{self.__expression}{maximum_str}'

    def __copy__(self):
        return Range(self.__expression, (self.__minimum, self.__maximum), (self.__min_operator, self.__max_operator), copy=True)

class RangeCollection:
    __slots__ = ['_ranges']

    def __init__(self, ranges: 'Iterable[Range, RangeCollection]', copy=False):
        if copy:
            self._ranges = [my_range.__copy__() for my_range in ranges]
        else:
            self._ranges = [my_range for my_range in ranges]

    @property
    def ranges(self):
        return self._ranges

    def chain(self, range_obj: Range, copy=False):
        if not isinstance(range_obj, Range):
            raise TypeError(f"Invalid type {type(range_obj)} for chaining Ranges. Expected type: 'Range' ")
        self._ranges.append(range_obj.__copy__() if copy else range_obj)
        return self

    def __or__(self, other: Range):
        return RangeOR((self, other))

    def __and__(self, other):
        return RangeAND((self, other))

    def __copy__(self):
        return RangeCollection(ranges=self._ranges, copy=True)

    def __str__(self):
        return ', '.join((f'({my_range.__str__()})' if isinstance(my_range, RangeCollection) else my_range.__str__() for my_range in self._ranges))

class RangeOR(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the OR method.
    For instance:
    (x^2 > 25) or (x^2 < 9)
    Or a more complicated example:
    (5<x<6 and x^2>26) or x<7 or (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: 'Iterable[Range, RangeCollection]', copy=False):
        super(RangeOR, self).__init__(ranges)

    def try_evaluate(self):
        results = [my_range.try_evaluate() for my_range in self._ranges]
        if any(result is True for result in results):
            return True
        if all(result is False for result in results):
            return False
        return None

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return ' or '.join((f'({my_range.__str__()})' if isinstance(my_range, RangeCollection) else my_range.__str__() for my_range in self._ranges))

    def __copy__(self):
        return RangeOR(self._ranges, copy=True)

class RangeAND(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the AND method.
    For instance:
    (x^2 > 25) and (x>0)
    Or a more complicated example:
    (5<x<6 and x^2>26) and x<7 and (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: 'Iterable[Range, RangeCollection]', copy=False):
        super(RangeAND, self).__init__(ranges)

    def try_evaluate(self):
        results = [my_range.try_evaluate() for my_range in self._ranges]
        if any(result is False for result in results):
            return False
        if all(result is True for result in results):
            return True
        return None

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return ' and '.join((f'({my_range.__str__()})' if isinstance(my_range, RangeCollection) else my_range.__str__() for my_range in self._ranges))

    def __copy__(self):
        return RangeAND(self._ranges, copy=True)

def values_in_range(func: Callable, start: float, end: float, step: float, round_results: bool=False):
    """
    fetches all the valid values of a function in the specified range
    :param func: A callable function, that accepts one parameter and returns a single result
    :param start: the beginning of the range
    :param end: the end of the range
    :param step: the interval between each item in the range
    :return: returns the values in the range, and their valid results
    """
    if round_results:
        values = [round_decimal(i) for i in decimal_range(start, end, step)]
        results = [round_decimal(func(i)) for i in values]
    else:
        values = [_ for _ in decimal_range(start, end, step)]
        results = [func(i) for i in values]
    filtered = [
        (value, float(result) if isinstance(result, bool) else result)
        for value, result in zip(values, results)
        if result is not None
    ]
    return ([value for value, _ in filtered], [result for _, result in filtered])

def create_range(expression: str, min_dtype: str='poly', expression_dtype: str='poly', max_dtype='poly', get_tuple=False):
    from kiwicalc.expressions.factory import create
    exprs = re.split('(<=|>=|>|<)', expression)
    num_of_expressions = len(exprs)
    if num_of_expressions == 5:
        limits = (create(exprs[0], dtype=min_dtype), create(exprs[4], dtype=max_dtype))
        middle = create(exprs[2], dtype=expression_dtype)
        min_operator, max_operator = (range_operator_from_string(exprs[1]), range_operator_from_string(exprs[3]))
    elif num_of_expressions == 3:
        middle = create(exprs[0], dtype=min_dtype)
        my_operator = exprs[1]
        if '>' in my_operator:
            my_operator = my_operator.replace('>', '<')
            min_operator, max_operator = (range_operator_from_string(my_operator), None)
            limits = (create(exprs[2], dtype=min_dtype), None)
        elif '<' in my_operator:
            min_operator, max_operator = (None, range_operator_from_string(my_operator))
            limits = (None, create(exprs[2], dtype=max_dtype))
        else:
            raise ValueError(f"Invalid operator: {my_operator}. Expected: '>', '<', '>=' ,'<='")
    else:
        raise ValueError(f"Invalid string for creating a Range expression: {expression}. Expected expressions such as '3<x<5', 'x^2 > 16', etc..")
    if get_tuple:
        return (middle, limits, (min_operator, max_operator))
    return Range(expression=middle, limits=limits, operators=(min_operator, max_operator))
