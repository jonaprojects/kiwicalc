from __future__ import annotations
import math
import cmath
from math import (
    sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    asinh, acosh, atanh, floor, ceil, gamma, comb, factorial, log
)
import random
import json
import warnings
from functools import reduce
from collections import Counter
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.constants import (
    TRIGONOMETRY_CONSTANTS, MATHEMATICAL_CONSTANTS,
    ptn, number_pattern, allowed_characters, pi, e, tau
)
from kiwicalc.core.interfaces import IExpression, IPlottable, IScatterable
from kiwicalc.core.utils import (
    copy_expression, clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, is_lambda, round_decimal, to_lambda,
    float_gcd, gcd, sorted_expressions, equal_ignore_order, process_object,
    contains_from_list, decimal_range, apply_on, is_evaluatable, only_numbers_letters,
    apply_parenthesis, handle_parenthesis, handle_abs, handle_factorial,
    _format_minus, ln, max_power, formatted_expression
)
from kiwicalc.parsing.parse_expression import (
    split_expression, extract_variables_from_expression, __data_from_single,
    mono_from_str, poly_from_str, log_from_str, TrigoExpr_from_str,
    TrigoExprs_from_str, poly_frac_from_str, fetch_power, fetch_variable,
    ParseExpression
)
from kiwicalc.expressions.factory import create, create_from_dict

from kiwicalc.expressions.mono import Mono

class ExpressionSum(IExpression, IPlottable, IScatterable):
    __slots__ = ['_expressions', '_current_index']

    def __init__(self, expressions: Iterable[IExpression]=None, copy=True):
        self._current_index = 0
        if expressions is None:
            self._expressions = []
        elif copy:
            self._expressions = [copy_expression(expression) for expression in expressions]
        else:
            self._expressions = [expression for expression in expressions]
        expressions_to_add = []
        indices_to_delete = []
        for index, expression in enumerate(self._expressions):
            if isinstance(expression, ExpressionSum):
                expressions_to_add.extend(expression._expressions)
                indices_to_delete.append(index)
            elif isinstance(expression, (int, float)):
                self._expressions[index] = Mono(expression)
        self._expressions = [expression for index, expression in enumerate(self._expressions) if index not in indices_to_delete]
        self._expressions.extend(expressions_to_add)

    @property
    def expressions(self):
        return self._expressions

    def append(self, expression: IExpression):
        self._expressions.append(expression)

    def assign_to_all(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def when_all(self, **kwargs):
        return ExpressionSum((expression.when(**kwargs) for expression in self._expressions), copy=False)

    def __add_or_sub(self, other: 'Union[IExpression, ExpressionSum]', operation='+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            if operation == '+':
                self._expressions.append(Mono(other))
            else:
                self._expressions.append(Mono(-other))
            self.simplify()
            return self
        elif isinstance(other, IExpression):
            my_evaluation, other_evaluation = (self.try_evaluate(), other.try_evaluate())
            if None not in (my_evaluation, other_evaluation):
                if operation == '+':
                    return Mono(my_evaluation + other_evaluation)
                return Mono(my_evaluation - other_evaluation)
            elif my_evaluation is not None:
                if operation == '+':
                    return other.__add__(my_evaluation)
                return Mono(my_evaluation).__sub__(other)
            elif other_evaluation is not None:
                if operation == '+':
                    self._expressions.append(Mono(other_evaluation))
                else:
                    self._expressions.append(Mono(-other_evaluation))
                self.simplify()
                return self
            else:
                if operation == '+':
                    self._expressions.append(other.__copy__())
                else:
                    self._expressions.append(other.__neg__())
                self.simplify()
                return self
            if isinstance(other, ExpressionSum):
                if operation == '+':
                    for expression in other._expressions:
                        self._expressions.append(expression)
                else:
                    for expression in other._expressions:
                        self._expressions.append(expression)
                self.simplify()
                return self
        self.simplify()
        return self

    def __iadd__(self, other: 'Union[IExpression, ExpressionSum]'):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other: 'Union[IExpression, int, float, ExpressionSum]'):
        return self.__add_or_sub(other, operation='-')

    def __rsub__(self, other: 'Union[IExpression, int, float, ExpressionSum]'):
        return ExpressionSum((other, -self))

    def __neg__(self):
        return ExpressionSum((expression.__neg__() for expression in self._expressions))

    def __imul__(self, other: 'Union[IExpression, int, float, ExpressionSum]'):
        if isinstance(other, ExpressionSum):
            final_expressions: List[Optional[IExpression]] = []
            for my_expression in self._expressions:
                for other_expression in other._expressions:
                    final_expressions.append(my_expression * other_expression)
            self._expressions = final_expressions
            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self
        else:
            for index in range(len(self._expressions)):
                self._expressions[index] *= other
            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self

    def __ipow__(self, power: Union[IExpression, int, float]):
        if isinstance(power, (int, float)):
            length = len(self._expressions)
            if power == 0:
                return Mono(1)
            if length == 0:
                return Mono(0) if power > 0 else Fraction(1, Mono(0))
            if length == 1:
                self._expressions[0] **= power
                return self
            if isinstance(power, int) or isinstance(power, float) and power.is_integer():
                integer_power = int(power)
                if integer_power < 0:
                    return Fraction(1, self.__ipow__(abs(integer_power)))
                copy_of_self = self.__copy__()
                result = self.__copy__()
                for _ in range(integer_power - 1):
                    result *= copy_of_self
                return result
            return Exponent(self, power)
        elif isinstance(power, IExpression):
            other_evaluation = power.try_evaluate()
            if other_evaluation is None:
                return Exponent(self, power)
            return self.__ipow__(other_evaluation)
        else:
            raise TypeError(f"Invalid type '{type(power)}' for raising an 'ExpressionSum' object by a power")

    def __pow__(self, power: Union[IExpression, int, float]):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[IExpression, int, float]) -> 'Union[ExpressionSum,IExpression]':
        if other == 0:
            raise ValueError('Cannot divide an ExpressionSum object by 0.')
        if isinstance(other, (int, float)):
            for my_expression in self._expressions:
                my_expression /= other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} for dividing with 'ExpressionSum' class.")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            if other == 0:
                raise ValueError('Cannot divide an ExpressionSum object by 0.')
            for my_expression in self._expressions:
                my_expression /= other_evaluation
            return self
        if isinstance(other, (ExpressionSum, Poly, TrigoExprs)):
            return Fraction(self, other)
        for my_expression in self._expressions:
            my_expression /= other
        result = self.to_poly()
        if result is not None:
            return result
        self.simplify()
        return self

    def assign(self, **kwargs) -> None:
        for expression in self._expressions:
            expression.assign(**kwargs)

    def is_poly(self):
        return all((isinstance(expression, (Mono, Poly)) for expression in self._expressions))

    def to_poly(self) -> 'Optional[Poly]':
        """Tries to convert the ExpressionSum object to a Poly object (to a polynomial).
        If not successful, None will be returned.
        """
        if not self.is_poly():
            return None
        my_poly = Poly(0)
        for expression in self._expressions:
            my_poly += expression
        return my_poly

    def simplify(self):
        for expression in self._expressions:
            expression.simplify()
        evaluation_sum: float = 0
        delete_indices = []
        for index, expression in enumerate(self._expressions):
            expression_evaluation = expression.try_evaluate()
            if expression_evaluation is not None:
                evaluation_sum += expression_evaluation
                delete_indices.append(index)
        self._expressions = [expression for index, expression in enumerate(self._expressions) if index not in delete_indices]
        if evaluation_sum:
            self._expressions.append(Mono(evaluation_sum))

    def try_evaluate(self):
        """ Try to evaluate the expressions into float or an int """
        evaluation_sum = 0
        for expression in self._expressions:
            expression_evaluation: Optional[Union[int, float]] = expression.try_evaluate()
            if expression_evaluation is None:
                return None
            evaluation_sum += expression_evaluation
        return evaluation_sum

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(variables.union(expression.variables))
        return variables

    def derivative(self):
        warnings.warn('This feature is still experimental, and might not work.')
        if any((not hasattr(expression, 'derivative') for expression in self._expressions)):
            raise AttributeError('Not all expressions support derivatives')
        return ExpressionSum([expression.derivative() for expression in self._expressions], copy=False)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self._expressions):
            value = self._expressions[self._current_index]
            self._current_index += 1
            return value
        raise StopIteration

    def __getitem__(self, item):
        return self._expressions.__getitem__(item)

    def __len__(self):
        return len(self._expressions)

    def __copy__(self):
        return ExpressionSum((expression.__copy__() for expression in self._expressions))

    def __str__(self):
        if not self._expressions:
            return '0'
        accumulator = ''
        for expression in self._expressions:
            expression_string: str = expression.__str__()
            if not expression_string.startswith('-'):
                accumulator += '+'
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def python_syntax(self) -> str:
        if not self._expressions:
            return '0'
        accumulator = ''
        for expression in self._expressions:
            expression_string: str = expression.python_syntax()
            if not expression_string.startswith('-'):
                accumulator += '+'
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def to_dict(self):
        return {'type': 'ExpressionSum', 'expressions': [expression.to_dict() for expression in self._expressions]}

    @staticmethod
    def from_dict(given_dict: dict):
        if given_dict.get('type', '').strip().lower() != 'expressionsum':
            raise ValueError("ExpressionSum.from_dict() expected an ExpressionSum serialization payload")
        return ExpressionSum([create_from_dict(expression) for expression in given_dict['expressions']], copy=False)

    def __eq__(self, other: Union[IExpression, int, float]):
        """Tries to figure out whether the expressions are equal. May not apply to special cases such as trigonometric
        identities"""
        if isinstance(other, (int, float)):
            evaluation = self.try_evaluate()
            return evaluation is not None and evaluation == other
        elif isinstance(other, IExpression):
            if isinstance(other, ExpressionSum):
                if len(self._expressions) != len(other._expressions):
                    return False
                for my_expression in self._expressions:
                    my_count = self._expressions.count(my_expression)
                    other_count = other._expressions.count(my_expression)
                    if my_count != other_count:
                        return False
                return True
            elif len(self._expressions) == 1:
                return self._expressions[0] == other
            else:
                other_evaluation = other.try_evaluate()
                if other_evaluation is None:
                    return False
                my_evaluation = self.try_evaluate()
                if my_evaluation is None:
                    return False
                return my_evaluation == other_evaluation

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


from kiwicalc.expressions.var import Var
from kiwicalc.expressions.mono import Mono
from kiwicalc.expressions.poly import FastPoly, Poly, synthetic_division
from kiwicalc.expressions.sum import ExpressionSum
from kiwicalc.expressions.mul import ExpressionMul
from kiwicalc.expressions.fractions import Fraction, PolyFraction
from kiwicalc.expressions.roots import Root, Sqrt
from kiwicalc.expressions.log import Log, PolyLog, Ln
from kiwicalc.expressions.trigonometry import (
    TrigoExpr, Sin, Asin, Cos, Acos, Tan, Atan, Cot,
    Sec, Acot, ASec, Csc, ACsc, TrigoExprs
)
from kiwicalc.expressions.special import Factorial, Abs, Exponent

