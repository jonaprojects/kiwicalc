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

class ExpressionMul(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, expressions: Union[Iterable[Union[IExpression, float, int, str]], str], gen_copies=True):
        if isinstance(expressions, str):
            raise NotImplementedError
        else:
            self._expressions = list()
            for expression in expressions:
                if isinstance(expression, (float, int)):
                    self._expressions.append(Mono(expression))
                elif isinstance(expression, IExpression):
                    if gen_copies:
                        self._expressions.append(expression.__copy__())
                    else:
                        self._expressions.append(expression)
                elif isinstance(expression, str):
                    raise NotImplementedError
                else:
                    raise TypeError(f"Encountered an invalid type: '{type(expression)}', when creating a new Expression object.")

    @property
    def expressions(self):
        return self._expressions

    def assign(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def python_syntax(self):
        if not self._expressions:
            return self._coefficient.python_syntax()
        accumulator = f'({self._coefficient})*'
        for iexpression in self._expressions:
            accumulator += f'({iexpression.python_syntax()})*'
        return accumulator[:-1]

    def simplify(self):
        if self._coefficient == 0:
            self._expressions = []

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(expression.variables)
        return variables

    def try_evaluate(self):
        evaluated_expressions = [expression.try_evaluate() for expression in self._expressions]
        if all(evaluated_expressions):
            return sum(evaluated_expressions)

    def __split_expressions(self, num_of_expressions: int):
        return (ExpressionMul(self._expressions[:num_of_expressions // 2]), ExpressionMul(self._expressions[num_of_expressions // 2:]))

    def derivative(self):
        print(f'calculating the derivative of {self}, num of expressions: {len(self._expressions)}')
        num_of_expressions = len(self._expressions)
        if num_of_expressions == 0:
            return None
        if num_of_expressions == 1:
            return self._expressions[0].derivative()
        elif num_of_expressions == 2:
            first, second = (self._expressions[0], self._expressions[1])
            return first.derivative() * second + second.derivative() * first
        else:
            expressionMul1, expressionMul2 = self.__split_expressions(num_of_expressions)
            first_derivative, second_derivative = (expressionMul1.derivative(), expressionMul2.derivative())
            if isinstance(first_derivative, (int, float)):
                first_derivative = Mono(first_derivative)
            if isinstance(second_derivative, (int, float)):
                second_derivative = Mono(second_derivative)
            return first_derivative * expressionMul2 + second_derivative * expressionMul1

    def __copy__(self):
        return ExpressionMul(self._expressions)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= 1
        return copy_of_self

    def __iadd__(self, other):
        return ExpressionSum((self, other))

    def __isub__(self, other):
        return ExpressionSum((self, other.__neg__()))

    def __imul__(self, other):
        self._expressions.append(other)
        return self

    def __itruediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __ipow__(self, power):
        for index, expression in enumerate(self._expressions):
            self._expressions[index] = expression.__pow__(power)
        return self

    def __rpow__(self, other):
        return Exponent(other, self)

    def __str__(self) -> str:
        accumulator = f''
        for index, expression in enumerate(self._expressions):
            content = expression.__str__()
            if index > 0 and (not content.startswith('-')):
                content = f'*{content}'
            if not content.endswith(')'):
                content = f'({content})'
            accumulator += content
        return accumulator

    def __eq__(self, other: Union[IExpression, int, float]) -> bool:
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation is not None and my_evaluation == other
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if isinstance(other, ExpressionMul):
                if len(self._expressions) != len(other._expressions):
                    return False
                return all(e in other._expressions for e in self._expressions)
            else:
                if len(self._expressions) == 1 and self._expressions[0] == other:
                    return True
                return False
        else:
            raise TypeError(f"Invalid type {type(other)} for equality checking in Expression class")

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        pass

    def from_dict(self):
        pass


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

