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
        self._coefficient = Mono(1)
        if isinstance(expressions, str):
            raise NotImplementedError
        else:
            self._expressions = list()
            for expression in expressions:
                if isinstance(expression, (float, int)):
                    self._coefficient *= expression
                elif isinstance(expression, IExpression):
                    if gen_copies:
                        self._expressions.append(expression.__copy__())
                    else:
                        self._expressions.append(expression)
                elif isinstance(expression, str):
                    raise NotImplementedError
                else:
                    raise TypeError(f"Encountered an invalid type: '{type(expression)}', when creating a new Expression object.")
            self.simplify()

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
        result = self._coefficient.try_evaluate()
        if result is None:
            return None
        for expression in self._expressions:
            evaluation = expression.try_evaluate()
            if evaluation is None:
                return None
            result *= evaluation
        return result

    def __split_expressions(self, num_of_expressions: int):
        return (ExpressionMul(self._expressions[:num_of_expressions // 2]), ExpressionMul(self._expressions[num_of_expressions // 2:]))

    def derivative(self):
        num_of_expressions = len(self._expressions)
        if num_of_expressions == 0:
            return Mono(0)
        coefficient = self._coefficient.try_evaluate()
        coefficient = self._coefficient if coefficient is None else coefficient
        terms = []
        for derivative_index, expression in enumerate(self._expressions):
            derivative = expression.derivative()
            derivative_value = derivative.try_evaluate() if isinstance(derivative, IExpression) else derivative
            derivative = derivative if derivative_value is None else derivative_value
            factors = [coefficient, derivative]
            factors.extend(
                factor for index, factor in enumerate(self._expressions)
                if index != derivative_index
            )
            terms.append(ExpressionMul(factors))
        result = ExpressionSum(terms, copy=False)
        polynomial = result.to_poly()
        return polynomial if polynomial is not None else result

    def __copy__(self):
        copied = ExpressionMul(self._expressions)
        copied._coefficient = self._coefficient.__copy__()
        return copied

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def __iadd__(self, other):
        return ExpressionSum((self, other))

    def __isub__(self, other):
        return ExpressionSum((self, -other))

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._coefficient *= other
        elif isinstance(other, IExpression):
            evaluation = other.try_evaluate()
            if evaluation is not None:
                self._coefficient *= evaluation
            elif isinstance(other, ExpressionMul):
                self._coefficient *= other._coefficient
                self._expressions.extend(expression.__copy__() for expression in other._expressions)
            else:
                self._expressions.append(other.__copy__())
        else:
            raise TypeError(f"Invalid type {type(other)} for multiplying an ExpressionMul")
        return self

    def __itruediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __ipow__(self, power):
        self._coefficient **= power
        for index, expression in enumerate(self._expressions):
            self._expressions[index] = expression.__pow__(power)
        return self

    def __rpow__(self, other):
        return Exponent(other, self)

    def __str__(self) -> str:
        if self._coefficient == 0:
            return '0'
        if not self._expressions:
            return str(self._coefficient)
        accumulator = '' if self._coefficient == 1 else '-' if self._coefficient == -1 else f'{self._coefficient}*'
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
                if self._coefficient != other._coefficient:
                    return False
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
        return {
            'type': 'ExpressionMul',
            'coefficient': self._coefficient.to_dict(),
            'expressions': [expression.to_dict() for expression in self._expressions],
        }

    @staticmethod
    def from_dict(given_dict: dict):
        if given_dict.get('type', '').strip().lower() != 'expressionmul':
            raise ValueError("ExpressionMul.from_dict() expected an ExpressionMul serialization payload")
        result = ExpressionMul([create_from_dict(expression) for expression in given_dict['expressions']], gen_copies=False)
        result._coefficient = create_from_dict(given_dict['coefficient'])
        return result


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

