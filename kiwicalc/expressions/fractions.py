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
from kiwicalc.expressions.poly import Poly

class Fraction(IExpression):
    __slots__ = ['_numerator', '_denominator']

    def __init__(self, numerator: 'Union[IExpression,float,int]', denominator: 'Optional[Union[IExpression,float,int]]'=None, gen_copies=True):
        if isinstance(numerator, (float, int)):
            self._numerator = Mono(numerator)
        elif isinstance(numerator, IExpression):
            self._numerator = numerator.__copy__() if gen_copies else numerator
        else:
            raise TypeError(f'Unexpected type {type(numerator)} in Fraction.__init__.Modify the type of the numerator parameter to a valid one.')
        if denominator is None:
            self._denominator = Mono(1)
            return
        if isinstance(denominator, (float, int)):
            self._denominator = Mono(denominator)
        elif isinstance(denominator, IExpression):
            self._denominator = denominator.__copy__() if gen_copies else denominator
        else:
            raise TypeError(f'Unexpected type {type(denominator)} in Fraction.__init__. Modify the type of thedenominator parameter to a valid one')

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    @property
    def variables(self):
        return set(self._numerator.variables).union(self._denominator.variables)

    def assign(self, **kwargs):
        self._numerator.assign(**kwargs)
        self._denominator.assign(**kwargs)

    def derivative(self):
        return (self._numerator.derivative() * self._denominator - self._numerator * self._denominator.derivative()) / self._denominator ** 2

    def integral(self):
        pass

    def simplify(self):
        pass

    def try_evaluate(self) -> Optional[Union[int, float]]:
        """ try to evaluate the expression into a float or int value, if not successful, return None"""
        numerator_evaluation = self._numerator.try_evaluate()
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation is None:
            if self._numerator == 0:
                return 0
        if denominator_evaluation == 0:
            raise ZeroDivisionError(f'Denominator of fraction {self.__str__()} was evaluated into 0. Cannot divide by 0.')
        if None not in (numerator_evaluation, denominator_evaluation):
            return numerator_evaluation / denominator_evaluation
        division_result = self._numerator / self._denominator
        if isinstance(division_result, Fraction):
            return None
        division_evaluation = division_result.try_evaluate()
        if division_evaluation is not None:
            return division_evaluation
        return None

    def to_dict(self):
        return {'type': 'Fraction', 'numerator': self._numerator.to_dict(), 'denominator': self._denominator.to_dict() if self._denominator is not None else None}

    @staticmethod
    def from_dict(given_dict: dict):
        numerator_obj = create_from_dict(given_dict['numerator'])
        denominator_obj = create_from_dict(given_dict['denominator'])
        return Fraction(numerator=numerator_obj, denominator=denominator_obj)

    def __iadd__(self, other: Union[IExpression, int, float]):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return Mono(coefficient=my_evaluation + other)
            else:
                return ExpressionSum((self, Mono(coefficient=other)))
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):
                return Mono(coefficient=my_evaluation + other_evaluation)
            if isinstance(other, ExpressionSum):
                copy_of_other = other.__copy__()
                copy_of_other += self
                return copy_of_other
            elif isinstance(other, Fraction):
                if self._denominator == other._denominator:
                    self._numerator += other._numerator
                    return self
                else:
                    return ExpressionSum((self, other))
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(f"Invalid type '{type(other)}' for addition with fractions")

    def __isub__(self, other: Union[IExpression, int, float]):
        return self.__iadd__(-other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._numerator *= -1
        return copy_of_self

    def __imul__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        if self._denominator == other:
            self._denominator = Mono(1)
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return Mono(my_evaluation)
            return self._numerator
        self._numerator *= other
        return self

    def __mul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._denominator
            self._denominator *= other._numerator
        else:
            self._denominator *= other
        return self

    def __rmul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        self._numerator **= other
        self._denominator **= other
        self.simplify()
        return self

    def __rpow__(self, other):
        return Exponent(self, other)

    def __copy__(self):
        return Fraction(self._numerator, self._denominator)

    def __eq__(self, other: Union[IExpression, int, float]) -> Optional[bool]:
        if other is None:
            return False
        numerator_evaluation = self._numerator.try_evaluate()
        if numerator_evaluation == 0:
            return other == 0
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation == 0:
            raise ValueError(f'Denominator of a fraction cannot be 0.')
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return my_evaluation == other
            return None
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if isinstance(other, Fraction):
                if self._numerator == other._numerator and self._denominator == other._denominator:
                    return True
                numerator_ratio = self._numerator / other._numerator
                denominator_ratio = self._denominator / other._denominator
                return numerator_ratio == denominator_ratio
            else:
                pass

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def python_syntax(self) -> str:
        return f'({self._numerator.python_syntax()})/({self._denominator.python_syntax()})'

    def __str__(self):
        return f'({self._numerator.__str__()})/({self._denominator.__str__()})'

class PolyFraction(Fraction):
    """
    Creating a new algebraic fraction with a polynomial numerator and denominator.
    In later version, further types of expressions will be allowed in fractions.
    """

    def __init__(self, numerator, denominator=None, gen_copies=True):
        if denominator is None:
            if isinstance(numerator, str):
                numerator1, denominator1 = poly_frac_from_str(numerator, get_tuple=True)
                super().__init__(numerator1, denominator1)
            elif isinstance(numerator, PolyFraction):
                super().__init__(numerator._numerator.__copy__() if gen_copies else numerator._numerator, numerator._denominator.__copy__() if gen_copies else numerator._denominator)
            elif isinstance(numerator, (int, float, Mono, Poly)):
                super().__init__(Poly(numerator), Mono(1))
            else:
                raise TypeError(f'Invalid type for a numerator in PolyFraction : {type(numerator)}.')
        else:
            if isinstance(numerator, Poly):
                numerator = numerator.__copy__()
            elif isinstance(numerator, (int, float, str, Mono)):
                numerator = Poly(numerator)
            else:
                raise TypeError(f'Invalid type for a numerator in PolyFraction : {type(numerator)}. Expected types  Poly, Mono, str , float , int')
            if isinstance(denominator, Poly):
                denominator = denominator.__copy__()
            elif isinstance(denominator, (int, float, str, Mono)):
                denominator = Poly(denominator)
            else:
                raise TypeError(f'Invalid type for a denominator in PolyFraction : {type(denominator)}. Expected types  Poly, Mono, str , float , int')
            super().__init__(numerator, denominator)

    def roots(self, epsilon: float=1e-06, nmax: int=100000):
        return self._numerator.roots(epsilon, nmax)

    def invalid_values(self):
        """ When the denominator evaluates to 0"""
        return self._denominator.roots()

    def horizontal_asymptote(self):
        power1, power2 = (self._numerator.expressions[0].highest_power(), self._denominator.expressions[0].highest_power())
        if power1 > power2 or power1 == power2 == 0:
            return tuple()
        if power1 < power2:
            return 0
        return (power1 / power2,)

    def __str__(self):
        return f'({self._numerator})/({self._denominator})'

    def __repr__(self):
        return f'PolyFraction({self._numerator.__str__()},{self._denominator.__str__()})'

    def __iadd__(self, other):
        if other == 0:
            return self
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator += other._numerator
                return self
            elif (division_result := self._denominator.__truediv__(other._denominator, get_remainder=True))[1] == 0:
                self._numerator += other._numerator * division_result[0]
                return self
            elif (division_result := (other._denominator / self._denominator))[1] == 0:
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator += other._numerator
                return self
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def __radd__(self, other):
        new_copy = self.__copy__()
        return new_copy.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator -= other._numerator
                return self
            elif (division_result := (self._denominator / other._denominator))[1] == 0:
                self._numerator -= other._numerator * division_result[0]
                return self
            elif (division_result := (other._denominator / self._denominator))[1] == 0:
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator -= other._numerator
                return self
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def __sub__(self, other):
        new_copy = self.__copy__()
        return new_copy.__isub__(other)

    def __rsub__(self, other):
        new_copy = self.__copy__()
        new_copy.__isub__(other)
        new_copy.__imul__(-1)

    def __imul__(self, other):
        if isinstance(other, PolyFraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        elif isinstance(other, (int, float, Mono, Poly)):
            self._numerator *= other
            return self
        else:
            raise TypeError(f'Invalid type {type(other)} for multiplying PolyFraction objects. Allowed types:  PolyFraction, Mono, Poly, int, float')

    def __mul__(self, other):
        new_copy = self.__copy__()
        return new_copy.__imul__(other)

    def __rmul__(self, other):
        new_copy = self.__copy__()
        new_copy.__imul__(other)
        return new_copy

    def __rtruediv__(self, other):
        inverse_fraction: PolyFraction = self.reciprocal()
        return inverse_fraction.__imul__(other)

    def reciprocal(self):
        return PolyFraction(self._denominator, self._numerator)

    def __copy__(self):
        """Create a new copy of the polynomial fraction"""
        return PolyFraction(self._numerator, self._denominator)


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

