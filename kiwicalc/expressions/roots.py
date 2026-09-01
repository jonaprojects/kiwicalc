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

class Root(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_inside', '_root']

    def __init__(self, inside: Union[IExpression, float, int], root_by: Union[IExpression, float, int]=2, coefficient: Union[int, float, IExpression]=Mono(1)):
        self._coefficient = process_object(coefficient, class_name='Root', method_name='__init__', param_name='coefficient')
        self._inside = process_object(inside, class_name='Root', method_name='__init__', param_name='inside')
        self._root = process_object(root_by, class_name='Root', method_name='__init__', param_name='root_by')

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def inside(self):
        return self._inside

    @property
    def root(self):
        return self._root

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._inside.assign(**kwargs)
        self._root.assign(**kwargs)

    def try_evaluate(self) -> Optional[Union[complex, float, ValueError]]:
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation == 0:
            return 0
        inside_evaluation = self._inside.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        if None not in (coefficient_evaluation, inside_evaluation, root_evaluation):
            if root_evaluation == 0:
                return ValueError('Cannot compute root by 0')
            return coefficient_evaluation * inside_evaluation ** (1 / root_evaluation)
        return None

    def simplify(self) -> None:
        self._coefficient.simplify()
        self._root.simplify()
        self._inside.simplify()

    @property
    def variables(self):
        variables = self._coefficient.variables
        variables.update(self._inside.variables)
        variables.update(self._root.variables)
        return variables

    def to_dict(self):
        return {'type': 'Root', 'coefficient': self._coefficient.to_dict(), 'inside': self._inside.to_dict(), 'root_by': self._root.to_dict()}

    @staticmethod
    def from_dict(given_dict: dict):
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        inside_obj = create_from_dict(given_dict['inside'])
        root_obj = create_from_dict(given_dict['root_by'])
        return Root(coefficient=coefficient_obj, inside=inside_obj, root_by=root_obj)

    @staticmethod
    def dependant_roots(first_root: 'Root', second_root: 'Root') -> Optional[Tuple[IExpression, str]]:
        if first_root._root != second_root._root:
            return None
        result = first_root._inside.__truediv__(second_root._inside)
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return (result, 'first')
            return None
        result = second_root._inside.__truediv__(first_root._inside)
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return (result, 'second')
            return None
        return (result, 'second')

    def __iadd__(self, other: Union[IExpression, float, int, str]):
        if other == 0:
            return self
        if isinstance(other, IExpression):
            if isinstance(other, Root):
                if self._root == other._root and self._inside == other._inside:
                    new_coef = self._coefficient + other._coefficient
                    if new_coef == 0:
                        return 0
                    self._coefficient = new_coef
                    return self
                division_result: Optional[IExpression] = Root.dependant_roots(self, other)
                if division_result is not None:
                    root_evaluation = self._root.try_evaluate()
                    if division_result[1] == "first":
                        other_copy = other.__copy__()
                        other_copy._coefficient = Mono(1)
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            return other_copy * (division_result ** (
                                    1 / root_evaluation) + other_copy._coefficient * self._coefficient)
                    else:
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            self_copy = self.__copy__()
                            self_copy._coefficient = Mono(1)
                            return self * (division_result ** (
                                    1 / root_evaluation) + self._coefficient * other._coefficient)
        if isinstance(other, (int, float)):
            other = Mono(other)
        return ExpressionSum((self, other))

    def __isub__(self, other):
        if other == 0:
            return self
        return self.__iadd__(-other)

    def multiply_by_root(self, other: 'Root'):
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            self._coefficient *= other_evaluation
            return self
        if self._root == other._root:
            self._inside *= other._inside
            return self
        else:
            return ExpressionMul((self, other))

    def __imul__(self, other: Union[IExpression, float, int, str]):
        if isinstance(other, (int, float)):
            self._coefficient *= other
            self.simplify()
            return self
        if isinstance(other, str):
            pass
        if isinstance(other, IExpression):
            if isinstance(other, Root):
                return self.multiply_by_root(other)
            else:
                self._coefficient *= other
                return self
        return TypeError(f'Invalid type {type(other)} for multiplying roots.')

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__mul__(other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def __ipow__(self, other: Union[int, float, IExpression]):
        if other == 1:
            return self
        if other == 0:
            return Mono(1)
        root_division = self._root / other
        if isinstance(root_division, IExpression):
            evaluated_division = root_division.try_evaluate()
            if evaluated_division is None:
                self._root = root_division
                return self
        elif isinstance(root_division, (int, float)):
            evaluated_division = root_division
        else:
            raise TypeError(f"Invalid type '{type(other)} when dividing Root objects.'")
        if 0 < evaluated_division < 1:
            return self._inside ** (1 / evaluated_division)
        elif evaluated_division == 1:
            return self._inside
        self._root = evaluated_division
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return ZeroDivisionError('Cannot divide a Root object by 0')
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        else:
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self._coefficient /= other_evaluation
                return self
            if isinstance(other, Root):
                if self._root == other._root:
                    if self == other:
                        return Mono(1)
                elif self._inside == other._inside:
                    my_root_evaluation = self._root.try_evaluate()
                    other_root_evaluation = other._root.try_evaluate()
                    if my_root_evaluation and other_root_evaluation:
                        self._coefficient /= other._coefficient
                        power_difference = 1 / my_root_evaluation - 1 / other_root_evaluation
                        self._root = 1 / power_difference
                        return self
            else:
                return Fraction(self, other)
            return Fraction(self, other)

    def __copy__(self):
        return Root(inside=self._inside, root_by=self._root, coefficient=self._coefficient)

    def __str__(self):
        if self._coefficient == 0:
            return '0'
        if self._coefficient == 1:
            coefficient = ''
        elif self._coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f'{self._coefficient} * '
        root = f'{self._root}^' if self._root != 2 else ''
        return f'{coefficient}{root}√({self._inside})'

    def __eq__(self, other: Union[IExpression, int, float]):
        """ Compare between a Root object and other expressions"""
        if other is None:
            return False
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            print(my_evaluation)
            return my_evaluation == other
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):
                return my_evaluation == other_evaluation
            if (my_evaluation, other_evaluation) == (None, None):
                if isinstance(other, Root):
                    if self._coefficient == other._coefficient and self._inside == other._inside and (self._root == other._root):
                        return True
                    return False
        return False

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def derivative(self):
        if self._inside is None:
            return 0
        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            if my_evaluation < 0:
                warnings.warn('Root evaluated to negative result. Complex Analysis is yet to be supported in this version')
            return 0
        if self._coefficient == 0:
            return 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        inside_evaluation = self._inside.try_evaluate()
        if None not in (coefficient_evaluation, root_evaluation, inside_evaluation):
            return 0
        inside_variables = self._inside.variables
        if None not in (coefficient_evaluation, root_evaluation) and len(inside_variables) == 1:
            new_power = 1 / root_evaluation - 1
            new_root = 1 / new_power
            inside_derivative = self._inside.derivative()
            derivative_coefficient = coefficient_evaluation / root_evaluation
            if new_power > 1:
                monomial = Mono(coefficient=derivative_coefficient, variables_dict={inside_variables: new_power})
                monomial *= inside_derivative
                return monomial
            elif new_power == 0:
                inside_derivative *= derivative_coefficient
                return inside_derivative
            else:
                if new_root == 1:
                    return derivative_coefficient * self._inside
                inside_derivative *= derivative_coefficient
                if new_root < 0:
                    return Fraction(numerator=inside_derivative, denominator=Root(coefficient=1, root_by=abs(new_root), inside=self._inside.__copy__()))
                else:
                    return Root(coefficient=inside_derivative, root_by=new_root, inside=self._inside.__copy__())
        else:
            pass

    def integral(self):
        pass

    def python_syntax(self) -> str:
        """ Returns a string that can be evaluated using the eval() method to actual objects from the class, if
        imported properly
        """
        if isinstance(self._coefficient, Log):
            coefficient_str = self._coefficient.python_syntax()
        else:
            coefficient_str = self._coefficient.__str__()
        if isinstance(self._inside, Log):
            inside_str = self._inside.python_syntax()
        else:
            inside_str = self._inside.__str__()
        if isinstance(self._root, Log):
            root_str = self._root.python_syntax()
        else:
            root_str = self._root.__str__()
        return f'{coefficient_str}*({inside_str}) ** (1/{root_str})'

class Sqrt(Root):

    def __init__(self, inside: Union[IExpression, float, int], coefficient: Union[int, float, IExpression]=Mono(1)):
        super(Sqrt, self).__init__(inside=inside, root_by=2, coefficient=coefficient)


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

