from __future__ import annotations
from enum import Enum
import math
from math import (
    sin, cos, tan, asin, acos, atan,
    sinh, cosh, tanh, asinh, acosh, atanh
)
import operator
from typing import Callable
from kiwicalc.core.constants import (
    TRIGONOMETRY_CONSTANTS, cot, sec, csc, acot, asec, acsc
)

class Operator:
    __slots__ = ['__sign', '__method']

    def __init__(self, sign: str, method: Callable):
        self.__sign = sign
        self.__method = method

    @property
    def sign(self) -> str:
        return self.__sign

    @property
    def method(self):
        return self.__method

    def __str__(self):
        return self.__sign

class GreaterThan(Operator):

    def __init__(self):
        super(GreaterThan, self).__init__('>', operator.gt)

class LessThan(Operator):

    def __init__(self):
        super(LessThan, self).__init__('<', operator.lt)

class GreaterOrEqual(Operator):

    def __init__(self):
        super(GreaterOrEqual, self).__init__('>=', operator.ge)

class LessOrEqual(Operator):

    def __init__(self):
        super(LessOrEqual, self).__init__('<=', operator.le)


def range_operator_from_string(operator_string: str) -> Operator:
    operators = {
        '>': GreaterThan,
        '>=': GreaterOrEqual,
        '<': LessThan,
        '<=': LessOrEqual,
    }
    try:
        return operators[operator_string.strip()]()
    except KeyError as exc:
        raise ValueError(f'Unsupported range operator: {operator_string!r}') from exc

class TrigoMethods(Enum):
    SIN = (sin,)
    ASIN = (asin,)
    COS = (cos,)
    ACOS = (acos,)
    TAN = (tan,)
    ATAN = (atan,)
    COT = (cot,)
    SEC = (sec,)
    CSC = (csc,)
    ACOT = (acot,)
    ASEC = (asec,)
    ACSC = (acsc,)

def _TrigoMethodFromString(method_string: str):
    """ Method for internal use. DO NOT USE IT IF YOU'RE NOT IN THE KIWICALC DEVELOPERS TEAM"""
    try:
        method_string = method_string.strip().upper()
        return operator.attrgetter(method_string)(TrigoMethods)
    except AttributeError:
        raise AttributeError(f"Unsupported trigonometric method:'{method_string}'")

GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL = GreaterThan(), GreaterOrEqual(), LessThan(), LessOrEqual()
