from __future__ import annotations
import math
from math import (
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    asinh, acosh, atanh, sqrt, e, pi, floor, ceil, log,
    log10, log2, exp, erf, erfc, gamma, lgamma, tau
)
import re
import string

cot = lambda x: 1 / tan(x) if tan(x) != 0 else float('nan')
sec = lambda x: 1 / cos(x) if cos(x) != 0 else float('nan')
csc = lambda x: 1 / sin(x) if sin(x) != 0 else float('nan')
acot = lambda x: 1 / atan(x) if x != 0 else math.pi / 2
asec = lambda x: acos(1 / x) if x != 0 else float('nan')
acsc = lambda x: asin(1 / x) if x != 0 else float('nan')

__all__ = [
    'TRIGONOMETRY_CONSTANTS', 'MATHEMATICAL_CONSTANTS',
    'ptn', 'number_pattern', 'allowed_characters', 'pi', 'e', 'tau',
    'cot', 'sec', 'csc', 'acot', 'asec', 'acsc'
]

TRIGONOMETRY_CONSTANTS = {
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'tan': lambda x: tan(x),
    'asin': lambda x: asin(x),
    'acos': lambda x: acos(x),
    'atan': lambda x: atan(x),
    'sinh': lambda x: sinh(x),
    'cosh': lambda x: cosh(x),
    'tanh': lambda x: tanh(x),
    'asinh': lambda x: asinh(x),
    'acosh': lambda x: acosh(x),
    'atanh': lambda x: atanh(x),
    'cot': cot,
    'sec': sec,
    'csc': csc,
    'acot': acot,
    'asec': asec,
    'acsc': acsc
}

MATHEMATICAL_CONSTANTS = {
    'floor': lambda x: floor(x),
    'ceil': lambda x: ceil(x),
    'log': lambda x, base=e: log(x, base),
    'log2': lambda x: log2(x),
    'log10': lambda x: log10(x),
    'ln': lambda x: log(x, e),
    'exp': lambda x: exp(x),
    'w': lambda x: NotImplemented,
    '&#8730;': lambda x: sqrt(x),
    'sqrt': lambda x: sqrt(x),
    'erf': lambda x: erf(x),
    'erfc': lambda x: erfc(x),
    'gamma': lambda x: gamma(x),
    'lgamma': lambda x: lgamma(x),
    'lambert': lambda x: NotImplemented
}

ptn = re.compile(r"a_(?:n|{n-\d})")
number_pattern = r"\d+[.,]?\d*"
allowed_characters = list(string.ascii_lowercase)
allowed_characters.remove('e')
allowed_characters.remove('i')
