from math import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, sqrt, e, pi, floor, ceil, log, \
    log10, log2, exp, erf, erfc, gamma, lgamma, tau, comb, degrees, radians

from typing import Union 

def ln(x) -> float:
    """ ln(x) = log_e(x)"""
    return log(x, e)

def factorial(number: Union[int, float]):
    return gamma(number + 1)