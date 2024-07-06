from math import sin, asin, cos, acos, tan, atan
from enum import Enum 

def cot(number):
    return cos(number) / sin(number)


def sec(number):
    return 1 / cos(number)


def asec(number):
    return acos(1 / number)

def csc(number):
    return 1 / sin(number)


def acsc(number):
    return asin(1 / number)

class TrigoMethods(Enum):
    SIN = sin,
    ASIN = asin,
    COS = cos,
    ACOS = acos,
    TAN = tan,
    ATAN = atan,
    COT = atan,
    SEC = sec
    CSC = csc
    ASEC = asec
    ACSC = acsc


