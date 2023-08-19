"""
Constants and relevant enums
"""

# Lowercase letters, but not 'i' and 'e' since they have special meaning
ALLOWED_CHARACTERS = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


TRIGONOMETRY_CONSTANTS = {
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'tan': lambda x: tan(x),
    'cot': lambda x: 1 / tan(x),
    'sec': lambda x: 1 / cos(x),
    'csc': lambda x: 1 / sin(x),
    'cosec': lambda x: 1 / sin(x),
    'asin': lambda x: asin(x),
    'acos': lambda x: acos(x),
    'atan': lambda x: atan(x),
    'sinh': lambda x: sinh(x),
    'cosh': lambda x: cosh(x),
    'tanh': lambda x: tanh(x),
    'asinh': lambda x: asinh(x),
    'acosh': lambda x: acosh(x),
    'atanh': lambda x: atanh(x)
}
MATHEMATICAL_CONSTANTS = {
    'e': e,
    'pi': pi,
    'tau': tau,
    'log': lambda x, base: log(x=x, base=base),
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
