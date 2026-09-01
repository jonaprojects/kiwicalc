from __future__ import annotations
from typing import Union, Any

def create(expression: str, dtype: str = 'poly'):
    from kiwicalc.expressions.poly import Poly
    from kiwicalc.expressions.log import Log, Ln
    from kiwicalc.expressions.trigonometry import TrigoExprs
    from kiwicalc.expressions.roots import Root
    from kiwicalc.expressions.special import Factorial
    if dtype == 'poly':
        return Poly(expression)
    elif dtype == 'log':
        return Log(expression)
    elif dtype == 'ln':
        return Ln(expression)
    elif dtype == 'trigo':
        return TrigoExprs(expression)
    elif dtype == 'root':
        return Root(expression)
    elif dtype == 'factorial':
        return Factorial(expression)
    else:
        raise ValueError(f"Invalid parameter 'dtype': {dtype}")

def create_from_dict(given_dict: dict):
    from kiwicalc.expressions.mono import Mono
    from kiwicalc.expressions.poly import FastPoly, Poly
    from kiwicalc.expressions.trigonometry import TrigoExpr, TrigoExprs
    from kiwicalc.expressions.log import Log, Ln
    from kiwicalc.expressions.fractions import Fraction
    from kiwicalc.expressions.roots import Root
    from kiwicalc.expressions.special import Abs, Exponent, Factorial
    from kiwicalc.expressions.sum import ExpressionSum
    from kiwicalc.expressions.mul import ExpressionMul
    if isinstance(given_dict, (int, float)):
        return Mono(given_dict)
    expression_type = given_dict['type'].lower()
    if expression_type == 'mono':
        return Mono.from_dict(given_dict)
    elif expression_type == 'poly':
        return Poly.from_dict(given_dict)
    elif expression_type == 'fastpoly':
        return FastPoly.from_dict(given_dict)
    elif expression_type == 'trigoexpr':
        return TrigoExpr.from_dict(given_dict)
    elif expression_type == 'trigoexprs':
        return TrigoExprs.from_dict(given_dict)
    elif expression_type == 'log':
        return Log.from_dict(given_dict)
    elif expression_type == 'ln':
        return Ln.from_dict(given_dict)
    elif expression_type == 'fraction':
        return Fraction.from_dict(given_dict)
    elif expression_type == 'root':
        return Root.from_dict(given_dict)
    elif expression_type == 'abs':
        return Abs.from_dict(given_dict)
    elif expression_type == 'exponent':
        return Exponent.from_dict(given_dict)
    elif expression_type == 'factorial':
        return Factorial.from_dict(given_dict)
    elif expression_type == 'expressionsum':
        return ExpressionSum.from_dict(given_dict)
    elif expression_type == 'expressionmul':
        return ExpressionMul.from_dict(given_dict)
    raise ValueError(f"Unknown expression type '{expression_type}'")
