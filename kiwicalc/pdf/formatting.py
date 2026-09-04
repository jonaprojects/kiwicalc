"""Explicit, dependency-free notation helpers for worksheet mathematics.

Plain prose is never parsed as an expression. Math strings use Mathtext syntax.
"""
from fractions import Fraction
from numbers import Integral, Real
import math
import re


def format_math(value):
    """Format a real number exactly where possible, or accept explicit Mathtext."""
    from kiwicalc.expressions.mono import Mono
    from kiwicalc.expressions.poly import Poly
    if isinstance(value, Mono):
        coefficient = value.coefficient
        format_math(coefficient)
        if coefficient == 0:
            return '0'
        factors = []
        for variable, power in sorted((value.variables_dict or {}).items()):
            if not re.fullmatch(r'[A-Za-z]', variable):
                raise ValueError('Monomial variable names must be single ASCII letters')
            exponent = format_math(power)
            if power != 0:
                factors.append(variable if power == 1 else rf'{variable}^{{{exponent}}}')
        prefix = ('-' if coefficient < 0 else '') if factors and abs(coefficient) == 1 else format_math(coefficient)
        return prefix + ''.join(factors)
    if isinstance(value, Poly):
        # Read the terms directly: no simplify(), sorting, or mutation of the input.
        terms = [format_math(term) for term in value.expressions if term.coefficient != 0]
        return ' '.join(term if i == 0 or term.startswith('-') else '+' + term
                        for i, term in enumerate(terms)) or '0'
    if isinstance(value, str):
        if not value.strip():
            raise ValueError('Math expression must not be empty')
        return value
    if isinstance(value, bool):
        raise TypeError('Boolean values are not mathematical expressions')
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        sign = '-' if value < 0 else ''
        return rf'{sign}\frac{{{abs(value.numerator)}}}{{{value.denominator}}}'
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError('Math numbers must be finite')
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        text = str(value)
        if 'e' in text.lower():
            mantissa, exponent = text.lower().split('e')
            return rf'{mantissa}\times 10^{{{int(exponent)}}}'
        return text
    raise TypeError('Expected a real number, Fraction, Mono, Poly, or explicit Mathtext string')


def _equation_text(equation):
    """Render known generated algebra, preserving unsimplified equation sides."""
    from .layout import PDFMath
    if not isinstance(equation, str) or not re.fullmatch(r'[A-Za-z0-9 .,+*/^=()\-]+', equation):
        return equation
    notation = re.sub(r'\^(-?\d+(?:\.\d+)?)', r'^{\1}', equation)
    notation = notation.replace('*', r'\cdot ')
    return PDFText(PDFMath(notation), plain=equation)


def _replace_math(text, token, expression):
    """Replace a known generated field, never attempt to parse prose."""
    from .layout import PDFMath
    parts = list(text.parts) if isinstance(text, PDFText) else [str(text)]
    for index, part in enumerate(parts):
        if not isinstance(part, str):
            continue
        before, separator, after = part.partition(token)
        if separator:
            parts[index:index+1] = [before, PDFMath(expression), after]
            return PDFText(*parts, plain=str(text))
    return text


def _numbered_equation(equation, number, *, role='question'):
    formatted = _equation_text(str(equation))
    from .blocks import PDFParagraph
    return PDFParagraph(formatted, number=number, role=role)


def format_polynomial(coefficients, variable='x'):
    """Format descending-power real coefficients without rounding or unit terms."""
    if not isinstance(variable, str) or not re.fullmatch(r'[a-zA-Z]', variable):
        raise ValueError('variable must be one ASCII letter')
    coefficients = list(coefficients)
    terms = []
    for index, coefficient in enumerate(coefficients):
        if not isinstance(coefficient, Real) or isinstance(coefficient, bool):
            raise TypeError('Polynomial coefficients must be real numbers')
        format_math(coefficient)  # Validate even coefficients that simplify away.
        if coefficient == 0:
            continue
        power = len(coefficients)-index-1
        magnitude = abs(coefficient)
        term = '' if power and magnitude == 1 else format_math(magnitude)
        if power:
            term += variable if power == 1 else rf'{variable}^{{{power}}}'
        sign = '-' if coefficient < 0 else ('+' if terms else '')
        terms.append(sign + term)
    return ' '.join(terms) or '0'


class PDFText(str):
    """Mixed prose and PDFMath segments, retaining a plain-text representation.

    Example: PDFText('Solve ', PDFMath(r'x^2=4'), ' for x.').
    The optional plain argument preserves legacy generator text for callers.
    """
    def __new__(cls, *parts, plain=None):
        from .layout import PDFMath
        if any(not isinstance(part, (str, PDFMath)) for part in parts):
            raise TypeError('PDFText parts must be strings or PDFMath objects')
        text = ''.join(part if isinstance(part, str) else str(part.expression) for part in parts)
        instance = super().__new__(cls, text if plain is None else plain)
        instance.parts = tuple(parts)
        return instance

    def numbered(self, number, *, role='question'):
        from .blocks import PDFParagraph
        return PDFParagraph(self, number=number, role=role)


__all__ = ['PDFText', 'format_math', 'format_polynomial']
