"""Bounded exercise families with known, checked answers."""
import random
from fractions import Fraction
from .formatting import PDFText, format_math, format_polynomial
from .layout import PDFMath


def _math(expression):
    return PDFMath(expression)


def _language(lang):
    if lang != 'en':
        raise ValueError('These exercise generators currently support lang="en" only')


def intersection(get_solution=True, variable='x', lang='en', rng=None, details=False):
    _language(lang)
    rng = random if rng is None else rng
    a = rng.randint(-6, 6)
    c = rng.choice([v for v in range(-6, 7) if v != a])
    x, y = rng.randint(-8, 8), rng.randint(-8, 8)
    b, d = y-a*x, y-c*x
    prompt = (f'Find the intersection of y = {a}*{variable} {b:+d} and '
              f'y = {c}*{variable} {d:+d}. Show your working.')
    answer = (f'Set the expressions equal: ({a-c})*{variable} = {d-b}.\n'
              f'{variable} = {x}; substitute into either equation to get y = {y}.\n'
              f'Intersection: ({x}, {y}).')
    if len(variable) == 1 and variable.isascii() and variable.isalpha():
        prompt = PDFText('Find the intersection of ', _math('y=' + format_polynomial([a, b], variable)),
                         ' and ', _math('y=' + format_polynomial([c, d], variable)),
                         '. Show your working.', plain=prompt)
        answer = PDFText('Set the expressions equal: ',
                         _math(format_polynomial([a-c, 0], variable) + f'={d-b}'),
                         '.\n', _math(f'{variable}={x}'),
                         '; substitute into either equation to get ', _math(f'y={y}'),
                         '.\nIntersection: ', _math(f'({x}, {y})'), '.', plain=answer)
    if get_solution and details:
        return prompt, answer, (a, b, c, d, x, y)
    return (prompt, answer) if get_solution else prompt


def trigonometric(get_solution=True, variable='x', lang='en', rng=None):
    """Sine/cosine equations with special-angle answers in [0,360) degrees."""
    _language(lang)
    rng = random if rng is None else rng
    name = rng.choice(['sin', 'cos'])
    value = rng.choice([-1, Fraction(-1, 2), 0, Fraction(1, 2), 1])
    solutions = {
        'sin': {-1: [270], Fraction(-1, 2): [210, 330], 0: [0, 180], Fraction(1, 2): [30, 150], 1: [90]},
        'cos': {-1: [180], Fraction(-1, 2): [120, 240], 0: [90, 270], Fraction(1, 2): [60, 300], 1: [0]},
    }[name][value]
    prompt = f'Solve {name}({variable}) = {value} for 0 <= {variable} < 360 degrees.'
    answer = f'{variable} = ' + ', '.join(map(str, solutions)) + ' degrees. Use the unit circle; 360 degrees is excluded.'
    if len(variable) == 1 and variable.isascii() and variable.isalpha():
        prompt = PDFText('Solve ', _math(rf'\{name}({variable}) = {format_math(value)}'),
                         ' for ', _math(rf'0\leq {variable}<360^\circ'), '.', plain=prompt)
        answer = PDFText(_math(variable + '=' + ', '.join(rf'{v}^\circ' for v in solutions)),
                         '. Use the unit circle; 360 degrees is excluded.', plain=answer)
    return (prompt, answer) if get_solution else prompt


def logarithmic(get_solution=True, variable='x', lang='en', rng=None):
    """log_base(a*x+b)=k, generated from a positive argument and known root."""
    _language(lang)
    rng = random if rng is None else rng
    base, exponent = rng.choice([2, 3, 5, 10]), rng.randint(0, 3)
    a, solution = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.randint(-8, 8)
    argument = base**exponent
    b = argument-a*solution
    inside = f'{a}*{variable} {b:+d}'
    prompt = f'Solve log_{base}({inside}) = {exponent}. State the domain restriction.'
    answer = (f'Domain: {inside} > 0.\nRewrite: {inside} = {base}^{exponent} = {argument}.\n'
              f'{variable} = {solution}. Check: the logarithm argument is {argument} > 0.')
    if len(variable) == 1 and variable.isascii() and variable.isalpha():
        formatted = format_polynomial([a, b], variable)
        prompt = PDFText('Solve ', _math(rf'\log_{{{base}}}({formatted})={exponent}'),
                         '. State the domain restriction.', plain=prompt)
        answer = PDFText('Domain: ', _math(formatted + '>0'),
                         '.\nRewrite: ', _math(formatted + rf'={base}^{{{exponent}}}={argument}'),
                         '.\n', _math(f'{variable}={solution}'),
                         '. Check: the logarithm argument is ', _math(f'{argument}>0'), '.', plain=answer)
    return (prompt, answer) if get_solution else prompt
