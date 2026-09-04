"""Deterministic core-algebra exercises with exact, render-ready solutions."""
from __future__ import annotations

import random

from .formatting import PDFText, format_polynomial
from .layout import PDFMath
from .worksheet import PDFExercise


DIFFICULTIES = ('easy', 'medium', 'hard')
ALGEBRA_EXERCISE_TYPES = (
    'simplify', 'expand', 'factor', 'complete_square', 'substitution',
    'linear_inequality', 'absolute_value', 'exponent_laws', 'rational',
    'radical', 'rearrange',
)


def _settings(difficulty, seed, rng):
    if difficulty not in DIFFICULTIES:
        raise ValueError(f'difficulty must be one of {", ".join(DIFFICULTIES)}')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError('seed must be an integer or None')
    if rng is not None and seed is not None:
        raise ValueError('seed and the internal random generator cannot both be supplied')
    return rng or random.Random(seed)


def _nonzero(rng, low, high):
    return rng.choice([value for value in range(low, high + 1) if value])


def _expression(terms, variable='x'):
    """Format ordered, intentionally unsimplified (coefficient, power) terms."""
    pieces = []
    for coefficient, power in terms:
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        body = '' if power and magnitude == 1 else str(magnitude)
        if power:
            body += variable if power == 1 else rf'{variable}^{{{power}}}'
        if not pieces:
            pieces.append(('-' if coefficient < 0 else '') + body)
        else:
            pieces.append((' - ' if coefficient < 0 else ' + ') + body)
    return ''.join(pieces) or '0'


def _linear(coefficient, constant, variable='x'):
    return format_polynomial([coefficient, constant], variable)


def _binomial(a, b, variable='x'):
    return '(' + _linear(a, b, variable) + ')'


def _factor(root, variable='x'):
    return f'({variable}{"-" if root >= 0 else "+"}{abs(root)})'


class PDFAlgebraExercise(PDFExercise):
    """Base class exposing stable metadata for checking and reuse."""
    __slots__ = ('kind', 'difficulty', 'data')

    def __init__(self, prompt, kind, solution, difficulty, data):
        super().__init__(prompt, 'algebra', kind, solution=solution)
        self.kind = kind
        self.difficulty = difficulty
        self.data = data


class PDFSimplifyExpression(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        limit = {'easy': 6, 'medium': 9, 'hard': 12}[difficulty]
        if difficulty == 'easy':
            terms = [(_nonzero(rng, -limit, limit), 1), (_nonzero(rng, -limit, limit), 1)]
            coefficients = [sum(value for value, _ in terms), 0]
        elif difficulty == 'medium':
            terms = [(_nonzero(rng, -limit, limit), 1), (rng.randint(-limit, limit), 0),
                     (_nonzero(rng, -limit, limit), 1), (rng.randint(-limit, limit), 0)]
            coefficients = [sum(value for value, power in terms if power == 1),
                            sum(value for value, power in terms if power == 0)]
        else:
            terms = [(_nonzero(rng, -limit, limit), 2), (_nonzero(rng, -limit, limit), 1),
                     (rng.randint(-limit, limit), 0), (_nonzero(rng, -limit, limit), 2),
                     (_nonzero(rng, -limit, limit), 1), (rng.randint(-limit, limit), 0)]
            coefficients = [sum(v for v, p in terms if p == 2),
                            sum(v for v, p in terms if p == 1),
                            sum(v for v, p in terms if p == 0)]
        source, answer = _expression(terms), format_polynomial(coefficients)
        prompt = PDFText('Simplify: ', PDFMath(source), '.', plain=f'Simplify: {source}.')
        solution = PDFText('Combine like terms: ', PDFMath(answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'simplify', solution, difficulty,
                         {'terms': tuple(terms), 'coefficients': tuple(coefficients)})


class PDFExpandExpression(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        limit = {'easy': 5, 'medium': 7, 'hard': 10}[difficulty]
        if difficulty == 'easy':
            a, b = _nonzero(rng, -limit, limit), _nonzero(rng, -limit, limit)
            source = f'{a}{_binomial(1, b)}'
            coefficients, factors = (a, a*b), ((a, 0), (1, b))
        else:
            a = 1 if difficulty == 'medium' else _nonzero(rng, -4, 4)
            c = 1 if difficulty == 'medium' else _nonzero(rng, -4, 4)
            b, d = _nonzero(rng, -limit, limit), _nonzero(rng, -limit, limit)
            source = _binomial(a, b) + _binomial(c, d)
            coefficients, factors = (a*c, a*d+b*c, b*d), ((a, b), (c, d))
        answer = format_polynomial(coefficients)
        prompt = PDFText('Expand and simplify: ', PDFMath(source), '.', plain=f'Expand: {source}.')
        solution = PDFText('After distributing: ', PDFMath(answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'expand', solution, difficulty,
                         {'factors': factors, 'coefficients': tuple(coefficients)})


class PDFFactorPolynomial(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        bound = {'easy': 5, 'medium': 8, 'hard': 10}[difficulty]
        first = _nonzero(rng, -bound, bound)
        second = _nonzero(rng, -bound, bound)
        while second == first:
            second = _nonzero(rng, -bound, bound)
        scale = 1 if difficulty == 'easy' else rng.randint(1, 3 if difficulty == 'medium' else 6)
        coefficients = (scale, -scale*(first+second), scale*first*second)
        source = format_polynomial(coefficients)
        answer = ('' if scale == 1 else str(scale)) + _factor(first) + _factor(second)
        prompt = PDFText('Factor completely: ', PDFMath(source), '.', plain=f'Factor: {source}.')
        solution = PDFText('The zeros are ', PDFMath(f'{first}, {second}'), ', so ',
                           PDFMath(source + '=' + answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'factor', solution, difficulty,
                         {'roots': (first, second), 'scale': scale, 'coefficients': coefficients})


class PDFCompleteSquare(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        bound = {'easy': 4, 'medium': 7, 'hard': 10}[difficulty]
        shift, remainder = _nonzero(rng, -bound, bound), rng.randint(-bound, bound)
        coefficients = (1, 2*shift, shift*shift+remainder)
        source = format_polynomial(coefficients)
        completed = _factor(-shift) + '^2' + (f'+{remainder}' if remainder > 0 else str(remainder) if remainder < 0 else '')
        completed = completed.replace('^2', '^{2}')
        prompt = PDFText('Complete the square: ', PDFMath(source), '.', plain=f'Complete the square: {source}.')
        solution = PDFText(PDFMath(source + '=' + completed), '.', plain=completed) if with_solution else None
        super().__init__(prompt, 'complete_square', solution, difficulty,
                         {'shift': shift, 'remainder': remainder, 'coefficients': coefficients})


class PDFSubstitution(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        degree = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        coefficients = tuple(_nonzero(rng, -6, 6) for _ in range(degree+1))
        value = _nonzero(rng, -4, 4)
        result = sum(coefficient*value**(degree-index) for index, coefficient in enumerate(coefficients))
        expression = format_polynomial(coefficients)
        prompt = PDFText('Evaluate ', PDFMath(expression), ' when ', PDFMath(f'x={value}'), '.',
                         plain=f'Evaluate {expression} when x={value}.')
        solution = PDFText('Substitution gives ', PDFMath(str(result)), '.', plain=str(result)) if with_solution else None
        super().__init__(prompt, 'substitution', solution, difficulty,
                         {'coefficients': coefficients, 'value': value, 'result': result})


class PDFLinearInequality(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        if difficulty == 'easy':
            coefficient = rng.randint(1, 8)
        elif difficulty == 'hard':
            coefficient = -rng.randint(2, 10)
        else:
            coefficient = _nonzero(rng, -8, 8)
        constant, boundary = rng.randint(-10, 10), rng.randint(-8, 8)
        inclusive = rng.choice((False, True))
        right = coefficient*boundary+constant
        relation = r'\leq' if inclusive else '<'
        flipped = (r'\geq' if inclusive else '>') if coefficient < 0 else relation
        source = _linear(coefficient, constant) + relation + str(right)
        answer = 'x' + flipped + str(boundary)
        prompt = PDFText('Solve the inequality: ', PDFMath(source), '.', plain=f'Solve: {source}.')
        note = ' Dividing by a negative reverses the inequality.' if coefficient < 0 else ''
        solution = PDFText(PDFMath(answer), '.', note, plain=answer) if with_solution else None
        super().__init__(prompt, 'linear_inequality', solution, difficulty,
                         {'coefficient': coefficient, 'constant': constant, 'boundary': boundary,
                          'inclusive': inclusive, 'right': right})


class PDFAbsoluteValueEquation(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        scale = 1 if difficulty == 'easy' else rng.randint(2, 5 if difficulty == 'medium' else 8)
        center, radius = rng.randint(-8, 8), rng.randint(1, 8)
        inside = _linear(scale, -scale*center)
        right = scale*radius
        roots = tuple(sorted((center-radius, center+radius)))
        source = rf'\left|{inside}\right|={right}'
        answer = 'x=' + ', '.join(map(str, roots))
        prompt = PDFText('Solve: ', PDFMath(source), '.', plain=f'Solve: |{inside}|={right}.')
        solution = PDFText('Use both cases to obtain ', PDFMath(answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'absolute_value', solution, difficulty,
                         {'scale': scale, 'center': center, 'radius': radius, 'roots': roots})


class PDFExponentLaws(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        top = {'easy': 6, 'medium': 10, 'hard': 14}[difficulty]
        first, second = rng.randint(1, top), rng.randint(1, top)
        divisor = rng.randint(1, first+second-1)
        exponent = first+second-divisor
        source = rf'\frac{{x^{{{first}}}x^{{{second}}}}}{{x^{{{divisor}}}}}'
        answer = 'x' if exponent == 1 else rf'x^{{{exponent}}}'
        prompt = PDFText('Simplify: ', PDFMath(source), '.', plain='Simplify using exponent laws.')
        solution = PDFText('Add numerator exponents and subtract the denominator exponent: ',
                           PDFMath(answer), ', where ', PDFMath(r'x\neq0'), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'exponent_laws', solution, difficulty,
                         {'exponents': (first, second, divisor), 'result_exponent': exponent})


class PDFRationalEquation(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        denominator_value = rng.randint(1, {'easy': 4, 'medium': 7, 'hard': 10}[difficulty])
        quotient = _nonzero(rng, -6, 6)
        numerator = denominator_value*quotient
        excluded = rng.randint(-8, 8)
        root = excluded+quotient
        denominator = _linear(1, -excluded)
        source = rf'\frac{{{numerator}}}{{{denominator}}}={denominator_value}'
        prompt = PDFText('Solve and state the restriction: ', PDFMath(source), '.', plain=f'Solve {source}.')
        solution = PDFText('Restriction: ', PDFMath(rf'x\neq{excluded}'), '. Solving gives ',
                           PDFMath(f'x={root}'), '.', plain=f'x={root}') if with_solution else None
        super().__init__(prompt, 'rational', solution, difficulty,
                         {'numerator': numerator, 'right': denominator_value,
                          'excluded': excluded, 'root': root})


class PDFRadicalEquation(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        right = rng.randint(1, {'easy': 5, 'medium': 8, 'hard': 12}[difficulty])
        offset = rng.randint(-10, 10)
        root = right*right-offset
        inside = _linear(1, offset)
        source = rf'\sqrt{{{inside}}}={right}'
        prompt = PDFText('Solve and check: ', PDFMath(source), '.', plain=f'Solve sqrt({inside})={right}.')
        solution = PDFText('Square both sides: ', PDFMath(f'{inside}={right*right}'), ', so ',
                           PDFMath(f'x={root}'), '.', plain=f'x={root}') if with_solution else None
        super().__init__(prompt, 'radical', solution, difficulty,
                         {'offset': offset, 'right': right, 'root': root})


class PDFRearrangeFormula(PDFAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        choices = {
            'easy': [('y=mx+b', 'x', r'x=\frac{y-b}{m}')],
            'medium': [(r'A=\frac{bh}{2}', 'h', r'h=\frac{2A}{b}'),
                       ('v=u+at', 'a', r'a=\frac{v-u}{t}')],
            'hard': [('P=2(l+w)', 'w', r'w=\frac{P}{2}-l'),
                     (r's=ut+\frac{at^2}{2}', 'a', r'a=\frac{2(s-ut)}{t^2}')],
        }[difficulty]
        formula, subject, answer = rng.choice(choices)
        prompt = PDFText('Make ', PDFMath(subject), ' the subject of ', PDFMath(formula), '.',
                         plain=f'Make {subject} the subject of {formula}.')
        solution = PDFText(PDFMath(answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'rearrange', solution, difficulty,
                         {'formula': formula, 'subject': subject, 'answer': answer})


_FACTORIES = {
    'simplify': PDFSimplifyExpression,
    'expand': PDFExpandExpression,
    'factor': PDFFactorPolynomial,
    'complete_square': PDFCompleteSquare,
    'substitution': PDFSubstitution,
    'linear_inequality': PDFLinearInequality,
    'absolute_value': PDFAbsoluteValueEquation,
    'exponent_laws': PDFExponentLaws,
    'rational': PDFRationalEquation,
    'radical': PDFRadicalEquation,
    'rearrange': PDFRearrangeFormula,
}

_ALIASES = {
    'simplifying': 'simplify', 'expanding': 'expand', 'factoring': 'factor',
    'evaluate': 'substitution', 'inequality': 'linear_inequality',
    'absolute': 'absolute_value', 'exponents': 'exponent_laws',
    'rational_equation': 'rational', 'radical_equation': 'radical',
    'rearrange_formula': 'rearrange',
}


def algebra_exercise(kind, *, difficulty='medium', seed=None, with_solution=True, _rng=None):
    """Create a core-algebra exercise by a friendly name."""
    if not isinstance(kind, str):
        raise TypeError('kind must be text')
    key = kind.strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        raise ValueError(f'Unknown algebra exercise {kind!r}; choose from {", ".join(ALGEBRA_EXERCISE_TYPES)}') from exc
    return factory(with_solution=with_solution, difficulty=difficulty, seed=seed, _rng=_rng)


__all__ = [
    'PDFAlgebraExercise', 'PDFSimplifyExpression', 'PDFExpandExpression',
    'PDFFactorPolynomial', 'PDFCompleteSquare', 'PDFSubstitution',
    'PDFLinearInequality', 'PDFAbsoluteValueEquation', 'PDFExponentLaws',
    'PDFRationalEquation', 'PDFRadicalEquation', 'PDFRearrangeFormula',
    'ALGEBRA_EXERCISE_TYPES', 'algebra_exercise',
]
