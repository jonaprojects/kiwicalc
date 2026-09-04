"""Deterministic sequence and series exercises with exact answers."""
from __future__ import annotations

from fractions import Fraction
import math
import random

from .formatting import PDFText, format_math, format_polynomial
from .layout import PDFMath
from .worksheet import PDFExercise


DIFFICULTIES = ('easy', 'medium', 'hard')
SEQUENCE_SERIES_EXERCISE_TYPES = (
    'identify_sequence', 'arithmetic_next_terms', 'arithmetic_nth_term',
    'arithmetic_difference', 'arithmetic_sum', 'arithmetic_missing_term',
    'geometric_next_terms', 'geometric_nth_term', 'geometric_ratio',
    'geometric_sum', 'infinite_geometric_sum', 'recursive_sequence',
    'fibonacci', 'sigma_evaluation', 'sequence_limit',
    'convergence_classification', 'p_series', 'geometric_series_test',
    'alternating_series', 'telescoping_series',
    'elementary_limit', 'euler_limit', 'removable_limit',
    'standard_trig_limit',
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
    return rng.choice([value for value in range(low, high+1) if value])


def _number(value):
    return format_math(value if isinstance(value, Fraction) else Fraction(value))


def _list(values):
    return r',\ '.join(_number(value) for value in values)


def _power_base(value):
    expression = _number(value)
    return rf'\left({expression}\right)' if value < 0 or isinstance(value, Fraction) else expression


def _arithmetic_term(first, difference, index):
    return first+(index-1)*difference


def _geometric_term(first, ratio, index):
    return first*ratio**(index-1)


def _arithmetic_formula(first, difference):
    return rf'a_n={_number(first)}+({_number(difference)})(n-1)'


def _geometric_formula(first, ratio):
    coefficient = '-' if first == -1 else '' if first == 1 else _number(first)+r'\cdot '
    return rf'a_n={coefficient}{_power_base(ratio)}^{{n-1}}'


def _recurrence(multiplier, constant):
    coefficient = '-' if multiplier == -1 else '' if multiplier == 1 else str(multiplier)
    tail = '' if constant == 0 else f'{constant:+d}'
    return rf'a_n={coefficient}a_{{n-1}}{tail}'


def _polynomial_coefficients(rng, degree, maximum, *, one_sign=False):
    """Return highest-power-first integer coefficients of an exact degree."""
    leading = _nonzero(rng, -maximum, maximum)
    if not one_sign:
        return (leading, *(rng.randint(-maximum, maximum) for _ in range(degree)))
    # A denominator whose coefficients all have the leading coefficient's
    # sign cannot vanish at a positive integer n.
    sign = 1 if leading > 0 else -1
    return (leading, *(sign*rng.randint(0, maximum) for _ in range(degree)))


def _pi_multiple(value):
    value = Fraction(value)
    if value == 0:
        return '0'
    sign = '-' if value < 0 else ''
    value = abs(value)
    numerator = '' if value.numerator == 1 else str(value.numerator)
    if value.denominator == 1:
        return rf'{sign}{numerator}\pi'
    return rf'{sign}\frac{{{numerator}\pi}}{{{value.denominator}}}'


def _e_power(exponent):
    exponent = Fraction(exponent)
    if exponent == 0:
        return '1'
    if exponent == 1:
        return r'\mathrm{e}'
    return rf'\mathrm{{e}}^{{{_number(exponent)}}}'


def _linear_factor(point):
    if point == 0:
        return 'x'
    return f'x-{point}' if point > 0 else f'x+{abs(point)}'


class PDFSequenceSeriesExercise(PDFExercise):
    """Base class exposing stable generated data for checking and reuse."""
    __slots__ = ('kind', 'difficulty', 'data')

    def __init__(self, prompt, kind, solution, difficulty, data):
        category = 'series' if kind in {
            'arithmetic_sum', 'geometric_sum', 'infinite_geometric_sum',
            'sigma_evaluation', 'p_series', 'geometric_series_test',
            'alternating_series', 'telescoping_series',
        } else 'limit' if kind in {
            'elementary_limit', 'euler_limit', 'removable_limit',
            'standard_trig_limit',
        } else 'sequence'
        super().__init__(prompt, category, kind, solution=solution)
        self.kind = kind
        self.difficulty = difficulty
        self.data = data


class PDFIdentifySequence(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        sequence_type = rng.choice(('arithmetic', 'geometric'))
        first = _nonzero(rng, -8, 8)
        if sequence_type == 'arithmetic':
            parameter = _nonzero(rng, -6, 6)
            terms = tuple(_arithmetic_term(first, parameter, index) for index in range(1, 6))
            label = 'common difference'
        else:
            choices = (-3, -2, Fraction(-1, 2), Fraction(1, 2), 2, 3)
            parameter = rng.choice(choices if difficulty != 'easy' else (2, 3, -2))
            terms = tuple(_geometric_term(first, parameter, index) for index in range(1, 6))
            label = 'common ratio'
        prompt = PDFText('Classify the sequence and state its common difference or ratio: ',
                         PDFMath(_list(terms)), '.')
        solution = PDFText(f'The sequence is {sequence_type}; its {label} is ',
                           PDFMath(_number(parameter)), '.') if with_solution else None
        super().__init__(prompt, 'identify_sequence', solution, difficulty,
                         {'sequence_type': sequence_type, 'first': first, 'parameter': parameter,
                          'terms': terms, 'result': (sequence_type, parameter)})


class PDFArithmeticNextTerms(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, difference = _nonzero(rng, -12, 12), _nonzero(rng, -8, 8)
        shown = tuple(_arithmetic_term(first, difference, index) for index in range(1, 5))
        count = {'easy': 2, 'medium': 3, 'hard': 4}[difficulty]
        result = tuple(_arithmetic_term(first, difference, index) for index in range(5, 5+count))
        prompt = PDFText(f'Write the next {count} terms: ', PDFMath(_list(shown)), ', ', PDFMath(r'\ldots'))
        solution = PDFText('The common difference is ', PDFMath(str(difference)), ', so the next terms are ',
                           PDFMath(_list(result)), '.') if with_solution else None
        super().__init__(prompt, 'arithmetic_next_terms', solution, difficulty,
                         {'first': first, 'difference': difference, 'shown': shown, 'result': result})


class PDFArithmeticNthTerm(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, difference = _nonzero(rng, -10, 10), _nonzero(rng, -7, 7)
        index = rng.randint(*{'easy': (5, 12), 'medium': (12, 30), 'hard': (30, 80)}[difficulty])
        result = _arithmetic_term(first, difference, index)
        prompt = PDFText('For the arithmetic sequence with ', PDFMath(rf'a_1={first}'), ' and ',
                         PDFMath(rf'd={difference}'), ', find ', PDFMath(rf'a_{{{index}}}'), '.')
        solution = PDFText(PDFMath(_arithmetic_formula(first, difference)), ', so ',
                           PDFMath(rf'a_{{{index}}}={result}'), '.') if with_solution else None
        super().__init__(prompt, 'arithmetic_nth_term', solution, difficulty,
                         {'first': first, 'difference': difference, 'index': index, 'result': result})


class PDFArithmeticDifference(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, difference = _nonzero(rng, -9, 9), _nonzero(rng, -8, 8)
        terms = tuple(_arithmetic_term(first, difference, index) for index in range(1, 5))
        prompt = PDFText('Find the common difference of ', PDFMath(_list(terms)), ', ', PDFMath(r'\ldots'))
        solution = PDFText(PDFMath(rf'd=a_2-a_1={terms[1]}-({terms[0]})={difference}'), '.') if with_solution else None
        super().__init__(prompt, 'arithmetic_difference', solution, difficulty,
                         {'first': first, 'terms': terms, 'result': difference})


class PDFArithmeticSum(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, difference = _nonzero(rng, -8, 10), _nonzero(rng, -6, 8)
        count = rng.randint(*{'easy': (5, 10), 'medium': (10, 25), 'hard': (25, 60)}[difficulty])
        last = _arithmetic_term(first, difference, count)
        result = Fraction(count*(first+last), 2)
        prompt = PDFText('Find the sum of the first ', str(count), ' terms of the arithmetic sequence with ',
                         PDFMath(rf'a_1={first}'), ' and ', PDFMath(rf'd={difference}'), '.')
        solution = PDFText(PDFMath(rf'S_{{{count}}}=\frac{{{count}}}{{2}}({first}+{last})={_number(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'arithmetic_sum', solution, difficulty,
                         {'first': first, 'difference': difference, 'count': count,
                          'last': last, 'result': result})


class PDFArithmeticMissingTerm(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, difference = _nonzero(rng, -10, 10), _nonzero(rng, -7, 7)
        position = rng.randint(2, 4)
        terms = tuple(_arithmetic_term(first, difference, index) for index in range(1, 6))
        display = tuple('x' if index == position-1 else value for index, value in enumerate(terms))
        prompt = PDFText('Find the missing term so the sequence is arithmetic: ',
                         PDFMath(r',\ '.join(str(value) for value in display)), '.')
        solution = PDFText(PDFMath(rf'x={terms[position-1]}'), '.') if with_solution else None
        super().__init__(prompt, 'arithmetic_missing_term', solution, difficulty,
                         {'first': first, 'difference': difference, 'position': position,
                          'terms': terms, 'result': terms[position-1]})


class PDFGeometricNextTerms(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _nonzero(rng, -8, 8)
        ratio = rng.choice((-3, -2, 2, 3) if difficulty != 'hard' else
                           (-3, -2, Fraction(-1, 2), Fraction(1, 2), 2, 3))
        shown = tuple(_geometric_term(first, ratio, index) for index in range(1, 5))
        count = {'easy': 2, 'medium': 3, 'hard': 4}[difficulty]
        result = tuple(_geometric_term(first, ratio, index) for index in range(5, 5+count))
        prompt = PDFText(f'Write the next {count} terms: ', PDFMath(_list(shown)), ', ', PDFMath(r'\ldots'))
        solution = PDFText('The common ratio is ', PDFMath(_number(ratio)), ', so the next terms are ',
                           PDFMath(_list(result)), '.') if with_solution else None
        super().__init__(prompt, 'geometric_next_terms', solution, difficulty,
                         {'first': first, 'ratio': ratio, 'shown': shown, 'result': result})


class PDFGeometricNthTerm(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _nonzero(rng, -6, 6)
        ratio = rng.choice((-3, -2, 2, 3))
        index = rng.randint(*{'easy': (4, 7), 'medium': (6, 10), 'hard': (9, 13)}[difficulty])
        result = _geometric_term(first, ratio, index)
        prompt = PDFText('For the geometric sequence with ', PDFMath(rf'a_1={first}'), ' and ',
                         PDFMath(rf'r={ratio}'), ', find ', PDFMath(rf'a_{{{index}}}'), '.')
        solution = PDFText(PDFMath(_geometric_formula(first, ratio)), ', so ',
                           PDFMath(rf'a_{{{index}}}={_number(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'geometric_nth_term', solution, difficulty,
                         {'first': first, 'ratio': ratio, 'index': index, 'result': result})


class PDFGeometricRatio(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _nonzero(rng, -8, 8)
        ratio = rng.choice((-3, -2, 2, 3) if difficulty != 'hard' else
                           (Fraction(-1, 2), Fraction(1, 2), -3, 3))
        terms = tuple(_geometric_term(first, ratio, index) for index in range(1, 5))
        prompt = PDFText('Find the common ratio of ', PDFMath(_list(terms)), ', ', PDFMath(r'\ldots'))
        solution = PDFText(PDFMath(rf'r=\frac{{a_2}}{{a_1}}={_number(ratio)}'), '.') if with_solution else None
        super().__init__(prompt, 'geometric_ratio', solution, difficulty,
                         {'first': first, 'terms': terms, 'result': ratio})


class PDFGeometricSum(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, ratio = _nonzero(rng, -6, 6), rng.choice((-2, 2, 3))
        count = rng.randint(*{'easy': (3, 5), 'medium': (5, 8), 'hard': (8, 11)}[difficulty])
        result = Fraction(first*(1-ratio**count), 1-ratio)
        prompt = PDFText('Find the sum of the first ', str(count), ' terms of the geometric sequence with ',
                         PDFMath(rf'a_1={first}'), ' and ', PDFMath(rf'r={ratio}'), '.')
        solution = PDFText(PDFMath(rf'S_{{{count}}}=a_1\frac{{1-r^{{{count}}}}}{{1-r}}={_number(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'geometric_sum', solution, difficulty,
                         {'first': first, 'ratio': ratio, 'count': count, 'result': result})


class PDFInfiniteGeometricSum(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _nonzero(rng, -10, 10)
        denominator = rng.randint(2, 6)
        ratio = Fraction(rng.choice((-1, 1))*rng.randint(1, denominator-1), denominator)
        result = Fraction(first)/(1-ratio)
        prompt = PDFText('Find the sum to infinity for the geometric series with ',
                         PDFMath(rf'a_1={first}'), ' and ', PDFMath(rf'r={_number(ratio)}'), '.')
        solution = PDFText(PDFMath(rf'S_\infty=\frac{{a_1}}{{1-r}}={_number(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'infinite_geometric_sum', solution, difficulty,
                         {'first': first, 'ratio': ratio, 'result': result})


class PDFRecursiveSequence(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _nonzero(rng, -6, 6)
        multiplier = rng.choice((1, 2, -1) if difficulty != 'hard' else (-2, -1, 1, 2))
        constant = rng.randint(-5, 5)
        index = {'easy': 4, 'medium': 5, 'hard': 6}[difficulty]
        terms = [first]
        while len(terms) < index:
            terms.append(multiplier*terms[-1]+constant)
        recurrence = _recurrence(multiplier, constant)
        prompt = PDFText('Given ', PDFMath(rf'a_1={first}'), ' and ', PDFMath(recurrence),
                         ', find ', PDFMath(rf'a_{{{index}}}'), '.')
        solution = PDFText('The generated terms are ', PDFMath(_list(terms)),
                           ', so ', PDFMath(rf'a_{{{index}}}={terms[-1]}'), '.') if with_solution else None
        super().__init__(prompt, 'recursive_sequence', solution, difficulty,
                         {'first': first, 'multiplier': multiplier, 'constant': constant,
                          'index': index, 'terms': tuple(terms), 'result': terms[-1]})


class PDFFibonacciSequence(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, second = rng.randint(0, 3), rng.randint(1, 4)
        index = rng.randint(*{'easy': (5, 7), 'medium': (7, 10), 'hard': (10, 14)}[difficulty])
        terms = [first, second]
        while len(terms) < index:
            terms.append(terms[-1]+terms[-2])
        prompt = PDFText('A Fibonacci-type sequence begins ', PDFMath(_list(terms[:2])),
                         ' and each new term is the sum of the previous two. Find ',
                         PDFMath(rf'a_{{{index}}}'), '.')
        solution = PDFText('Continuing gives ', PDFMath(_list(terms)),
                           ', so the requested term is ', PDFMath(str(terms[-1])), '.') if with_solution else None
        super().__init__(prompt, 'fibonacci', solution, difficulty,
                         {'initial': (first, second), 'index': index,
                          'terms': tuple(terms), 'result': terms[-1]})


class PDFSigmaEvaluation(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        upper = rng.randint(*{'easy': (4, 7), 'medium': (6, 10), 'hard': (9, 14)}[difficulty])
        a, b = _nonzero(rng, -4, 5), rng.randint(-5, 5)
        quadratic = difficulty == 'hard'
        c = rng.randint(-3, 3) if quadratic else 0
        values = tuple(c*k*k+a*k+b for k in range(1, upper+1))
        result = sum(values)
        expression = format_polynomial((c, a, b), variable='k') if c else format_polynomial((a, b), variable='k')
        prompt = PDFText('Evaluate ', PDFMath(rf'\sum_{{k=1}}^{{{upper}}}({expression})'), '.')
        solution = PDFText('The terms are ', PDFMath(_list(values)), '; their sum is ',
                           PDFMath(str(result)), '.') if with_solution else None
        super().__init__(prompt, 'sigma_evaluation', solution, difficulty,
                         {'upper': upper, 'coefficients': (c, a, b),
                          'terms': values, 'result': result})


class PDFSequenceLimit(PDFSequenceSeriesExercise):
    """A polynomial-quotient limit with an optional prescribed outcome."""

    CASES = ('finite_zero', 'finite_ratio', 'positive_infinity',
             'negative_infinity', 'oscillating')

    def __init__(self, with_solution=True, difficulty='medium', *, seed=None,
                 case=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        aliases = {
            'zero': 'finite_zero', 'finite': 'finite_ratio',
            '+infinity': 'positive_infinity', '-infinity': 'negative_infinity',
            'does_not_exist': 'oscillating', 'dne': 'oscillating',
        }
        if case is not None:
            if not isinstance(case, str):
                raise TypeError('case must be text or None')
            case = case.strip().lower()
            if case not in ('+infinity', '-infinity'):
                case = case.replace('-', '_').replace(' ', '_')
            case = aliases.get(case, case)
            if case not in self.CASES:
                raise ValueError(f'case must be one of {", ".join(self.CASES)}')
        choices = {
            'easy': self.CASES[:-1],
            'medium': self.CASES,
            'hard': self.CASES,
        }[difficulty]
        case = case or rng.choice(choices)
        maximum_degree = {'easy': 1, 'medium': 3, 'hard': 5}[difficulty]
        maximum_coefficient = {'easy': 5, 'medium': 7, 'hard': 9}[difficulty]

        if case == 'finite_zero':
            denominator_degree = rng.randint(1, maximum_degree)
            numerator_degree = rng.randint(0, denominator_degree-1)
        elif case in ('finite_ratio', 'oscillating'):
            numerator_degree = denominator_degree = rng.randint(1, maximum_degree)
        else:
            denominator_degree = rng.randint(0, max(0, maximum_degree-1))
            numerator_degree = rng.randint(denominator_degree+1, maximum_degree)

        denominator = _polynomial_coefficients(
            rng, denominator_degree, maximum_coefficient, one_sign=True)
        numerator = _polynomial_coefficients(rng, numerator_degree, maximum_coefficient)
        if case == 'positive_infinity' and numerator[0]*denominator[0] < 0:
            numerator = (-numerator[0], *numerator[1:])
        elif case == 'negative_infinity' and numerator[0]*denominator[0] > 0:
            numerator = (-numerator[0], *numerator[1:])

        leading_ratio = Fraction(numerator[0], denominator[0])
        numerator_text = format_polynomial(numerator, variable='n')
        denominator_text = format_polynomial(denominator, variable='n')
        quotient = rf'\frac{{{numerator_text}}}{{{denominator_text}}}'
        expression = rf'(-1)^n{quotient}' if case == 'oscillating' else quotient
        ask_style = rng.choice(('evaluate', 'existence'))
        if ask_style == 'evaluate':
            prompt = PDFText('Evaluate ', PDFMath(rf'\lim_{{n\to\infty}}{expression}'),
                             ', or state that it does not exist.')
        else:
            prompt = PDFText('Determine whether ', PDFMath(rf'a_n={expression}'),
                             ' has a finite limit. If it does, find it; otherwise describe its behavior.')

        if case == 'finite_zero':
            limit, behavior = Fraction(0), 'finite'
            explanation = PDFText('The denominator has higher degree, so the limit is ',
                                  PDFMath('0'), '.')
        elif case == 'finite_ratio':
            limit, behavior = leading_ratio, 'finite'
            explanation = PDFText('The degrees are equal, so the limit is the ratio of the leading coefficients: ',
                                  PDFMath(_number(limit)), '.')
        elif case in ('positive_infinity', 'negative_infinity'):
            limit, behavior = None, case
            gap = numerator_degree-denominator_degree
            coefficient = ('-' if leading_ratio == -1 else '' if leading_ratio == 1
                           else _number(leading_ratio))
            dominant = rf'{coefficient}n' if gap == 1 else rf'{coefficient}n^{{{gap}}}'
            infinity = r'+\infty' if case == 'positive_infinity' else r'-\infty'
            explanation = PDFText('There is no finite limit. The quotient behaves like ',
                                  PDFMath(dominant), ' and tends to ', PDFMath(infinity), '.')
        else:
            limit, behavior = None, 'oscillating'
            explanation = PDFText('The polynomial quotient tends to ', PDFMath(_number(leading_ratio)),
                                  ', but ', PDFMath(r'(-1)^n'), ' makes the sequence oscillate. '
                                  'The limit does not exist.')

        result = limit if limit is not None else behavior
        data = {
            'numerator': numerator, 'denominator': denominator,
            'numerator_degree': numerator_degree,
            'denominator_degree': denominator_degree,
            'degree': (numerator_degree, denominator_degree),
            'leading_ratio': leading_ratio, 'case': case, 'behavior': behavior,
            'exists': limit is not None, 'converges': limit is not None,
            'limit': limit, 'ask_style': ask_style, 'result': result,
        }
        super().__init__(prompt, 'sequence_limit', explanation if with_solution else None,
                         difficulty, data)


class PDFElementaryFunctionLimit(PDFSequenceSeriesExercise):
    """A continuity-based elementary-function limit at a finite point."""

    FUNCTIONS = ('polynomial', 'rational', 'square_root', 'exponential',
                 'logarithm', 'sine', 'cosine', 'absolute_value')

    def __init__(self, with_solution=True, difficulty='medium', *, seed=None,
                 function=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        aliases = {'root': 'square_root', 'sqrt': 'square_root', 'exp': 'exponential',
                   'log': 'logarithm', 'ln': 'logarithm', 'sin': 'sine',
                   'cos': 'cosine', 'absolute': 'absolute_value', 'abs': 'absolute_value'}
        if function is not None:
            if not isinstance(function, str):
                raise TypeError('function must be text or None')
            function = function.strip().lower().replace('-', '_').replace(' ', '_')
            function = aliases.get(function, function)
            if function not in self.FUNCTIONS:
                raise ValueError(f'function must be one of {", ".join(self.FUNCTIONS)}')
        function = function or rng.choice(self.FUNCTIONS)
        point = rng.randint(-4, 4)
        result_expression = None

        if function == 'polynomial':
            degree = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
            coefficients = _polynomial_coefficients(rng, degree, 5)
            result = sum(coefficient*point**power for power, coefficient in
                         enumerate(reversed(coefficients)))
            expression = format_polynomial(coefficients, variable='x')
            parameters = {'coefficients': coefficients}
        elif function == 'rational':
            numerator = _polynomial_coefficients(rng, 1 if difficulty == 'easy' else 2, 5)
            denominator_degree = 1 if difficulty != 'hard' else 2
            denominator_value = _nonzero(rng, -7, 7)
            tail = [rng.randint(-4, 4) for _ in range(denominator_degree)]
            leading = _nonzero(rng, -5, 5)
            denominator = [leading, *tail]
            current = sum(coefficient*point**power for power, coefficient in
                          enumerate(reversed(denominator)))
            denominator[-1] += denominator_value-current
            numerator_value = sum(coefficient*point**power for power, coefficient in
                                  enumerate(reversed(numerator)))
            result = Fraction(numerator_value, denominator_value)
            expression = rf'\frac{{{format_polynomial(numerator, variable="x")}}}' \
                         rf'{{{format_polynomial(denominator, variable="x")}}}'
            parameters = {'numerator': numerator, 'denominator': tuple(denominator),
                          'denominator_value': denominator_value}
        elif function == 'square_root':
            slope = _nonzero(rng, -5, 5)
            root = rng.randint(1, 6)
            intercept = root*root-slope*point
            expression = rf'\sqrt{{{format_polynomial((slope, intercept), variable="x")}}}'
            result = root
            parameters = {'slope': slope, 'intercept': intercept, 'radicand': root*root}
        elif function in ('exponential', 'logarithm'):
            slope = _nonzero(rng, -4, 4)
            target = rng.randint(-3, 3) if function == 'exponential' else rng.randint(1, 9)
            intercept = target-slope*point
            inner = format_polynomial((slope, intercept), variable='x')
            if function == 'exponential':
                expression = rf'\mathrm{{e}}^{{{inner}}}'
                result_expression = _e_power(target)
                numeric_result = math.exp(target)
            else:
                expression = rf'\ln\left({inner}\right)'
                result_expression = '0' if target == 1 else rf'\ln\left({target}\right)'
                numeric_result = math.log(target)
            result = result_expression
            parameters = {'slope': slope, 'intercept': intercept, 'inner_value': target,
                          'numeric_result': numeric_result}
        elif function in ('sine', 'cosine'):
            sixths = rng.randint(-6, 6)
            point = Fraction(sixths, 6)
            command = 'sin' if function == 'sine' else 'cos'
            expression = rf'\{command}\left(x\right)'
            values = {
                'sine': ('0', r'\frac{1}{2}', r'\frac{\sqrt{3}}{2}', '1',
                         r'\frac{\sqrt{3}}{2}', r'\frac{1}{2}', '0',
                         r'-\frac{1}{2}', r'-\frac{\sqrt{3}}{2}', '-1',
                         r'-\frac{\sqrt{3}}{2}', r'-\frac{1}{2}'),
                'cosine': ('1', r'\frac{\sqrt{3}}{2}', r'\frac{1}{2}', '0',
                           r'-\frac{1}{2}', r'-\frac{\sqrt{3}}{2}', '-1',
                           r'-\frac{\sqrt{3}}{2}', r'-\frac{1}{2}', '0',
                           r'\frac{1}{2}', r'\frac{\sqrt{3}}{2}'),
            }
            result_expression = values[function][sixths % 12]
            angle = math.pi*float(point)
            numeric_result = math.sin(angle) if function == 'sine' else math.cos(angle)
            result = result_expression
            parameters = {'pi_multiple': point, 'numeric_result': numeric_result}
        else:
            slope, intercept = _nonzero(rng, -5, 5), rng.randint(-7, 7)
            inner_value = slope*point+intercept
            expression = rf'\left|{format_polynomial((slope, intercept), variable="x")}\right|'
            result = abs(inner_value)
            parameters = {'slope': slope, 'intercept': intercept, 'inner_value': inner_value}

        point_expression = _pi_multiple(point) if function in ('sine', 'cosine') else str(point)
        shown_result = result_expression if result_expression is not None else _number(result)
        prompt = PDFText('Evaluate ', PDFMath(rf'\lim_{{x\to {point_expression}}}{expression}'), '.')
        solution = PDFText('The function is continuous and defined at the given point, so direct substitution gives ',
                           PDFMath(shown_result), '.') if with_solution else None
        data = {'function': function, 'point': point, 'expression': expression,
                'defined_at_point': True, 'result': result, **parameters}
        super().__init__(prompt, 'elementary_limit', solution, difficulty, data)


class PDFEulerLimit(PDFSequenceSeriesExercise):
    """A closed-form limit derived from (1+u/n)^n -> e^u."""

    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        if difficulty == 'easy':
            numerator, denominator, multiplier = 1, 1, 1
        elif difficulty == 'medium':
            numerator, denominator = _nonzero(rng, -4, 4), 1
            multiplier = rng.choice((1, 2))
        else:
            numerator = _nonzero(rng, -6, 6)
            denominator = rng.randint(1, 5)
            multiplier = _nonzero(rng, -4, 4)
        alpha = Fraction(numerator, denominator)
        exponent = alpha*multiplier
        sign = '+' if alpha > 0 else '-'
        magnitude = abs(alpha)
        increment = (rf'\frac{{{magnitude.numerator}}}{{n}}' if magnitude.denominator == 1 else
                     rf'\frac{{{magnitude.numerator}}}{{{magnitude.denominator}n}}')
        power = 'n' if multiplier == 1 else '-n' if multiplier == -1 else f'{multiplier}n'
        expression = rf'\left(1{sign}{increment}\right)^{{{power}}}'
        result_expression = _e_power(exponent)
        prompt = PDFText('Evaluate ', PDFMath(rf'\lim_{{n\to\infty}}{expression}'), '.')
        solution = PDFText('Using ', PDFMath(r'\lim_{n\to\infty}(1+u/n)^n=\mathrm{e}^u'),
                           ', the exponent in the result is ', PDFMath(_number(exponent)),
                           ', so the limit is ', PDFMath(result_expression), '.') if with_solution else None
        data = {'numerator': numerator, 'denominator': denominator,
                'base_increment': alpha, 'multiplier': multiplier,
                'exponent': exponent, 'result': result_expression,
                'numeric_result': math.exp(float(exponent))}
        super().__init__(prompt, 'euler_limit', solution, difficulty, data)


class PDFRemovableLimit(PDFSequenceSeriesExercise):
    """A rational limit with one common factor and a removable discontinuity."""

    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        point = rng.randint(-5, 5)
        numerator_slope, numerator_intercept = _nonzero(rng, -6, 6), rng.randint(-7, 7)
        denominator_slope = _nonzero(rng, -6, 6)
        denominator_value = _nonzero(rng, -8, 8)
        denominator_intercept = denominator_value-denominator_slope*point
        result = Fraction(numerator_slope*point+numerator_intercept, denominator_value)
        factor = _linear_factor(point)
        numerator_linear = format_polynomial((numerator_slope, numerator_intercept), variable='x')
        denominator_linear = format_polynomial((denominator_slope, denominator_intercept), variable='x')
        if difficulty == 'hard':
            numerator = (numerator_slope,
                         numerator_intercept-point*numerator_slope,
                         -point*numerator_intercept)
            denominator = (denominator_slope,
                           denominator_intercept-point*denominator_slope,
                           -point*denominator_intercept)
            expression = rf'\frac{{{format_polynomial(numerator, variable="x")}}}' \
                         rf'{{{format_polynomial(denominator, variable="x")}}}'
        else:
            numerator = denominator = None
            expression = rf'\frac{{({factor})({numerator_linear})}}{{({factor})({denominator_linear})}}'
        prompt = PDFText('Evaluate ', PDFMath(rf'\lim_{{x\to {point}}}{expression}'), '.')
        solution = PDFText('Cancel the common factor ', PDFMath(factor), ' and substitute ',
                           PDFMath(rf'x={point}'), ' into ',
                           PDFMath(rf'\frac{{{numerator_linear}}}{{{denominator_linear}}}'),
                           ' to obtain ', PDFMath(_number(result)), '.') if with_solution else None
        data = {'point': point, 'numerator_linear': (numerator_slope, numerator_intercept),
                'denominator_linear': (denominator_slope, denominator_intercept),
                'expanded_numerator': numerator, 'expanded_denominator': denominator,
                'defined_at_point': False, 'removable': True, 'result': result}
        super().__init__(prompt, 'removable_limit', solution, difficulty, data)


class PDFStandardTrigLimit(PDFSequenceSeriesExercise):
    """A scaled standard trigonometric limit at zero."""

    FORMS = ('sine_ratio', 'tangent_ratio', 'one_minus_cosine')

    def __init__(self, with_solution=True, difficulty='medium', *, seed=None,
                 form=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        aliases = {'sine': 'sine_ratio', 'sin': 'sine_ratio',
                   'tangent': 'tangent_ratio', 'tan': 'tangent_ratio',
                   'cosine': 'one_minus_cosine', 'cos': 'one_minus_cosine'}
        if form is not None:
            if not isinstance(form, str):
                raise TypeError('form must be text or None')
            form = form.strip().lower().replace('-', '_').replace(' ', '_')
            form = aliases.get(form, form)
            if form not in self.FORMS:
                raise ValueError(f'form must be one of {", ".join(self.FORMS)}')
        available = ('sine_ratio',) if difficulty == 'easy' else self.FORMS
        form = form or rng.choice(available)
        scale = _nonzero(rng, -6, 6)
        divisor = _nonzero(rng, -6, 6)
        argument = 'x' if scale == 1 else '-x' if scale == -1 else f'{scale}x'
        denominator = 'x' if divisor == 1 else '-x' if divisor == -1 else f'{divisor}x'
        if form == 'sine_ratio':
            expression = rf'\frac{{\sin({argument})}}{{{denominator}}}'
            result = Fraction(scale, divisor)
            identity = r'\lim_{u\to0}\frac{\sin u}{u}=1'
        elif form == 'tangent_ratio':
            expression = rf'\frac{{\tan({argument})}}{{{denominator}}}'
            result = Fraction(scale, divisor)
            identity = r'\lim_{u\to0}\frac{\tan u}{u}=1'
        else:
            divisor = abs(divisor)
            denominator = 'x^2' if divisor == 1 else f'{divisor}x^2'
            expression = rf'\frac{{1-\cos({argument})}}{{{denominator}}}'
            result = Fraction(scale*scale, 2*divisor)
            identity = r'\lim_{u\to0}\frac{1-\cos u}{u^2}=\frac{1}{2}'
        prompt = PDFText('Evaluate ', PDFMath(rf'\lim_{{x\to0}}{expression}'), '.')
        solution = PDFText('Apply ', PDFMath(identity), ' and account for the scale factors to obtain ',
                           PDFMath(_number(result)), '.') if with_solution else None
        data = {'form': form, 'scale': scale, 'divisor': divisor,
                'point': 0, 'result': result}
        super().__init__(prompt, 'standard_trig_limit', solution, difficulty, data)


class PDFConvergenceClassification(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        case = rng.choice(('zero', 'unbounded', 'oscillating'))
        if case == 'zero':
            ratio = rng.choice((Fraction(1, 2), Fraction(-1, 2), Fraction(2, 3)))
            expression, converges, limit = rf'({_number(ratio)})^n', True, Fraction(0)
        elif case == 'unbounded':
            ratio = rng.choice((2, 3, -2))
            expression, converges, limit = rf'({ratio})^n', False, None
        else:
            ratio = -1
            expression, converges, limit = r'(-1)^n', False, None
        prompt = PDFText('Determine whether ', PDFMath(rf'a_n={expression}'),
                         ' converges. If it does, give the limit.')
        answer = PDFText('It converges to ', PDFMath(_number(limit)), '.') if converges else PDFText('It diverges.')
        solution = answer if with_solution else None
        super().__init__(prompt, 'convergence_classification', solution, difficulty,
                         {'case': case, 'ratio': ratio, 'converges': converges,
                          'limit': limit, 'result': limit if converges else 'diverges'})


class PDFPSeries(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        powers = (Fraction(1, 2), 1, 2) if difficulty != 'hard' else (Fraction(2, 3), 1, Fraction(3, 2), 2, 3)
        power = rng.choice(powers)
        converges = power > 1
        prompt = PDFText('Determine whether ', PDFMath(rf'\sum_{{n=1}}^\infty\frac{{1}}{{n^{{{_number(power)}}}}}'),
                         ' converges.')
        solution = PDFText(f'This p-series {"converges" if converges else "diverges"} because ',
                           PDFMath(rf'p={_number(power)}'),
                           ' is greater than 1.' if converges else ' is not greater than 1.') if with_solution else None
        super().__init__(prompt, 'p_series', solution, difficulty,
                         {'power': power, 'converges': converges,
                          'result': 'converges' if converges else 'diverges'})


class PDFGeometricSeriesTest(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        ratio = rng.choice((Fraction(1, 2), Fraction(-2, 3), 2, -1, 3))
        first = _nonzero(rng, -7, 7)
        converges = abs(ratio) < 1
        result = Fraction(first)/(1-ratio) if converges else None
        prompt = PDFText('Determine whether the geometric series with ', PDFMath(rf'a_1={first}'),
                         ' and ', PDFMath(rf'r={_number(ratio)}'), ' converges; find its sum when possible.')
        solution = (PDFText('It converges because ', PDFMath(r'|r|<1'), ', and ',
                            PDFMath(rf'S_\infty={_number(result)}'), '.') if converges else
                    PDFText('It diverges because ', PDFMath(r'|r|\geq1'), '.')) if with_solution else None
        super().__init__(prompt, 'geometric_series_test', solution, difficulty,
                         {'first': first, 'ratio': ratio, 'converges': converges,
                          'sum': result, 'result': result if converges else 'diverges'})


class PDFAlternatingSeries(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        power = rng.choice((1, 2, 3))
        classification = 'conditionally' if power == 1 else 'absolutely'
        prompt = PDFText('Classify the convergence of ',
                         PDFMath(rf'\sum_{{n=1}}^\infty\frac{{(-1)^{{n+1}}}}{{n^{{{power}}}}}'), '.')
        reason = ('the alternating-series test applies, but the absolute harmonic series diverges'
                  if power == 1 else f'the absolute p-series has p={power}>1')
        solution = PDFText(f'It converges {classification}; {reason}.') if with_solution else None
        super().__init__(prompt, 'alternating_series', solution, difficulty,
                         {'power': power, 'classification': classification,
                          'result': classification})


class PDFTelescopingSeries(PDFSequenceSeriesExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        count = rng.randint(*{'easy': (4, 8), 'medium': (8, 20), 'hard': (20, 60)}[difficulty])
        terms = tuple(Fraction(1, k)-Fraction(1, k+1) for k in range(1, count+1))
        result = Fraction(count, count+1)
        prompt = PDFText('Evaluate the telescoping sum ',
                         PDFMath(rf'\sum_{{k=1}}^{{{count}}}\left(\frac{{1}}{{k}}-\frac{{1}}{{k+1}}\right)'), '.')
        solution = PDFText('All intermediate terms cancel, leaving ',
                           PDFMath(rf'1-\frac{{1}}{{{count+1}}}={_number(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'telescoping_series', solution, difficulty,
                         {'count': count, 'terms': terms, 'result': result})


_FACTORIES = {
    'identify_sequence': PDFIdentifySequence,
    'arithmetic_next_terms': PDFArithmeticNextTerms,
    'arithmetic_nth_term': PDFArithmeticNthTerm,
    'arithmetic_difference': PDFArithmeticDifference,
    'arithmetic_sum': PDFArithmeticSum,
    'arithmetic_missing_term': PDFArithmeticMissingTerm,
    'geometric_next_terms': PDFGeometricNextTerms,
    'geometric_nth_term': PDFGeometricNthTerm,
    'geometric_ratio': PDFGeometricRatio,
    'geometric_sum': PDFGeometricSum,
    'infinite_geometric_sum': PDFInfiniteGeometricSum,
    'recursive_sequence': PDFRecursiveSequence,
    'fibonacci': PDFFibonacciSequence,
    'sigma_evaluation': PDFSigmaEvaluation,
    'sequence_limit': PDFSequenceLimit,
    'convergence_classification': PDFConvergenceClassification,
    'p_series': PDFPSeries,
    'geometric_series_test': PDFGeometricSeriesTest,
    'alternating_series': PDFAlternatingSeries,
    'telescoping_series': PDFTelescopingSeries,
    'elementary_limit': PDFElementaryFunctionLimit,
    'euler_limit': PDFEulerLimit,
    'removable_limit': PDFRemovableLimit,
    'standard_trig_limit': PDFStandardTrigLimit,
}

_ALIASES = {
    'identify': 'identify_sequence', 'classify_sequence': 'identify_sequence',
    'arithmetic_next': 'arithmetic_next_terms', 'arithmetic_term': 'arithmetic_nth_term',
    'common_difference': 'arithmetic_difference', 'sum_arithmetic': 'arithmetic_sum',
    'missing_term': 'arithmetic_missing_term', 'geometric_next': 'geometric_next_terms',
    'geometric_term': 'geometric_nth_term', 'common_ratio': 'geometric_ratio',
    'sum_geometric': 'geometric_sum', 'geometric_infinite_sum': 'infinite_geometric_sum',
    'recurrence': 'recursive_sequence', 'fibonacci_sequence': 'fibonacci',
    'sigma': 'sigma_evaluation', 'limit': 'sequence_limit',
    'convergence': 'convergence_classification', 'pseries': 'p_series',
    'geometric_test': 'geometric_series_test', 'alternating_test': 'alternating_series',
    'telescoping': 'telescoping_series',
    'elementary': 'elementary_limit', 'direct_limit': 'elementary_limit',
    'euler': 'euler_limit', 'e_limit': 'euler_limit',
    'removable': 'removable_limit', 'hole_limit': 'removable_limit',
    'trig_limit': 'standard_trig_limit', 'standard_trigonometric_limit': 'standard_trig_limit',
}


def sequence_exercise(kind, *, difficulty='medium', seed=None,
                      with_solution=True, _rng=None, **options):
    """Create a sequence or series exercise by a canonical or friendly name."""
    if not isinstance(kind, str):
        raise TypeError('kind must be text')
    key = kind.strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        choices = ', '.join(SEQUENCE_SERIES_EXERCISE_TYPES)
        raise ValueError(f'Unknown sequence exercise {kind!r}; choose from {choices}') from exc
    return factory(with_solution=with_solution, difficulty=difficulty, seed=seed,
                   _rng=_rng, **options)


__all__ = [
    'PDFSequenceSeriesExercise', 'PDFIdentifySequence', 'PDFArithmeticNextTerms',
    'PDFArithmeticNthTerm', 'PDFArithmeticDifference', 'PDFArithmeticSum',
    'PDFArithmeticMissingTerm', 'PDFGeometricNextTerms', 'PDFGeometricNthTerm',
    'PDFGeometricRatio', 'PDFGeometricSum', 'PDFInfiniteGeometricSum',
    'PDFRecursiveSequence', 'PDFFibonacciSequence', 'PDFSigmaEvaluation',
    'PDFSequenceLimit', 'PDFConvergenceClassification', 'PDFPSeries',
    'PDFGeometricSeriesTest', 'PDFAlternatingSeries', 'PDFTelescopingSeries',
    'PDFElementaryFunctionLimit', 'PDFEulerLimit', 'PDFRemovableLimit',
    'PDFStandardTrigLimit',
    'SEQUENCE_SERIES_EXERCISE_TYPES', 'sequence_exercise',
]
