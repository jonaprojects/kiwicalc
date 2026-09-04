"""Calculus and numerical-method exercises with reproducible exact data."""
from __future__ import annotations

from fractions import Fraction
import random

from .formatting import PDFText, format_math, format_polynomial
from .layout import PDFMath
from .worksheet import PDFExercise


DIFFICULTIES = ('easy', 'medium', 'hard')
CALCULUS_EXERCISE_TYPES = (
    'difference_quotient', 'derivative', 'tangent_line', 'critical_points',
    'monotonicity', 'concavity', 'optimization', 'definite_integral',
    'area_between', 'numerical_derivative', 'trapezoidal_rule', 'simpson_rule',
    'newton_iteration', 'euler_method', 'runge_kutta',
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


def _evaluate(coefficients, x):
    value = 0
    for coefficient in coefficients:
        value = value*x+coefficient
    return value


def _derivative(coefficients):
    degree = len(coefficients)-1
    return tuple(coefficient*(degree-index) for index, coefficient in enumerate(coefficients[:-1]))


def _antiderivative_value(coefficients, x):
    degree = len(coefficients)-1
    return sum(Fraction(coefficient, degree-index+1)*x**(degree-index+1)
               for index, coefficient in enumerate(coefficients))


def _number(value):
    return format_math(value if isinstance(value, Fraction) else Fraction(value))


def _coefficient_term(coefficient, variable):
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return '-'+variable
    return f'{coefficient}{variable}'


def _interval(left, right, *, left_closed=False, right_closed=False):
    return ('[' if left_closed else '(') + left + ',' + right + (']' if right_closed else ')')


class PDFCalculusNumericalExercise(PDFExercise):
    """Base class exposing stable source data for checking and reuse."""
    __slots__ = ('kind', 'difficulty', 'data')

    def __init__(self, prompt, kind, solution, difficulty, data):
        category = 'numerical method' if kind in {
            'numerical_derivative', 'trapezoidal_rule', 'simpson_rule',
            'newton_iteration', 'euler_method', 'runge_kutta',
        } else 'calculus'
        super().__init__(prompt, category, kind, solution=solution)
        self.kind = kind
        self.difficulty = difficulty
        self.data = data


class PDFDifferenceQuotient(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        degree = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        coefficients = tuple(_nonzero(rng, -5, 5) for _ in range(degree+1))
        point = rng.randint(-3, 3)
        derivative = _derivative(coefficients)
        slope = _evaluate(derivative, point)
        function = format_polynomial(coefficients)
        prompt = PDFText('Using the difference quotient, find ', PDFMath(rf'f\prime({point})'),
                         ' for ', PDFMath('f(x)='+function), '.', plain=f'Find f\'({point}) from first principles.')
        solution = PDFText('Evaluate ', PDFMath(rf'\lim_{{h\to0}}\frac{{f({point}+h)-f({point})}}{{h}}'),
                           ' to obtain ', PDFMath(str(slope)), '.', plain=str(slope)) if with_solution else None
        super().__init__(prompt, 'difference_quotient', solution, difficulty,
                         {'coefficients': coefficients, 'point': point,
                          'derivative_coefficients': derivative, 'result': slope})


class PDFDerivativeExercise(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        if difficulty == 'easy':
            coefficients = tuple(_nonzero(rng, -6, 6) for _ in range(3))
            source = format_polynomial(coefficients)
            derivative = format_polynomial(_derivative(coefficients))
            rule, data = 'power', {'coefficients': coefficients,
                                   'derivative_coefficients': _derivative(coefficients)}
        elif difficulty == 'medium':
            a, b, c, d = (_nonzero(rng, -5, 5) for _ in range(4))
            source = f'({format_polynomial((a, b))})({format_polynomial((c, d))})'
            derivative_coefficients = (2*a*c, a*d+b*c)
            derivative = format_polynomial(derivative_coefficients)
            rule, data = 'product', {'factors': ((a, b), (c, d)),
                                     'derivative_coefficients': derivative_coefficients}
        else:
            a, b, power = _nonzero(rng, -5, 5), _nonzero(rng, -7, 7), rng.randint(3, 7)
            inside = format_polynomial((a, b))
            source = rf'({inside})^{{{power}}}'
            coefficient = power*a
            derivative = ('' if coefficient == 1 else '-' if coefficient == -1 else str(coefficient)) + rf'({inside})^{{{power-1}}}'
            rule, data = 'chain', {'inside': (a, b), 'power': power, 'coefficient': coefficient}
        prompt = PDFText('Differentiate: ', PDFMath('f(x)='+source), '.', plain=f'Differentiate {source}.')
        solution = PDFText('Using the ', rule, ' rule, ', PDFMath(rf'f\prime(x)={derivative}'), '.',
                           plain=derivative) if with_solution else None
        data.update({'rule': rule, 'source': source, 'derivative': derivative})
        super().__init__(prompt, 'derivative', solution, difficulty, data)


class PDFTangentLine(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        degree = 2 if difficulty != 'hard' else 3
        coefficients = tuple(_nonzero(rng, -4, 4) for _ in range(degree+1))
        point = rng.randint(-3, 3)
        value = _evaluate(coefficients, point)
        slope = _evaluate(_derivative(coefficients), point)
        function = format_polynomial(coefficients)
        line = rf'y-{value}={slope}(x-{point})'
        prompt = PDFText('Find the tangent line to ', PDFMath('f(x)='+function), ' at ',
                         PDFMath(f'x={point}'), '.', plain=f'Find the tangent at x={point}.')
        solution = PDFText(PDFMath(rf'f({point})={value}'), ' and ', PDFMath(rf'f\prime({point})={slope}'),
                           ', so ', PDFMath(line), '.', plain=line) if with_solution else None
        super().__init__(prompt, 'tangent_line', solution, difficulty,
                         {'coefficients': coefficients, 'point': point, 'value': value,
                          'slope': slope, 'line': line})


class PDFCriticalPoints(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        radius = rng.randint(1, {'easy': 3, 'medium': 5, 'hard': 8}[difficulty])
        scale = 1 if difficulty == 'easy' else _nonzero(rng, -3, 3)
        constant = rng.randint(-8, 8)
        coefficients = (scale, 0, -3*scale*radius*radius, constant)
        points = (-radius, radius)
        values = tuple(_evaluate(coefficients, x) for x in points)
        kinds = ('maximum', 'minimum') if scale > 0 else ('minimum', 'maximum')
        function = format_polynomial(coefficients)
        prompt = PDFText('Find and classify the critical points of ', PDFMath('f(x)='+function), '.',
                         plain=f'Find critical points of {function}.')
        answer = rf'({points[0]},{values[0]})\ \mathrm{{{kinds[0]}}},\quad({points[1]},{values[1]})\ \mathrm{{{kinds[1]}}}'
        solution = PDFText('Solve ', PDFMath(rf'f\prime(x)={3*scale}(x^2-{radius*radius})=0'),
                           '. The sign change gives ', PDFMath(answer), '.', plain=answer) if with_solution else None
        super().__init__(prompt, 'critical_points', solution, difficulty,
                         {'coefficients': coefficients, 'points': points, 'values': values,
                          'classifications': kinds})


class PDFMonotonicity(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        radius = rng.randint(1, {'easy': 3, 'medium': 5, 'hard': 8}[difficulty])
        scale = 1 if difficulty != 'hard' else _nonzero(rng, -3, 3)
        coefficients = (scale, 0, -3*scale*radius*radius, rng.randint(-5, 5))
        outer = (rf'(-\infty,-{radius})', rf'({radius},\infty)')
        inner = rf'(-{radius},{radius})'
        increasing, decreasing = (outer, (inner,)) if scale > 0 else ((inner,), outer)
        function = format_polynomial(coefficients)
        prompt = PDFText('Determine where ', PDFMath('f(x)='+function), ' is increasing and decreasing.',
                         plain=f'Find monotonicity intervals of {function}.')
        inc = r'\cup'.join(increasing)
        dec = r'\cup'.join(decreasing)
        solution = PDFText('Increasing on ', PDFMath(inc), '; decreasing on ', PDFMath(dec), '.',
                           plain=f'Increasing: {inc}; decreasing: {dec}') if with_solution else None
        super().__init__(prompt, 'monotonicity', solution, difficulty,
                         {'coefficients': coefficients, 'critical': (-radius, radius),
                          'increasing': increasing, 'decreasing': decreasing})


class PDFConcavity(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        scale = 1 if difficulty == 'easy' else _nonzero(rng, -3, 3)
        inflection = rng.randint(-5, 5)
        coefficients = (scale, -3*scale*inflection, rng.randint(-6, 6), rng.randint(-6, 6))
        left, right = rf'(-\infty,{inflection})', rf'({inflection},\infty)'
        up, down = ((right,), (left,)) if scale > 0 else ((left,), (right,))
        y = _evaluate(coefficients, inflection)
        function = format_polynomial(coefficients)
        prompt = PDFText('Find the concavity intervals and inflection point of ',
                         PDFMath('f(x)='+function), '.', plain=f'Analyze concavity of {function}.')
        solution = PDFText('Concave up on ', PDFMath(up[0]), ', down on ', PDFMath(down[0]),
                           '; inflection point ', PDFMath(f'({inflection},{y})'), '.',
                           plain=f'Inflection ({inflection},{y})') if with_solution else None
        super().__init__(prompt, 'concavity', solution, difficulty,
                         {'coefficients': coefficients, 'inflection': (inflection, y),
                          'concave_up': up, 'concave_down': down})


class PDFOptimization(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        side = rng.randint(3, {'easy': 8, 'medium': 14, 'hard': 20}[difficulty])
        perimeter = 4*side
        prompt = PDFText('A rectangle has perimeter ', PDFMath(str(perimeter)),
                         '. Find the dimensions that maximize its area.',
                         plain=f'Maximize a rectangle with perimeter {perimeter}.')
        solution = PDFText('Write ', PDFMath(rf'A(x)=x({2*side}-x)'), '. Its vertex occurs at ',
                           PDFMath(f'x={side}'), ', so the rectangle is ', PDFMath(rf'{side}\times{side}'),
                           ' with maximum area ', PDFMath(str(side*side)), '.', plain=f'{side} by {side}') if with_solution else None
        super().__init__(prompt, 'optimization', solution, difficulty,
                         {'perimeter': perimeter, 'dimensions': (side, side),
                          'maximum_area': side*side})


class PDFDefiniteIntegral(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        degree = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        anti = tuple(_nonzero(rng, -4, 4) for _ in range(degree+1))
        integrand = tuple(anti[index]*(degree-index+1) for index in range(degree+1))
        lower = rng.randint(-3, 1)
        upper = rng.randint(lower+1, 4)
        value = Fraction(_evaluate(anti+(0,), upper)-_evaluate(anti+(0,), lower))
        expression = format_polynomial(integrand)
        antiderivative = format_polynomial(anti+(0,))
        prompt = PDFText('Evaluate exactly: ', PDFMath(rf'\int_{{{lower}}}^{{{upper}}}({expression})\,dx'), '.',
                         plain=f'Integrate {expression} from {lower} to {upper}.')
        solution = PDFText('An antiderivative is ', PDFMath(antiderivative), '; evaluation gives ',
                           PDFMath(_number(value)), '.', plain=str(value)) if with_solution else None
        super().__init__(prompt, 'definite_integral', solution, difficulty,
                         {'coefficients': integrand, 'antiderivative_coefficients': anti+(0,),
                          'bounds': (lower, upper), 'result': value})


class PDFAreaBetweenCurves(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        radius = rng.randint(1, {'easy': 2, 'medium': 4, 'hard': 6}[difficulty])
        height = 2*radius*radius
        upper, lower = format_polynomial((-1, 0, height)), 'x^{2}'
        area = Fraction(8*radius**3, 3)
        prompt = PDFText('Find the enclosed area between ', PDFMath(f'y={upper}'), ' and ',
                         PDFMath(f'y={lower}'), '.', plain='Find the area between the two curves.')
        solution = PDFText('They intersect at ', PDFMath(rf'x=\pm{radius}'), '. Therefore ',
                           PDFMath(rf'A=\int_{{-{radius}}}^{{{radius}}}({height}-2x^2)\,dx={_number(area)}'), '.',
                           plain=str(area)) if with_solution else None
        super().__init__(prompt, 'area_between', solution, difficulty,
                         {'radius': radius, 'upper_coefficients': (-1, 0, height),
                          'lower_coefficients': (1, 0, 0), 'intersections': (-radius, radius),
                          'area': area})


class PDFNumericalDerivative(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        coefficients = tuple(_nonzero(rng, -5, 5) for _ in range(3))
        point = rng.randint(-3, 3)
        step = {'easy': Fraction(1), 'medium': Fraction(1, 2), 'hard': Fraction(1, 4)}[difficulty]
        left, right = Fraction(point)-step, Fraction(point)+step
        left_value, right_value = _evaluate(coefficients, left), _evaluate(coefficients, right)
        result = (right_value-left_value)/(2*step)
        function = format_polynomial(coefficients)
        prompt = PDFText('Use a central difference with ', PDFMath(f'h={_number(step)}'), ' to estimate ',
                         PDFMath(rf'f\prime({point})'), ' for ', PDFMath('f(x)='+function), '.',
                         plain='Use a central-difference derivative.')
        solution = PDFText(PDFMath(rf'\frac{{f({format_math(right)})-f({format_math(left)})}}{{2h}}={_number(result)}'), '.',
                           plain=str(result)) if with_solution else None
        super().__init__(prompt, 'numerical_derivative', solution, difficulty,
                         {'coefficients': coefficients, 'point': point, 'step': step,
                          'samples': ((left, left_value), (right, right_value)), 'result': result})


def _composite_values(coefficients, lower, upper, intervals):
    step = Fraction(upper-lower, intervals)
    return step, tuple(_evaluate(coefficients, Fraction(lower)+index*step)
                       for index in range(intervals+1))


class PDFTrapezoidalRule(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        coefficients = tuple(_nonzero(rng, -4, 4) for _ in range(3))
        intervals = {'easy': 2, 'medium': 4, 'hard': 6}[difficulty]
        lower, upper = 0, intervals
        step, values = _composite_values(coefficients, lower, upper, intervals)
        approximation = step*(values[0]+values[-1]+2*sum(values[1:-1]))/2
        exact = _antiderivative_value(coefficients, upper)-_antiderivative_value(coefficients, lower)
        expression = format_polynomial(coefficients)
        prompt = PDFText('Approximate ', PDFMath(rf'\int_0^{{{upper}}}({expression})\,dx'),
                         ' using the composite trapezoidal rule with ', PDFMath(f'n={intervals}'), '.',
                         plain='Use the composite trapezoidal rule.')
        solution = PDFText('The tabulated values give ', PDFMath(rf'T_{{{intervals}}}={_number(approximation)}'),
                           '; absolute error ', PDFMath(_number(abs(approximation-exact))), '.',
                           plain=str(approximation)) if with_solution else None
        super().__init__(prompt, 'trapezoidal_rule', solution, difficulty,
                         {'coefficients': coefficients, 'bounds': (lower, upper), 'intervals': intervals,
                          'step': step, 'values': values, 'result': approximation,
                          'exact': exact, 'absolute_error': abs(approximation-exact)})


class PDFSimpsonRule(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        coefficients = tuple(_nonzero(rng, -3, 3) for _ in range(4))
        intervals = {'easy': 2, 'medium': 4, 'hard': 6}[difficulty]
        lower, upper = 0, intervals
        step, values = _composite_values(coefficients, lower, upper, intervals)
        approximation = step*(values[0]+values[-1]+4*sum(values[1:-1:2])+2*sum(values[2:-1:2]))/3
        exact = _antiderivative_value(coefficients, upper)-_antiderivative_value(coefficients, lower)
        expression = format_polynomial(coefficients)
        prompt = PDFText('Apply composite Simpson\'s rule to ',
                         PDFMath(rf'\int_0^{{{upper}}}({expression})\,dx'), ' with ',
                         PDFMath(f'n={intervals}'), '.', plain='Use composite Simpson\'s rule.')
        solution = PDFText(PDFMath(rf'S_{{{intervals}}}={_number(approximation)}'),
                           '. For this cubic, Simpson\'s rule is exact.', plain=str(approximation)) if with_solution else None
        super().__init__(prompt, 'simpson_rule', solution, difficulty,
                         {'coefficients': coefficients, 'bounds': (lower, upper), 'intervals': intervals,
                          'step': step, 'values': values, 'result': approximation, 'exact': exact})


class PDFNewtonIteration(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        radicand = rng.choice((2, 3, 5, 6, 7, 10))
        initial = Fraction(rng.randint(1, 4))
        iterations = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        values = [initial]
        for _ in range(iterations):
            current = values[-1]
            values.append((current+Fraction(radicand, current))/2)
        prompt = PDFText('Starting from ', PDFMath(f'x_0={_number(initial)}'), ', perform ',
                         str(iterations), ' Newton iteration', 's' if iterations != 1 else '',
                         ' for ', PDFMath(f'f(x)=x^2-{radicand}'), '.', plain='Perform Newton iterations.')
        steps = r',\quad'.join(rf'x_{{{index}}}={_number(value)}'
                              for index, value in enumerate(values[1:], start=1))
        solution = PDFText('Using ', PDFMath(rf'x_{{n+1}}=\frac{{1}}{{2}}(x_n+\frac{{{radicand}}}{{x_n}})'),
                           ': ', PDFMath(steps), '.', plain=str(values[-1])) if with_solution else None
        super().__init__(prompt, 'newton_iteration', solution, difficulty,
                         {'radicand': radicand, 'initial': initial, 'iterations': iterations,
                          'values': tuple(values), 'result': values[-1]})


class PDFEulerMethod(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        coefficient = _nonzero(rng, -3, 3)
        initial = Fraction(_nonzero(rng, -5, 5))
        step = {'easy': Fraction(1), 'medium': Fraction(1, 2), 'hard': Fraction(1, 4)}[difficulty]
        iterations = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        values = [initial]
        for _ in range(iterations):
            values.append(values[-1]+step*coefficient*values[-1])
        target = step*iterations
        prompt = PDFText('Use Euler\'s method with ', PDFMath(f'h={_number(step)}'), ' to approximate ',
                         PDFMath(f'y({_number(target)})'), ' for ',
                         PDFMath(rf'y\prime={_coefficient_term(coefficient, "y")},\ y(0)={_number(initial)}'), '.',
                         plain='Apply Euler\'s method.')
        steps = r',\quad'.join(rf'y_{{{index}}}={_number(value)}'
                              for index, value in enumerate(values[1:], start=1))
        solution = PDFText('Using ', PDFMath(r'y_{n+1}=y_n+h f(t_n,y_n)'), ': ', PDFMath(steps), '.',
                           plain=str(values[-1])) if with_solution else None
        super().__init__(prompt, 'euler_method', solution, difficulty,
                         {'coefficient': coefficient, 'initial': initial, 'step': step,
                          'iterations': iterations, 'values': tuple(values), 'target': target,
                          'result': values[-1]})


class PDFRungeKuttaMethod(PDFCalculusNumericalExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        coefficient = _nonzero(rng, -2, 2)
        initial = Fraction(_nonzero(rng, -4, 4))
        step = {'easy': Fraction(1), 'medium': Fraction(1, 2), 'hard': Fraction(1, 4)}[difficulty]
        iterations = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        values, stage_data = [initial], []
        for _ in range(iterations):
            y = values[-1]
            k1 = coefficient*y
            k2 = coefficient*(y+step*k1/2)
            k3 = coefficient*(y+step*k2/2)
            k4 = coefficient*(y+step*k3)
            values.append(y+step*(k1+2*k2+2*k3+k4)/6)
            stage_data.append((k1, k2, k3, k4))
        target = step*iterations
        prompt = PDFText('Use classical RK4 with ', PDFMath(f'h={_number(step)}'), ' to approximate ',
                         PDFMath(f'y({_number(target)})'), ' for ',
                         PDFMath(rf'y\prime={_coefficient_term(coefficient, "y")},\ y(0)={_number(initial)}'), '.',
                         plain='Apply the classical fourth-order Runge-Kutta method.')
        solution = PDFText('After ', str(iterations), ' RK4 step', 's' if iterations != 1 else '', ', ',
                           PDFMath(rf'y({_number(target)})\approx{float(values[-1]):.6g}'), '.',
                           plain=str(values[-1])) if with_solution else None
        super().__init__(prompt, 'runge_kutta', solution, difficulty,
                         {'coefficient': coefficient, 'initial': initial, 'step': step,
                          'iterations': iterations, 'stages': tuple(stage_data),
                          'values': tuple(values), 'target': target, 'result': values[-1]})


_FACTORIES = {
    'difference_quotient': PDFDifferenceQuotient,
    'derivative': PDFDerivativeExercise,
    'tangent_line': PDFTangentLine,
    'critical_points': PDFCriticalPoints,
    'monotonicity': PDFMonotonicity,
    'concavity': PDFConcavity,
    'optimization': PDFOptimization,
    'definite_integral': PDFDefiniteIntegral,
    'area_between': PDFAreaBetweenCurves,
    'numerical_derivative': PDFNumericalDerivative,
    'trapezoidal_rule': PDFTrapezoidalRule,
    'simpson_rule': PDFSimpsonRule,
    'newton_iteration': PDFNewtonIteration,
    'euler_method': PDFEulerMethod,
    'runge_kutta': PDFRungeKuttaMethod,
}

_ALIASES = {
    'first_principles': 'difference_quotient', 'differentiate': 'derivative',
    'tangent': 'tangent_line', 'critical': 'critical_points',
    'increasing_decreasing': 'monotonicity', 'inflection': 'concavity',
    'integral': 'definite_integral', 'area': 'area_between',
    'central_difference': 'numerical_derivative', 'trapezoid': 'trapezoidal_rule',
    'simpson': 'simpson_rule', 'newton': 'newton_iteration', 'euler': 'euler_method',
    'rk4': 'runge_kutta', 'runge_kutta_method': 'runge_kutta',
}


def calculus_exercise(kind, *, difficulty='medium', seed=None, with_solution=True, _rng=None):
    """Create a calculus or numerical-method exercise by a friendly name."""
    if not isinstance(kind, str):
        raise TypeError('kind must be text')
    key = kind.strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        choices = ', '.join(CALCULUS_EXERCISE_TYPES)
        raise ValueError(f'Unknown calculus exercise {kind!r}; choose from {choices}') from exc
    return factory(with_solution=with_solution, difficulty=difficulty, seed=seed, _rng=_rng)


__all__ = [
    'PDFCalculusNumericalExercise', 'PDFDifferenceQuotient', 'PDFDerivativeExercise',
    'PDFTangentLine', 'PDFCriticalPoints', 'PDFMonotonicity', 'PDFConcavity',
    'PDFOptimization', 'PDFDefiniteIntegral', 'PDFAreaBetweenCurves',
    'PDFNumericalDerivative', 'PDFTrapezoidalRule', 'PDFSimpsonRule',
    'PDFNewtonIteration', 'PDFEulerMethod', 'PDFRungeKuttaMethod',
    'CALCULUS_EXERCISE_TYPES', 'calculus_exercise',
]
