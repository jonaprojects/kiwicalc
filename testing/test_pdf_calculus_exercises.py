from fractions import Fraction

import pytest

import kiwicalc as kw
from kiwicalc.pdf import layout


KINDS = kw.CALCULUS_EXERCISE_TYPES


def evaluate(coefficients, x):
    value = 0
    for coefficient in coefficients:
        value = value*x+coefficient
    return value


def derivative(coefficients):
    degree = len(coefficients)-1
    return tuple(coefficient*(degree-index)
                 for index, coefficient in enumerate(coefficients[:-1]))


def antiderivative_value(coefficients, x):
    degree = len(coefficients)-1
    return sum(Fraction(coefficient, degree-index+1)*x**(degree-index+1)
               for index, coefficient in enumerate(coefficients))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('difficulty', ('easy', 'medium', 'hard'))
def test_every_calculus_generator_is_deterministic(kind, difficulty):
    first = kw.calculus_exercise(kind, difficulty=difficulty, seed=123)
    second = kw.calculus_exercise(kind, difficulty=difficulty, seed=123)
    assert isinstance(first, kw.PDFCalculusNumericalExercise)
    assert first.kind == kind and first.difficulty == difficulty
    assert str(first.exercise) == str(second.exercise)
    assert str(first.solution) == str(second.solution)
    assert first.data == second.data
    assert first.solution is not None
    assert kw.calculus_exercise(kind, difficulty=difficulty, seed=123,
                                with_solution=False).solution is None


def test_symbolic_calculus_answer_invariants_across_many_seeds():
    for seed in range(30):
        quotient = kw.PDFDifferenceQuotient(seed=seed, difficulty='hard').data
        assert quotient['derivative_coefficients'] == derivative(quotient['coefficients'])
        assert quotient['result'] == evaluate(quotient['derivative_coefficients'], quotient['point'])

        tangent = kw.PDFTangentLine(seed=seed, difficulty='hard').data
        assert tangent['value'] == evaluate(tangent['coefficients'], tangent['point'])
        assert tangent['slope'] == evaluate(derivative(tangent['coefficients']), tangent['point'])

        critical = kw.PDFCriticalPoints(seed=seed, difficulty='hard').data
        derivative_coefficients = derivative(critical['coefficients'])
        assert all(evaluate(derivative_coefficients, point) == 0 for point in critical['points'])
        expected = ('maximum', 'minimum') if critical['coefficients'][0] > 0 else ('minimum', 'maximum')
        assert critical['classifications'] == expected

        monotonicity = kw.PDFMonotonicity(seed=seed, difficulty='hard').data
        radius = monotonicity['critical'][1]
        if monotonicity['coefficients'][0] > 0:
            assert monotonicity['increasing'] == ((rf'(-\infty,-{radius})', rf'({radius},\infty)'))
        else:
            assert monotonicity['increasing'] == ((rf'(-{radius},{radius})',))

        concavity = kw.PDFConcavity(seed=seed, difficulty='hard').data
        second_derivative = derivative(derivative(concavity['coefficients']))
        assert evaluate(second_derivative, concavity['inflection'][0]) == 0

        optimum = kw.PDFOptimization(seed=seed, difficulty='hard').data
        width, height = optimum['dimensions']
        assert 2*(width+height) == optimum['perimeter']
        assert width*height == optimum['maximum_area']

        integral = kw.PDFDefiniteIntegral(seed=seed, difficulty='hard').data
        lower, upper = integral['bounds']
        assert derivative(integral['antiderivative_coefficients']) == integral['coefficients']
        assert integral['result'] == (
            evaluate(integral['antiderivative_coefficients'], upper)
            - evaluate(integral['antiderivative_coefficients'], lower)
        )

        area = kw.PDFAreaBetweenCurves(seed=seed, difficulty='hard').data
        assert area['area'] == Fraction(8*area['radius']**3, 3)


def test_numerical_method_answer_invariants_across_many_seeds():
    for seed in range(30):
        numerical = kw.PDFNumericalDerivative(seed=seed, difficulty='hard').data
        (left, left_value), (right, right_value) = numerical['samples']
        assert numerical['result'] == (right_value-left_value)/(right-left)
        assert numerical['result'] == evaluate(derivative(numerical['coefficients']), numerical['point'])

        trapezoid = kw.PDFTrapezoidalRule(seed=seed, difficulty='hard').data
        values, step = trapezoid['values'], trapezoid['step']
        assert trapezoid['result'] == step*(values[0]+values[-1]+2*sum(values[1:-1]))/2
        lower, upper = trapezoid['bounds']
        exact = (antiderivative_value(trapezoid['coefficients'], upper)
                 - antiderivative_value(trapezoid['coefficients'], lower))
        assert trapezoid['exact'] == exact
        assert trapezoid['absolute_error'] == abs(trapezoid['result']-exact)

        simpson = kw.PDFSimpsonRule(seed=seed, difficulty='hard').data
        values, step = simpson['values'], simpson['step']
        result = step*(values[0]+values[-1]+4*sum(values[1:-1:2])
                       + 2*sum(values[2:-1:2]))/3
        assert simpson['result'] == result == simpson['exact']

        newton = kw.PDFNewtonIteration(seed=seed, difficulty='hard').data
        for current, following in zip(newton['values'], newton['values'][1:]):
            assert following == (current+Fraction(newton['radicand'], current))/2

        euler = kw.PDFEulerMethod(seed=seed, difficulty='hard').data
        for current, following in zip(euler['values'], euler['values'][1:]):
            assert following == current+euler['step']*euler['coefficient']*current

        rk4 = kw.PDFRungeKuttaMethod(seed=seed, difficulty='hard').data
        for current, following, stages in zip(rk4['values'], rk4['values'][1:], rk4['stages']):
            k1, k2, k3, k4 = stages
            h, coefficient = rk4['step'], rk4['coefficient']
            assert k1 == coefficient*current
            assert k2 == coefficient*(current+h*k1/2)
            assert k3 == coefficient*(current+h*k2/2)
            assert k4 == coefficient*(current+h*k3)
            assert following == current+h*(k1+2*k2+2*k3+k4)/6


def test_factory_aliases_and_validation():
    aliases = {
        'first principles': 'difference_quotient', 'differentiate': 'derivative',
        'tangent': 'tangent_line', 'critical': 'critical_points',
        'increasing-decreasing': 'monotonicity', 'inflection': 'concavity',
        'integral': 'definite_integral', 'area': 'area_between',
        'central difference': 'numerical_derivative', 'trapezoid': 'trapezoidal_rule',
        'simpson': 'simpson_rule', 'newton': 'newton_iteration',
        'euler': 'euler_method', 'rk4': 'runge_kutta',
    }
    for alias, expected in aliases.items():
        assert kw.calculus_exercise(alias, seed=4).kind == expected
    with pytest.raises(ValueError, match='Unknown calculus exercise'):
        kw.calculus_exercise('telepathy')
    with pytest.raises(TypeError, match='kind'):
        kw.calculus_exercise(1)
    with pytest.raises(ValueError, match='difficulty'):
        kw.calculus_exercise('derivative', difficulty='expert')
    with pytest.raises(TypeError, match='seed'):
        kw.calculus_exercise('derivative', seed=True)
    with pytest.raises(ValueError, match='cannot both'):
        kw.calculus_exercise('derivative', seed=1, _rng=__import__('random').Random(1))


@pytest.mark.parametrize('kind', KINDS)
def test_every_calculus_family_integrates_with_batch_worksheet(kind, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append((args, kwargs)))
    kw.worksheet(tmp_path/f'{kind}.pdf', dtype=kind, equations_per_page=3,
                 get_solutions=True, difficulty='hard', seed=42, theme='assessment')
    _, titles, pages = calls[0][0]
    assert len(titles) == len(pages) == 2
    assert titles[1].endswith('Solutions')
    assert len(pages[0]) == len(pages[1]) == 3
    assert calls[0][1]['theme'] == 'assessment'


def test_batch_without_solutions_and_title_validation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append((args, kwargs)))
    kw.worksheet(tmp_path/'derivatives.pdf', dtype='derivative', equations_per_page=2,
                 get_solutions=False, seed=7)
    assert len(calls[0][0][1]) == 1
    with pytest.raises(ValueError, match='one title'):
        kw.worksheet(tmp_path/'bad.pdf', dtype='derivative', num_of_pages=2,
                     titles=['Only one'], seed=7)


def test_all_calculus_math_renders_in_one_document(tmp_path):
    sheet = kw.PDFWorksheet('Calculus and Numerical Methods', theme='academic')
    for index, kind in enumerate(KINDS):
        assert sheet.add_exercise(kw.calculus_exercise(kind, difficulty='hard', seed=index)) is sheet
    assert sheet.end_page() is sheet
    sheet.create(tmp_path/'calculus.pdf')
    assert (tmp_path/'calculus.pdf').read_bytes().startswith(b'%PDF')
