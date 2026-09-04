from fractions import Fraction
import math

import pytest

import kiwicalc as kw
from kiwicalc.pdf import layout


KINDS = kw.SEQUENCE_SERIES_EXERCISE_TYPES


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('difficulty', ('easy', 'medium', 'hard'))
def test_every_sequence_generator_is_deterministic(kind, difficulty):
    first = kw.sequence_exercise(kind, difficulty=difficulty, seed=123)
    second = kw.sequence_exercise(kind, difficulty=difficulty, seed=123)
    assert isinstance(first, kw.PDFSequenceSeriesExercise)
    assert first.kind == kind and first.difficulty == difficulty
    assert str(first.exercise) == str(second.exercise)
    assert str(first.solution) == str(second.solution)
    assert first.data == second.data
    assert first.solution is not None
    assert kw.sequence_exercise(kind, difficulty=difficulty, seed=123,
                                with_solution=False).solution is None


def test_arithmetic_answer_invariants_and_sequence_bridge():
    for seed in range(40):
        identified = kw.PDFIdentifySequence(seed=seed, difficulty='hard').data
        if identified['sequence_type'] == 'arithmetic':
            assert all(b-a == identified['parameter'] for a, b in
                       zip(identified['terms'], identified['terms'][1:]))
        else:
            assert all(Fraction(b, a) == identified['parameter'] for a, b in
                       zip(identified['terms'], identified['terms'][1:]))

        upcoming = kw.PDFArithmeticNextTerms(seed=seed, difficulty='hard').data
        sequence = kw.ArithmeticProg(upcoming['first'], difference=upcoming['difference'])
        assert upcoming['shown'] == tuple(sequence.in_index(index) for index in range(1, 5))
        assert upcoming['result'] == tuple(sequence.in_index(index) for index in range(5, 9))

        nth = kw.PDFArithmeticNthTerm(seed=seed, difficulty='hard').data
        sequence = kw.ArithmeticProg(nth['first'], difference=nth['difference'])
        assert nth['result'] == sequence.in_index(nth['index'])

        difference = kw.PDFArithmeticDifference(seed=seed, difficulty='hard').data
        assert all(b-a == difference['result'] for a, b in
                   zip(difference['terms'], difference['terms'][1:]))

        total = kw.PDFArithmeticSum(seed=seed, difficulty='hard').data
        expected = Fraction(total['count']*(2*total['first']
                            +(total['count']-1)*total['difference']), 2)
        assert total['result'] == expected
        assert total['result'] == kw.ArithmeticProg(
            total['first'], difference=total['difference']).sum_first_n(total['count'])

        missing = kw.PDFArithmeticMissingTerm(seed=seed, difficulty='hard').data
        assert missing['result'] == missing['terms'][missing['position']-1]
        assert all(b-a == missing['difference'] for a, b in
                   zip(missing['terms'], missing['terms'][1:]))


def test_geometric_answer_invariants_and_sequence_bridge():
    for seed in range(40):
        upcoming = kw.PDFGeometricNextTerms(seed=seed, difficulty='hard').data
        sequence = kw.GeometricSeq(upcoming['first'], ratio=upcoming['ratio'])
        assert upcoming['shown'] == tuple(sequence.in_index(index) for index in range(1, 5))
        assert upcoming['result'] == tuple(sequence.in_index(index) for index in range(5, 9))

        nth = kw.PDFGeometricNthTerm(seed=seed, difficulty='hard').data
        sequence = kw.GeometricSeq(nth['first'], ratio=nth['ratio'])
        assert nth['result'] == sequence.in_index(nth['index'])

        ratio = kw.PDFGeometricRatio(seed=seed, difficulty='hard').data
        assert all(Fraction(b, a) == ratio['result'] for a, b in
                   zip(ratio['terms'], ratio['terms'][1:]))

        total = kw.PDFGeometricSum(seed=seed, difficulty='hard').data
        expected = Fraction(total['first']*(1-total['ratio']**total['count']),
                            1-total['ratio'])
        assert total['result'] == expected
        assert total['result'] == kw.GeometricSeq(
            total['first'], ratio=total['ratio']).sum_first_n(total['count'])

        infinite = kw.PDFInfiniteGeometricSum(seed=seed, difficulty='hard').data
        assert abs(infinite['ratio']) < 1
        assert infinite['result'] == Fraction(infinite['first'])/(1-infinite['ratio'])


def test_recurrence_sigma_and_limit_invariants():
    for seed in range(40):
        recursive = kw.PDFRecursiveSequence(seed=seed, difficulty='hard').data
        for previous, following in zip(recursive['terms'], recursive['terms'][1:]):
            assert following == recursive['multiplier']*previous+recursive['constant']
        assert recursive['result'] == recursive['terms'][-1]

        fibonacci = kw.PDFFibonacciSequence(seed=seed, difficulty='hard').data
        assert fibonacci['terms'][:2] == fibonacci['initial']
        assert all(current == left+right for left, right, current in
                   zip(fibonacci['terms'], fibonacci['terms'][1:], fibonacci['terms'][2:]))
        assert fibonacci['result'] == fibonacci['terms'][fibonacci['index']-1]

        sigma = kw.PDFSigmaEvaluation(seed=seed, difficulty='hard').data
        c, a, b = sigma['coefficients']
        assert sigma['terms'] == tuple(c*k*k+a*k+b for k in range(1, sigma['upper']+1))
        assert sigma['result'] == sum(sigma['terms'])

        for difficulty in ('easy', 'medium', 'hard'):
            limit = kw.PDFSequenceLimit(seed=seed, difficulty=difficulty).data
            numerator_degree, denominator_degree = limit['degree']
            assert numerator_degree == len(limit['numerator'])-1
            assert denominator_degree == len(limit['denominator'])-1
            assert limit['leading_ratio'] == Fraction(
                limit['numerator'][0], limit['denominator'][0])
            assert all(sum(coefficient*n**power for power, coefficient in enumerate(
                reversed(limit['denominator']))) != 0 for n in range(1, 30))
            if limit['case'] == 'finite_zero':
                assert numerator_degree < denominator_degree
                assert limit['exists'] and limit['result'] == limit['limit'] == 0
            elif limit['case'] == 'finite_ratio':
                assert numerator_degree == denominator_degree
                assert limit['exists']
                assert limit['result'] == limit['limit'] == limit['leading_ratio']
            elif limit['case'] in ('positive_infinity', 'negative_infinity'):
                assert numerator_degree > denominator_degree
                assert not limit['exists'] and limit['limit'] is None
                expected_sign = 1 if limit['case'] == 'positive_infinity' else -1
                assert (1 if limit['leading_ratio'] > 0 else -1) == expected_sign
            else:
                assert limit['case'] == 'oscillating'
                assert numerator_degree == denominator_degree
                assert not limit['exists'] and limit['limit'] is None


@pytest.mark.parametrize('case', kw.PDFSequenceLimit.CASES)
def test_sequence_limit_can_request_each_outcome(case):
    first = kw.sequence_exercise('sequence_limit', difficulty='hard', seed=17, case=case)
    second = kw.PDFSequenceLimit(difficulty='hard', seed=17, case=case)
    assert first.data == second.data
    assert first.data['case'] == case
    assert first.data['converges'] is first.data['exists']


def test_sequence_limit_random_generation_reaches_every_outcome():
    outcomes = {kw.PDFSequenceLimit(seed=seed, difficulty='hard').data['case']
                for seed in range(100)}
    assert outcomes == set(kw.PDFSequenceLimit.CASES)


@pytest.mark.parametrize('case', [1, object()])
def test_sequence_limit_rejects_non_text_case(case):
    with pytest.raises(TypeError, match='case'):
        kw.PDFSequenceLimit(case=case)


def test_sequence_limit_case_aliases_and_unknown_case():
    aliases = {'zero': 'finite_zero', 'finite': 'finite_ratio',
               '+infinity': 'positive_infinity', '-infinity': 'negative_infinity',
               'does not exist': 'oscillating', 'dne': 'oscillating'}
    for alias, expected in aliases.items():
        assert kw.sequence_exercise('limit', case=alias, seed=4).data['case'] == expected
    with pytest.raises(ValueError, match='case must be one of'):
        kw.PDFSequenceLimit(case='sideways')


@pytest.mark.parametrize('function', kw.PDFElementaryFunctionLimit.FUNCTIONS)
def test_elementary_limits_are_defined_and_exact(function):
    exercise = kw.sequence_exercise('elementary_limit', function=function,
                                    difficulty='hard', seed=23)
    data = exercise.data
    assert data['function'] == function
    assert data['defined_at_point'] is True
    point = data['point']
    if function == 'polynomial':
        expected = sum(coefficient*point**power for power, coefficient in
                       enumerate(reversed(data['coefficients'])))
        assert data['result'] == expected
    elif function == 'rational':
        numerator = sum(coefficient*point**power for power, coefficient in
                        enumerate(reversed(data['numerator'])))
        denominator = sum(coefficient*point**power for power, coefficient in
                          enumerate(reversed(data['denominator'])))
        assert denominator == data['denominator_value'] != 0
        assert data['result'] == Fraction(numerator, denominator)
    elif function == 'square_root':
        assert data['radicand'] == data['result']**2
    elif function == 'exponential':
        assert math.isclose(data['numeric_result'], math.exp(data['inner_value']))
    elif function == 'logarithm':
        assert data['inner_value'] > 0
        assert math.isclose(data['numeric_result'], math.log(data['inner_value']))
    elif function in ('sine', 'cosine'):
        angle = math.pi*float(data['pi_multiple'])
        expected = math.sin(angle) if function == 'sine' else math.cos(angle)
        assert math.isclose(data['numeric_result'], expected, abs_tol=1e-12)
    else:
        assert function == 'absolute_value'
        assert data['result'] == abs(data['inner_value'])


def test_euler_limits_match_closed_form_rapidly():
    for seed in range(30):
        data = kw.PDFEulerLimit(difficulty='hard', seed=seed).data
        assert data['exponent'] == data['base_increment']*data['multiplier']
        assert math.isclose(data['numeric_result'], math.exp(float(data['exponent'])))
        n = 100_000
        approximation = (1+float(data['base_increment'])/n)**(data['multiplier']*n)
        assert math.isclose(approximation, data['numeric_result'], rel_tol=1e-3)


def test_removable_limits_cancel_to_stored_exact_result():
    for seed in range(30):
        data = kw.PDFRemovableLimit(difficulty='hard', seed=seed).data
        point = data['point']
        top_slope, top_intercept = data['numerator_linear']
        bottom_slope, bottom_intercept = data['denominator_linear']
        assert bottom_slope*point+bottom_intercept != 0
        assert data['result'] == Fraction(top_slope*point+top_intercept,
                                          bottom_slope*point+bottom_intercept)
        assert sum(coefficient*point**power for power, coefficient in
                   enumerate(reversed(data['expanded_numerator']))) == 0
        assert sum(coefficient*point**power for power, coefficient in
                   enumerate(reversed(data['expanded_denominator']))) == 0


@pytest.mark.parametrize('form', kw.PDFStandardTrigLimit.FORMS)
def test_standard_trig_limits_use_closed_forms(form):
    data = kw.PDFStandardTrigLimit(form=form, difficulty='hard', seed=8).data
    expected = (Fraction(data['scale'], data['divisor'])
                if form != 'one_minus_cosine'
                else Fraction(data['scale']**2, 2*data['divisor']))
    assert data['form'] == form
    assert data['result'] == expected


def test_limit_subtype_aliases_and_validation():
    aliases = {'elementary': 'elementary_limit', 'e limit': 'euler_limit',
               'hole limit': 'removable_limit', 'trig limit': 'standard_trig_limit'}
    for alias, kind in aliases.items():
        assert kw.sequence_exercise(alias, seed=3).kind == kind
    assert kw.PDFElementaryFunctionLimit(function='sqrt', seed=1).data['function'] == 'square_root'
    assert kw.PDFStandardTrigLimit(form='sin', seed=1).data['form'] == 'sine_ratio'
    with pytest.raises(ValueError, match='function must be one of'):
        kw.PDFElementaryFunctionLimit(function='secant')
    with pytest.raises(TypeError, match='function'):
        kw.PDFElementaryFunctionLimit(function=1)
    with pytest.raises(ValueError, match='form must be one of'):
        kw.PDFStandardTrigLimit(form='secant')
    with pytest.raises(TypeError, match='form'):
        kw.PDFStandardTrigLimit(form=1)


def test_infinite_series_classification_invariants():
    for seed in range(40):
        sequence = kw.PDFConvergenceClassification(seed=seed, difficulty='hard').data
        assert sequence['converges'] is (abs(sequence['ratio']) < 1)
        assert sequence['limit'] == (0 if sequence['converges'] else None)

        pseries = kw.PDFPSeries(seed=seed, difficulty='hard').data
        assert pseries['converges'] is (pseries['power'] > 1)
        assert pseries['result'] == ('converges' if pseries['converges'] else 'diverges')

        geometric = kw.PDFGeometricSeriesTest(seed=seed, difficulty='hard').data
        assert geometric['converges'] is (abs(geometric['ratio']) < 1)
        expected = Fraction(geometric['first'])/(1-geometric['ratio']) if geometric['converges'] else None
        assert geometric['sum'] == expected

        alternating = kw.PDFAlternatingSeries(seed=seed, difficulty='hard').data
        expected = 'conditionally' if alternating['power'] == 1 else 'absolutely'
        assert alternating['classification'] == alternating['result'] == expected

        telescoping = kw.PDFTelescopingSeries(seed=seed, difficulty='hard').data
        assert telescoping['result'] == sum(telescoping['terms'])
        assert telescoping['result'] == Fraction(telescoping['count'], telescoping['count']+1)


def test_factory_aliases_and_validation():
    aliases = {
        'identify': 'identify_sequence', 'arithmetic next': 'arithmetic_next_terms',
        'arithmetic term': 'arithmetic_nth_term', 'common difference': 'arithmetic_difference',
        'sum arithmetic': 'arithmetic_sum', 'missing term': 'arithmetic_missing_term',
        'geometric next': 'geometric_next_terms', 'geometric term': 'geometric_nth_term',
        'common ratio': 'geometric_ratio', 'sum geometric': 'geometric_sum',
        'geometric infinite sum': 'infinite_geometric_sum', 'recurrence': 'recursive_sequence',
        'fibonacci sequence': 'fibonacci', 'sigma': 'sigma_evaluation',
        'limit': 'sequence_limit', 'convergence': 'convergence_classification',
        'pseries': 'p_series', 'geometric test': 'geometric_series_test',
        'alternating test': 'alternating_series', 'telescoping': 'telescoping_series',
        'elementary': 'elementary_limit', 'e limit': 'euler_limit',
        'hole limit': 'removable_limit', 'trig limit': 'standard_trig_limit',
    }
    for alias, expected in aliases.items():
        assert kw.sequence_exercise(alias, seed=4).kind == expected
    with pytest.raises(ValueError, match='Unknown sequence exercise'):
        kw.sequence_exercise('telepathy')
    with pytest.raises(TypeError, match='kind'):
        kw.sequence_exercise(1)
    with pytest.raises(ValueError, match='difficulty'):
        kw.sequence_exercise('fibonacci', difficulty='expert')
    with pytest.raises(TypeError, match='seed'):
        kw.sequence_exercise('fibonacci', seed=True)
    with pytest.raises(ValueError, match='cannot both'):
        kw.sequence_exercise('fibonacci', seed=1, _rng=__import__('random').Random(1))


@pytest.mark.parametrize('kind', KINDS)
def test_every_family_integrates_with_batch_worksheet(kind, tmp_path, monkeypatch):
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
    kw.worksheet(tmp_path/'series.pdf', dtype='p_series', equations_per_page=2,
                 get_solutions=False, seed=7)
    assert len(calls[0][0][1]) == 1
    with pytest.raises(ValueError, match='one title'):
        kw.worksheet(tmp_path/'bad.pdf', dtype='geometric_sum', num_of_pages=2,
                     titles=['Only one'], seed=7)


def test_all_sequence_math_renders_in_one_document(tmp_path):
    sheet = kw.PDFWorksheet('Sequences and Series', theme='academic')
    for index, kind in enumerate(KINDS):
        sheet.add_exercise(kw.sequence_exercise(kind, difficulty='hard', seed=index))
    sheet.end_page().create(tmp_path/'sequences.pdf')
    assert (tmp_path/'sequences.pdf').read_bytes().startswith(b'%PDF')
