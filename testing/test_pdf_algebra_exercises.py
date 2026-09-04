import pytest
import kiwicalc as kw
from kiwicalc.pdf import layout


KINDS = kw.ALGEBRA_EXERCISE_TYPES


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('difficulty', ('easy', 'medium', 'hard'))
def test_every_algebra_generator_is_deterministic(kind, difficulty):
    first = kw.algebra_exercise(kind, difficulty=difficulty, seed=123)
    second = kw.algebra_exercise(kind, difficulty=difficulty, seed=123)
    assert isinstance(first, kw.PDFAlgebraExercise)
    assert first.kind == kind and first.difficulty == difficulty
    assert str(first.exercise) == str(second.exercise)
    assert str(first.solution) == str(second.solution)
    assert first.data == second.data
    assert first.solution is not None
    assert kw.algebra_exercise(kind, difficulty=difficulty, seed=123,
                               with_solution=False).solution is None


def test_generated_answer_invariants_across_many_seeds():
    for seed in range(50):
        simplify = kw.PDFSimplifyExpression(seed=seed, difficulty='hard').data
        terms = simplify['terms']
        assert simplify['coefficients'] == (
            sum(v for v, p in terms if p == 2),
            sum(v for v, p in terms if p == 1),
            sum(v for v, p in terms if p == 0),
        )

        factor = kw.PDFFactorPolynomial(seed=seed, difficulty='hard').data
        r1, r2 = factor['roots']
        scale = factor['scale']
        assert factor['coefficients'] == (scale, -scale*(r1+r2), scale*r1*r2)

        square = kw.PDFCompleteSquare(seed=seed).data
        h, k = square['shift'], square['remainder']
        assert square['coefficients'] == (1, 2*h, h*h+k)

        substitution = kw.PDFSubstitution(seed=seed, difficulty='hard').data
        coefficients, value = substitution['coefficients'], substitution['value']
        degree = len(coefficients)-1
        assert substitution['result'] == sum(
            coefficient*value**(degree-index)
            for index, coefficient in enumerate(coefficients)
        )

        inequality = kw.PDFLinearInequality(seed=seed, difficulty='hard').data
        assert inequality['coefficient'] < 0
        assert inequality['right'] == (inequality['coefficient']*inequality['boundary']
                                       + inequality['constant'])

        absolute = kw.PDFAbsoluteValueEquation(seed=seed).data
        assert absolute['roots'] == tuple(sorted((absolute['center']-absolute['radius'],
                                                   absolute['center']+absolute['radius'])))

        exponents = kw.PDFExponentLaws(seed=seed).data
        a, b, c = exponents['exponents']
        assert exponents['result_exponent'] == a+b-c > 0

        rational = kw.PDFRationalEquation(seed=seed).data
        assert rational['root'] != rational['excluded']
        assert rational['numerator']/(rational['root']-rational['excluded']) == rational['right']

        radical = kw.PDFRadicalEquation(seed=seed).data
        assert radical['root']+radical['offset'] == radical['right']**2


def test_factory_aliases_and_validation():
    aliases = {'simplifying': 'simplify', 'expanding': 'expand', 'factoring': 'factor',
               'evaluate': 'substitution', 'inequality': 'linear_inequality',
               'absolute': 'absolute_value', 'exponents': 'exponent_laws',
               'rational equation': 'rational', 'radical-equation': 'radical',
               'rearrange formula': 'rearrange'}
    for alias, expected in aliases.items():
        assert kw.algebra_exercise(alias, seed=4).kind == expected
    with pytest.raises(ValueError, match='Unknown algebra exercise'):
        kw.algebra_exercise('telepathy')
    with pytest.raises(TypeError, match='kind'):
        kw.algebra_exercise(1)
    with pytest.raises(ValueError, match='difficulty'):
        kw.algebra_exercise('factor', difficulty='expert')
    with pytest.raises(TypeError, match='seed'):
        kw.algebra_exercise('factor', seed=True)


@pytest.mark.parametrize('kind', KINDS)
def test_every_algebra_family_integrates_with_batch_worksheet(kind, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append((args, kwargs)))
    kw.worksheet(tmp_path/f'{kind}.pdf', dtype=kind, equations_per_page=3,
                 get_solutions=True, difficulty='hard', seed=42, theme='assessment')
    _, titles, pages = calls[0][0]
    assert len(titles) == len(pages) == 2
    assert titles[1].endswith('Solutions')
    assert len(pages[0]) == len(pages[1]) == 3
    assert calls[0][1]['theme'] == 'assessment'


def test_all_algebra_math_renders_in_one_document(tmp_path):
    sheet = kw.PDFWorksheet('Core Algebra', theme='classroom')
    for index, kind in enumerate(KINDS):
        assert sheet.add_exercise(kw.algebra_exercise(kind, difficulty='hard', seed=index)) is sheet
    assert sheet.end_page() is sheet
    sheet.create(tmp_path/'algebra.pdf')
    assert (tmp_path/'algebra.pdf').read_bytes().startswith(b'%PDF')
