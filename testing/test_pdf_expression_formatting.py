import importlib
from fractions import Fraction
import pytest
import kiwicalc as kw
from kiwicalc.pdf.formatting import _equation_text


def test_objects_without_mutation():
    mono = kw.Mono(-1, {'x': 2, 'y': -1})
    before = mono.to_dict()
    assert kw.format_math(mono) == '-x^{2}y^{-1}'
    assert mono.to_dict() == before
    poly = kw.Poly('x^2-2x+1')
    before = poly.to_dict()
    assert kw.format_math(poly) == 'x^{2} -2x +1'
    assert poly.to_dict() == before
    assert kw.format_math(kw.Mono(0, {'x': 3})) == '0'
    assert kw.format_math(kw.Poly(0)) == '0'
    mono = kw.Mono(1, {'x': Fraction(1, 2)})
    assert kw.format_math(mono) == r'x^{\frac{1}{2}}'
    with pytest.raises(ValueError):
        kw.format_math(kw.Mono(1, {'unsafe_name': 2}))


def test_equations_preserve_unsimplified_terms():
    equation = '2x+3x-1 = 4x+8'
    rich = _equation_text(equation)
    assert str(rich) == equation
    assert rich.parts[0].expression == equation
    assert _equation_text('Ordinary prose: x < 3') == 'Ordinary prose: x < 3'
    answer = kw.PDFText('\n   Answer: ', kw.PDFMath('x=1')).numbered(2)
    assert answer.parts[0] == 'Answer: '
    assert str(answer).startswith('2. \n')


@pytest.mark.parametrize('factory', [kw.PDFLinearEquation, kw.PDFQuadraticEquation,
    kw.PDFCubicEquation, kw.PDFQuarticEquation, kw.PDFPolyEquation,
    kw.PDFLinearSystem, kw.PDFLinearFunction, kw.PDFQuadraticFunction])
def test_legacy_classes_render_rich_questions(factory, tmp_path):
    exercise = factory()
    assert isinstance(exercise.exercise, kw.PDFText)
    sheet = kw.PDFWorksheet('Expression formatting')
    sheet.add_exercise(exercise)
    sheet.end_page()
    sheet.create(tmp_path/'exercise.pdf')
    assert (tmp_path/'exercise.pdf').read_bytes().startswith(b'%PDF')
    assert isinstance(factory(with_solution=False).exercise, kw.PDFText)


def test_batch_polynomial_keeps_whole_expression(monkeypatch):
    single = importlib.import_module('kiwicalc.equations.single')
    worksheet = importlib.import_module('kiwicalc.pdf.worksheet')
    calls = []
    monkeypatch.setattr(single, 'random_polynomial', lambda *a, **k: '12x^3-7x+4')
    monkeypatch.setattr(worksheet, 'create_pages', lambda *a, **k: calls.append((a, k)))
    kw.PolyEquation.random_worksheets('unused.pdf', num_of_pages=1, equations_per_page=2)
    questions = calls[0][0][3][0]
    assert questions == ['1. 12x^3-7x+4 = 0', '2. 12x^3-7x+4 = 0']
    assert all(isinstance(q, kw.PDFText) for q in questions)


def test_quadratic_batch_numbers_and_numeric_answers(monkeypatch):
    worksheet = importlib.import_module('kiwicalc.pdf.worksheet')
    calls = []
    monkeypatch.setattr(kw.QuadraticEquation, 'random', lambda **k: ('x^2-1=0', [-1, 1]))
    monkeypatch.setattr(worksheet, 'create_pages', lambda *a, **k: calls.append(k))
    kw.QuadraticEquation.random_worksheets('unused.pdf', num_of_pages=1,
                                         equations_per_page=2, get_solutions=True)
    questions, answers = calls[0]['lines']
    assert questions == ['1. x^2-1=0', '2. x^2-1=0']
    assert answers == ['1. -1, 1', '2. -1, 1']
