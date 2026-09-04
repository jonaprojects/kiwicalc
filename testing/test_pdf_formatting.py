from fractions import Fraction
import pytest
import kiwicalc as kw


@pytest.mark.parametrize('value,expected', [(Fraction(-2, 3), r'-\frac{2}{3}'),
    (Fraction(4, 2), '2'), (0, '0'), (0.125, '0.125'), (1e-8, r'1\times 10^{-8}')])
def test_exact_numbers(value, expected):
    assert kw.format_math(value) == expected


def test_polynomial_notation():
    assert kw.format_polynomial([1, 0, -1, 0]) == 'x^{3} -x'
    assert kw.format_polynomial([Fraction(1, 3), -2]) == r'\frac{1}{3}x -2'
    assert kw.format_polynomial([]) == '0'
    assert kw.format_polynomial([0, 0]) == '0'
    for value in [True, float('nan'), object()]:
        with pytest.raises((TypeError, ValueError)):
            kw.format_math(value)
    with pytest.raises(ValueError):
        kw.format_polynomial([1], variable='not a variable')


def test_rich_text_survives_numbering_answers_and_render(tmp_path):
    question = kw.PDFText('Solve <this> & ', kw.PDFMath(r'\frac{x}{2}=3', font_size=12), '.')
    sheet = kw.PDFWorksheet('Formatted worksheet')
    sheet.add_exercise(kw.PDFExercise(question, 'equation', 'linear',
                                    solution=kw.PDFText('Answer: ', kw.PDFMath('x=6'))))
    sheet.end_page()
    assert isinstance(sheet.pages[1].exercises[0], kw.PDFText)
    assert sheet.pages[1].exercises[0].startswith('1. ')
    sheet.create(tmp_path/'mixed.pdf')
    assert (tmp_path/'mixed.pdf').read_bytes().startswith(b'%PDF')
    with pytest.raises(TypeError):
        kw.PDFText(42)
    with pytest.raises(ValueError, match='Unsupported Mathtext'):
        kw.PDFMath(r'\unknowncommand{x}').image()


@pytest.mark.parametrize('factory', [kw.PDFTrigonometricEquation, kw.PDFLogarithmicEquation, kw.PDFLinearIntersection])
def test_generators_keep_plain_text_and_rich_content(factory, tmp_path):
    exercise = factory(seed=7)
    assert isinstance(exercise.exercise, kw.PDFText)
    assert isinstance(exercise.solution, kw.PDFText)
    assert str(exercise) == exercise.exercise
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise)
    sheet.end_page()
    sheet.create(tmp_path/'generated.pdf')
