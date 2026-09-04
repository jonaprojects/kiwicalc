from fractions import Fraction
from io import BytesIO

import numpy as np
import pytest

import kiwicalc as kw


def test_matrix_accepts_common_data_sources_and_is_detached():
    sources = (
        [[1, 2], [3, 4]],
        np.array([[1, 2], [3, 4]]),
        kw.Matrix([[1, 2], [3, 4]]),
    )
    for source in sources:
        rendered = kw.PDFMatrix(source)
        assert rendered.values == ((1, 2), (3, 4))
        assert rendered.shape == (2, 2)
        assert str(rendered) == '[1, 2; 3, 4]'
    values = [[1, 2], [3, 4]]
    rendered = kw.PDFMatrix(values)
    values[0][0] = 99
    assert rendered.values[0][0] == 1


def test_matrix_treats_a_flat_sequence_as_a_row():
    matrix = kw.PDFMatrix([1, 2, 3], brackets='none')
    assert matrix.values == ((1, 2, 3),)
    assert matrix.shape == (1, 3)


def test_vector_accepts_geometry_matrix_numpy_and_sequences():
    sources = (
        [1, 2, 3],
        np.array([1, 2, 3]),
        np.array([[1], [2], [3]]),
        kw.Matrix.column_vector([1, 2, 3]),
        kw.Vector((1, 2, 3)),
    )
    for source in sources:
        vector = kw.PDFVector(source)
        assert vector.values == ((1,), (2,), (3,))
        assert vector.shape == (3, 1)
        assert vector.orientation == 'column'
    row = kw.PDFVector(kw.Matrix([[1, 2, 3]]), orientation='row')
    assert row.values == ((1, 2, 3),)


@pytest.mark.parametrize('brackets', ('square', 'round', 'determinant', 'none',
                                      'brackets', 'parentheses', 'bars', None))
def test_every_bracket_style_renders_without_latex(brackets):
    image = kw.PDFMatrix([[Fraction(1, 2), -3], ['x^2', 1+2j]], brackets=brackets).image()
    assert isinstance(image, BytesIO)
    assert image.read(8) == b'\x89PNG\r\n\x1a\n'


def test_array_validation_is_clear_and_early():
    invalid = (
        lambda: kw.PDFMatrix([]),
        lambda: kw.PDFMatrix([[], []]),
        lambda: kw.PDFMatrix([[1], [2, 3]]),
        lambda: kw.PDFMatrix('123'),
        lambda: kw.PDFMatrix([[True]]),
        lambda: kw.PDFMatrix([[float('inf')]]),
        lambda: kw.PDFMatrix([[1]], brackets='curly'),
        lambda: kw.PDFMatrix([[1]], font_size=0),
        lambda: kw.PDFVector([]),
        lambda: kw.PDFVector([[1, 2], [3, 4]]),
        lambda: kw.PDFVector([1], orientation='diagonal'),
        lambda: kw.PDFVector(range(31)),
        lambda: kw.PDFMatrix([range(31)]),
    )
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_unsupported_cell_math_has_a_contextual_error():
    matrix = kw.PDFMatrix([[r'\unknowncommand{x}']])
    with pytest.raises(ValueError, match='Unsupported matrix entry'):
        matrix.image()


def test_matrix_and_vector_work_inline_and_as_display_blocks(tmp_path):
    matrix = kw.PDFMatrix([[1, Fraction(2, 3)], ['x', -4]])
    vector = kw.PDFVector([5, -2])
    question = kw.PDFText('Compute ', matrix, vector, '.')
    assert 'Compute' in str(question)

    sheet = kw.PDFWorksheet('Matrix and Vector Notation', theme='academic')
    assert sheet.add_exercise(kw.PDFExercise(question, 'linear algebra', 'product',
        solution=kw.PDFText('Result: ', kw.PDFVector([Fraction(11, 3), 13])))) is sheet
    assert sheet.add_matrix(kw.Matrix([[1, 2, 3], [4, 5, 6]]),
                            brackets='determinant') is sheet
    assert sheet.add_vector(np.array([1, -1, 2]), orientation='row') is sheet
    sheet.end_page().create(tmp_path/'arrays.pdf')
    assert (tmp_path/'arrays.pdf').read_bytes().startswith(b'%PDF')
    assert sheet.pages[0].exercises[0].number == 1


def test_inline_matrix_obeys_page_width_guard(tmp_path):
    wide = kw.PDFMatrix([['x^{1000}']*20], font_size=24)
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(kw.PDFExercise(kw.PDFText('Wide: ', wide), 'test', 'wide'))
    with pytest.raises(ValueError, match='wider than the page'):
        sheet.create(tmp_path/'wide.pdf')
