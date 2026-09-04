import importlib
from fractions import Fraction
import re

import pytest

import kiwicalc as kw

m = importlib.import_module('kiwicalc.pdf.worksheet')


def exercise(text='x=1', solution=1):
    return kw.PDFExercise(text, 'equation', 'linear', solution=solution)


@pytest.mark.parametrize('coords', [(2, 3, 2, 5), (15, 3, 15, 5), (0, 0, 3, 1)])
def test_two_point_answers_are_exact_and_nonvertical(monkeypatch, coords):
    numbers = iter(coords)
    monkeypatch.setattr(m.random, 'randint', lambda *_: next(numbers))
    prompt, answer = m.linear_from_points_exercise()
    points = [tuple(map(int, match)) for match in re.findall(r'\((-?\d+), (-?\d+)\)', prompt)]
    assert points[0][0] != points[1][0]
    slope = Fraction(points[1][1]-points[0][1], points[1][0]-points[0][0])
    intercept = points[0][1]-slope*points[0][0]
    assert all(slope*x+intercept == y for x, y in points)
    if slope.denominator != 1:
        assert f'({slope})x' in answer
    assert 'Increasing' in answer if slope > 0 else 'Decreasing' in answer


@pytest.mark.parametrize('point,slope', [((2, 3), 4), ((-2, 5), -2), ((3, 0), 2)])
def test_point_slope_x_intercept_satisfies_the_line(monkeypatch, point, slope):
    numbers = iter((*point, slope))
    monkeypatch.setattr(m.random, 'randint', lambda *_: next(numbers))
    _, answer = m.linearFromPointAndSlope_exercise()
    value = float(re.search(r'b\.\s+\(([^,]+), 0\)', answer).group(1))
    assert slope*value + point[1]-slope*point[0] == pytest.approx(0, abs=1e-4)


@pytest.fixture
def rendered(monkeypatch):
    output = []
    monkeypatch.setattr(m, 'create_pages', lambda path, count, titles, lines: output.append((count, titles, lines)))
    return output


def test_repeated_end_page_and_add_after_answers_stay_consistent(rendered):
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise())
    question_page = sheet.current_page
    sheet.end_page()
    sheet.end_page()
    assert sheet.num_of_pages == 2 and sheet.current_page is question_page
    sheet.add_exercise(exercise('x=2', 2))
    sheet.create('unused.pdf')
    count, titles, lines = rendered[-1]
    assert count == 2 and titles[-1] == 'Solutions'
    assert lines == [['1. x=1', '2. x=2'], ['1. 1', '2. 2']]


def test_delete_question_then_add_targets_surviving_question_page(rendered):
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise())
    first = sheet.current_page
    sheet.end_page()
    sheet.next_page('discard')
    sheet.add_exercise(exercise('discarded', 99))
    sheet.del_last_page()
    assert sheet.current_page is first
    added = exercise('x=2', 2)
    sheet.add_exercise(added)
    assert added.number == 2
    sheet.create('unused.pdf')
    assert 'discarded' not in str(rendered[-1])
    assert rendered[-1][2][1] == ['1. 1', '2. 2']


def test_delete_answers_and_recreate_once(rendered):
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise())
    sheet.end_page()
    sheet.del_last_page()
    sheet.end_page()
    sheet.end_page()
    assert sheet.num_of_pages == 2
    sheet.create('unused.pdf')
    assert rendered[-1][0] == 2


def test_delete_every_page_and_recover(rendered):
    sheet = kw.PDFWorksheet()
    sheet.del_last_page()
    sheet.del_last_page()
    assert sheet.num_of_pages == 0 and sheet.current_page is None
    with pytest.raises(ValueError, match='no pages'):
        sheet.create('unused.pdf')
    sheet.add_exercise(exercise())
    assert sheet.current_page is sheet.pages[0]
    sheet.create('unused.pdf')
    assert rendered[-1][2] == [['1. x=1']]


def test_direct_page_edits_are_reflected_and_create_is_repeatable(rendered):
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise())
    sheet.end_page()
    sheet.current_page.exercises[:] = [exercise('replacement\nshow work', [2, 3])]
    sheet.create('unused.pdf')
    first = rendered[-1]
    sheet.create('unused.pdf')
    assert first == rendered[-1]
    assert first[2] == [['1. replacement\nshow work'], ['1. 2,3']]
    sheet.current_page.exercises.clear()
    sheet.create('unused.pdf')
    assert sheet.num_of_pages == 1 and rendered[-1][2] == [[]]


def test_unordered_answers_and_independent_page_iterators(rendered):
    sheet = kw.PDFWorksheet(ordered=False)
    sheet.add_exercise(exercise('question\npart b', 'answer\nexplanation'))
    sheet.end_page()
    sheet.create('unused.pdf')
    assert rendered[-1][2] == [['question\npart b'], ['answer\nexplanation']]
    items = [exercise('a'), exercise('b')]
    page = kw.PDFPage(exercises=items)
    assert len([(a, b) for a in page for b in page]) == 4
    items.clear()
    assert len(list(page)) == 2


def test_generator_answers_survive_repeated_creates(rendered):
    sheet = kw.PDFWorksheet()
    sheet.add_exercise(exercise(solution=(value for value in [1, 2])))
    sheet.end_page()
    sheet.create('unused.pdf')
    first = rendered[-1]
    sheet.create('unused.pdf')
    assert first == rendered[-1]
    assert first[2][-1] == ['1. 1,2']


def test_invalid_exercise_is_rejected_without_mutation():
    sheet = kw.PDFWorksheet()
    with pytest.raises(TypeError, match='PDFExercise'):
        sheet.add_exercise('not an exercise')
    assert sheet.current_page.exercises == []


@pytest.mark.parametrize('count', [0, -1, True, 2.5])
def test_invalid_system_sizes_fail_early(count):
    with pytest.raises(ValueError, match='positive integer'):
        m.PDFLinearSystem(num_of_equations=count)


def test_invalid_page_arrays_do_not_open_output(monkeypatch):
    monkeypatch.setattr(m, 'Canvas', lambda *args: pytest.fail('Must validate before opening output'))
    for count, titles, lines in [(1, [], [[]]), (2, ['a', 'b'], [[]]), (-1, [], []), (True, [], [])]:
        with pytest.raises(ValueError):
            m.create_pages('unused.pdf', count, titles, lines)
