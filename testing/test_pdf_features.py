import importlib
import math
import random
import re
from fractions import Fraction
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw
from kiwicalc.pdf import generators
from kiwicalc.pdf.layout import render_pages


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('value', [-1, Fraction(-1, 2), 0, Fraction(1, 2), 1])
def test_trig_answers_are_complete_on_stated_interval(name, value):
    rng = Mock()
    rng.choice.side_effect = [name, value]
    prompt, answer = generators.trigonometric(rng=rng)
    reported = [int(v) for v in answer.split(' = ')[1].split(' degrees')[0].split(', ')]
    fn = math.sin if name == 'sin' else math.cos
    expected = [x for x in range(360) if abs(fn(math.radians(x))-float(value)) < 1e-10]
    assert reported == expected
    assert '0 <= x < 360 degrees' in prompt


@pytest.mark.parametrize('seed', range(20))
def test_log_domain_and_intersection_answers(seed):
    prompt, answer = generators.logarithmic(rng=random.Random(seed))
    base, a, b, power = map(int, re.search(r'log_(\d+)\((-?\d+)\*x ([+-]\d+)\) = (\d+)', prompt).groups())
    x = int(re.search(r'\nx = (-?\d+)', answer).group(1))
    assert a*x+b > 0 and a*x+b == base**power
    _, _, data = generators.intersection(rng=random.Random(seed), details=True)
    a, b, c, d, x, y = data
    assert a != c and a*x+b == c*x+d == y


@pytest.mark.parametrize('factory', [kw.PDFTrigonometricEquation, kw.PDFLogarithmicEquation, kw.PDFLinearIntersection])
def test_seeded_generators_do_not_change_global_random_state(factory):
    state = random.getstate()
    first, second = factory(seed=4), factory(seed=4)
    assert first.exercise == second.exercise and first.solution == second.solution
    assert random.getstate() == state
    assert not factory(with_solution=False, seed=4).has_solution
    with pytest.raises(ValueError, match='lang'):
        factory(lang='he')


@pytest.mark.parametrize('dtype', ['trigo', 'log', 'intersection'])
def test_new_worksheet_dispatch_and_seed(dtype, monkeypatch):
    module = importlib.import_module('kiwicalc.pdf.worksheet')
    calls = []
    monkeypatch.setattr(module, 'create_pages', lambda *args: calls.append(args))
    for _ in range(2):
        kw.worksheet('unused.pdf', dtype=dtype, num_of_pages=2, equations_per_page=3, seed=7)
    assert calls[0] == calls[1] and calls[0][1] == 4
    kw.worksheet('unused.pdf', dtype=dtype, equations_per_page=1, get_solutions=False)
    assert calls[-1][1] == 1
    with pytest.raises(ValueError, match='titles'):
        kw.worksheet('unused.pdf', dtype=dtype, titles=[])


def test_flow_layout_wraps_overflows_and_preserves_literals(tmp_path, monkeypatch):
    from reportlab.pdfgen.canvas import Canvas
    original = Canvas.showPage
    pages = []
    def record(canvas):
        pages.append(canvas._pagesize)
        return original(canvas)
    monkeypatch.setattr(Canvas, 'showPage', record)
    target = tmp_path/'overflow.pdf'
    kw.create_pages(target, 1, ['Long worksheet'], [['x < 2 & y > 1 ' * 50 for _ in range(20)]], page_size='Letter')
    assert len(pages) > 1 and all(size == (612, 792) for size in pages)
    assert target.read_bytes().startswith(b'%PDF')


def test_formula_and_plot_blocks_preserve_figure_ownership(tmp_path):
    sheet = kw.PDFWorksheet('Formulas and plots')
    before = plt.get_fignums()
    sheet.add_math(r'\frac{1}{2}x^2 + \sqrt{x}')
    sheet.add_plot(lambda ax: ax.plot([0, 1, 2], [0, 1, 4]))
    sheet.add_exercise(kw.PDFExercise('Solve x=1', 'equation', 'linear', solution=1))
    sheet.end_page()
    sheet.create(tmp_path/'blocks.pdf')
    assert plt.get_fignums() == before
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    assert kw.PDFPlot(fig).image().read(8) == b'\x89PNG\r\n\x1a\n'
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_answer_sketches_and_polynomial_asymptotes(tmp_path):
    sheet = kw.PDFWorksheet('Sketches')
    for ex in (kw.PDFLinearIntersection(seed=2), kw.PDFLinearFunction(),
               kw.PDFLinearFromPoints(), kw.PDFLinearFromPointAndSlope(), kw.PDFQuadraticFunction()):
        assert isinstance(ex.solution_plot, kw.PDFPlot)
        sheet.add_exercise(ex)
    sheet.end_page()
    sheet.create(tmp_path/'sketches.pdf')
    assert 'Horizontal Asymptotes: None' in ex.solution


@pytest.mark.parametrize('options', [{'page_size': 'bad'}, {'page_size': (1,)}, {'margin': -1},
                                    {'font_size': 0}, {'margin': 400}])
def test_layout_validation(tmp_path, options):
    with pytest.raises(ValueError):
        render_pages(tmp_path/'invalid.pdf', ['title'], [['body']], **options)


def test_block_and_count_validation():
    for call in (lambda: kw.PDFMath(''), lambda: kw.PDFPlot(3), lambda: kw.PDFMath('x', font_size=0),
                 lambda: kw.worksheet(dtype='trigo', equations_per_page=-1),
                 lambda: kw.worksheet(dtype='linear', seed=2)):
        with pytest.raises((ValueError, TypeError)):
            call()
