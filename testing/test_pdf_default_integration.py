import importlib
import pytest
import matplotlib.pyplot as plt
import kiwicalc as kw
from kiwicalc.pdf import layout
from kiwicalc.pdf.blocks import PDFParagraph


@pytest.mark.parametrize('factory', [kw.LinearEquation, kw.QuadraticEquation,
    kw.CubicEquation, kw.QuarticEquation, kw.PolyEquation])
@pytest.mark.parametrize('answers', [False, True])
def test_single_helpers_use_semantic_styled_questions(factory, answers, tmp_path, monkeypatch):
    calls = []
    original = layout.render_pages
    def record(path, titles, pages, **options):
        calls.append((titles, pages, options))
        return original(path, titles, pages, **options)
    monkeypatch.setattr(layout, 'render_pages', record)
    target = tmp_path/'legacy.pdf'
    result = factory.random_worksheet(target, num_of_equations=2, get_solutions=answers)
    assert result is not False
    titles, pages, options = calls[0]
    assert len(pages) == (2 if answers else 1)
    assert all(isinstance(question, PDFParagraph) for question in pages[0])
    assert [question.number for question in pages[0]] == [1, 2]
    assert all(any(isinstance(part, kw.PDFMath) for part in question.parts) for question in pages[0])
    assert options == {}  # no opt-in style required


def test_quartic_single_generates_fourth_degree(monkeypatch):
    calls = []
    monkeypatch.setattr(kw.PolyEquation, 'random_worksheet', lambda **kwargs: calls.append(kwargs))
    kw.QuarticEquation.random_worksheet('unused.pdf')
    assert calls[0]['degrees_range'] == (4, 4)
    assert 'Quartic' in calls[0]['title']


@pytest.mark.parametrize('expression', ['x^2-4', 'x^2+y^2', '3'])
def test_report_uses_styled_renderer_without_temp_images(expression, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = plt.get_fignums()
    poly = kw.Poly(expression)
    poly.export_report(tmp_path/'report.pdf', font_size=14)
    assert (tmp_path/'report.pdf').read_bytes().startswith(b'%PDF')
    assert not list(tmp_path.glob('*.png'))
    assert plt.get_fignums() == before


def test_adjusted_supports_path_and_formats_equations(tmp_path):
    target = tmp_path/'manual.pdf'
    assert kw.LinearEquation.adjusted_worksheet(equations=['x^2=4', '2x+1=5'], path=target)
    assert target.exists()


def test_replacing_two_fields_preserves_both_formula_parts():
    from kiwicalc.pdf.formatting import _replace_math
    text = _replace_math('A then B', 'A', 'x^2')
    text = _replace_math(text, 'B', 'y^2')
    assert str(text) == 'A then B'
    assert [part.expression for part in text.parts if isinstance(part, kw.PDFMath)] == ['x^2', 'y^2']


def test_duplicate_random_factors_include_introduced_zero_root(monkeypatch):
    single = importlib.import_module('kiwicalc.equations.single')
    numbers = iter([4, -3, -3])
    monkeypatch.setattr(single.random, 'uniform', lambda *args: next(numbers))
    expression, roots = single.random_polynomial(degree=2, get_solutions=True)
    assert set(roots) == {0, 3}
    fn = kw.Poly(expression).to_lambda()
    assert all(fn(root) == 0 for root in roots)
