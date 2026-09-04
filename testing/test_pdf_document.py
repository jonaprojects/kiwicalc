import pytest
import kiwicalc as kw
from kiwicalc.pdf import layout


def test_document_numbering_overflow_refresh_and_repeated_export(tmp_path, monkeypatch):
    labels = []
    original = layout.Paragraph.draw
    def record(paragraph):
        if paragraph.style.name == 'PageLabel':
            labels.append(paragraph.text)
        return original(paragraph)
    monkeypatch.setattr(layout.Paragraph, 'draw', record)
    first = kw.PDFWorksheet('First')
    first.add_exercise(kw.PDFExercise('Long question. '*150, 'test', 'test', solution='Answer'))
    first.end_page()
    second = kw.PDFWorksheet('Second', style=kw.PDFStyle(page_start=100))
    second.add_exercise(kw.PDFEquationExercise('x=1', 'linear', solution=1))
    second.end_page()
    document = kw.PDFDocument(style=kw.PDFStyle(page_size=(350, 500), margin=35))
    document.add(first).add(second).add_report(kw.Poly('x^2-1'))
    for attempt in range(2):
        labels.clear()
        document.create(tmp_path/f'combined-{attempt}.pdf')
        assert len(labels) > 5
        assert labels == [f'Page {i}' for i in range(1, len(labels)+1)]
    first.add_exercise(kw.PDFExercise('Added later', 'test', 'test', solution='New answer'))
    document.create(tmp_path/'updated.pdf')
    assert 'New answer' in str(first.pages[1].exercises)


def test_footer_component_in_standalone_and_document(tmp_path, monkeypatch):
    seen = []
    original = layout.Paragraph.draw
    def record(paragraph):
        if paragraph.style.name == 'PageLabel':
            seen.append((paragraph.text, paragraph.style.fontSize, paragraph.style.alignment))
        return original(paragraph)
    monkeypatch.setattr(layout.Paragraph, 'draw', record)
    footer = kw.PDFFooter('Practice | {page}', alignment='center', font_size=10, rule=False)
    style = kw.PDFStyle(footer=footer)
    page = kw.PDFPage('Test', [kw.PDFExercise('Question', 'test', 'test')])
    kw.PDFDocument(style=style).add(page).create(tmp_path/'page.pdf', page_start=4)
    assert seen == [('Practice | 4', 10, 1)]
    seen.clear()
    kw.create_pages(tmp_path/'standalone.pdf', 1, ['Test'], [['Text']], style=style)
    assert seen == [('Practice | 1', 10, 1)]


def test_validation(tmp_path):
    with pytest.raises(ValueError, match='no sections'):
        kw.PDFDocument().create(tmp_path/'empty.pdf')
    with pytest.raises(TypeError):
        kw.PDFDocument(style='bad')
    with pytest.raises(TypeError):
        kw.PDFDocument().add('existing.pdf')
    with pytest.raises(TypeError):
        kw.PDFDocument().add_report('x^2')
    for options in ({'text': '{bad}'}, {'text': 1}, {'alignment': 'bad'},
                    {'font_size': 0}, {'color': 1}, {'rule': 1}):
        with pytest.raises((ValueError, TypeError)):
            kw.PDFFooter(**options)
