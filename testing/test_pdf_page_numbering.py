import re
import pytest
import kiwicalc as kw
from kiwicalc.pdf import layout


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '1'])
def test_page_start_validation(value):
    with pytest.raises(ValueError, match='page_start'):
        kw.PDFStyle(page_start=value)


def test_page_numbers_follow_physical_overflow_and_answer_pages(tmp_path, monkeypatch):
    labels = []
    original = layout.Paragraph.draw
    def record(paragraph):
        if paragraph.style.name == 'PageLabel':
            labels.append(paragraph.text)
        return original(paragraph)
    monkeypatch.setattr(layout.Paragraph, 'draw', record)
    sheet = kw.PDFWorksheet('Title')
    sheet.add_exercise(kw.PDFExercise('Long question. '*180, 'test', 'test', solution='Answer'))
    sheet.end_page()
    sheet.create(tmp_path/'overflow.pdf', page_size=(350, 500), margin=35, page_start=7)
    assert len(labels) > sheet.num_of_pages
    assert labels == [f'Page {page}' for page in range(7, 7+len(labels))]
    labels.clear()
    kw.create_pages(tmp_path/'fresh.pdf', 1, ['New document'], [['Text']])
    assert labels == ['Page 1']  # no leaking numbering state between exports


def test_merged_sections_continue_numbering(tmp_path):
    pypdf = pytest.importorskip('pypdf')
    writer = pypdf.PdfWriter()
    for index in range(3):
        target = tmp_path/f'section-{index}.pdf'
        kw.create_pages(target, 2, ['Questions', 'Solutions'], [['Question'], ['Answer']],
                        page_start=len(writer.pages)+1)
        writer.append(target)
    result = tmp_path/'combined.pdf'
    writer.write(result)
    reader = pypdf.PdfReader(result)
    assert [re.findall(r'Page (\d+)', page.extract_text()) for page in reader.pages] == [[str(i)] for i in range(1, 7)]
