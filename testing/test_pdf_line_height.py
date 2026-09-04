import pytest
from kiwicalc.pdf import layout
import kiwicalc as kw


@pytest.mark.parametrize('page_size', ['A4', 'Letter', (600, 800)])
@pytest.mark.parametrize('height', [None, 1.25, 2])
def test_line_height_applies_to_plain_rich_and_heading(tmp_path, monkeypatch, page_size, height):
    stories = []
    monkeypatch.setattr(layout.SimpleDocTemplate, 'build', lambda self, story, **kwargs: stories.append(story))
    options = {} if height is None else {'line_height': height}
    kw.create_pages(tmp_path/'spacing.pdf', 1, ['Heading'],
                    [['Plain text', kw.PDFText('Rich text')]], page_size=page_size, **options)
    paragraphs = [item for item in stories[0] if isinstance(item, layout.Paragraph)]
    assert len(paragraphs) == 3
    for paragraph in paragraphs:
        assert paragraph.style.leading == paragraph.style.fontSize * (1.5 if height is None else height)
        assert paragraph.style.autoLeading == 'max'


@pytest.mark.parametrize('value', [0, -1, True, '1.5', float('nan'), float('inf')])
def test_invalid_line_height(tmp_path, value):
    target = tmp_path/'invalid.pdf'
    with pytest.raises(ValueError, match='line_height'):
        kw.create_pages(target, 1, ['Title'], [['Text']], line_height=value)
    assert not target.exists()


def test_public_exports_forward_line_height(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append(kwargs))
    sheet = kw.PDFWorksheet()
    sheet.create(tmp_path/'sheet.pdf', line_height=1.25)
    assert kw.create_pdf(tmp_path/'plain.pdf', lines=['Question'], line_height=2)
    assert [call['line_height'] for call in calls] == [1.25, 2]
