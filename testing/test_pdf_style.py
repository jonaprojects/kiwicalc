import pytest
import kiwicalc as kw
from kiwicalc.pdf import layout
from kiwicalc.pdf.blocks import PDFParagraph


def test_style_validation_and_immutable_variants():
    original = kw.PDFStyle()
    variant = original.with_changes(font_size=14, margin_left=70)
    assert original.font_size == 12 and variant.font_size == 14
    assert variant.margins == (50, 50, 50, 70)
    for options in ({'line_height': 0}, {'font_name': 'missing'}, {'alignment': 'bad'},
                    {'text_color': 'notacolor'}, {'header': '{arbitrary}'}, {'margin_top': -1},
                    {'keep_questions_together': 1}):
        with pytest.raises((ValueError, TypeError)):
            kw.PDFStyle(**options)


def test_plain_and_rich_newlines_share_semantics():
    sheet = kw.PDFWorksheet()
    plain = sheet._text_lines('one\ntwo', 12)[0]
    rich = sheet._text_lines(kw.PDFText('one\ntwo'), 12)[0]
    assert isinstance(plain, PDFParagraph) and isinstance(rich, PDFParagraph)
    assert plain.parts == rich.parts and plain.number == rich.number == 12


def test_style_render_inheritance_and_overrides(tmp_path, monkeypatch):
    seen = []
    original = kw.PDFMath.image
    def record(self, **kwargs):
        seen.append((self.font_size, kwargs))
        return original(self, **kwargs)
    monkeypatch.setattr(kw.PDFMath, 'image', record)
    style = kw.PDFStyle(font_size=14, header='Algebra | {title}', footer='Sheet {page}',
                        margin_left=65, heading_color='#223355')
    sheet = kw.PDFWorksheet('Test', style=style)
    sheet.add_heading('Section')
    sheet.add_exercise(kw.PDFTrigonometricEquation(seed=7))
    sheet.add_math('x^2')
    sheet.add_math('x=1', font_size=22)
    sheet.end_page()
    sheet.create(tmp_path/'style.pdf', font_size=15)
    assert (tmp_path/'style.pdf').read_bytes().startswith(b'%PDF')
    assert seen[0][0] is None and seen[0][1]['font_size'] == 15
    assert any(size == 22 for size, _ in seen)
    assert style.font_size == 14


def test_hanging_indent_does_not_include_number_in_content(tmp_path, monkeypatch):
    stories = []
    monkeypatch.setattr(layout.SimpleDocTemplate, 'build', lambda self, story, **kwargs: stories.append(story))
    layout.render_pages(tmp_path/'unused.pdf', ['Title'], [[PDFParagraph('Long\nquestion', number=123)]],
                        style=kw.PDFStyle(keep_questions_together=False))
    paragraph = stories[0][-1]
    assert paragraph.bulletText == '123.'
    assert paragraph.style.leftIndent >= 24
    assert '123' not in paragraph.text


@pytest.mark.parametrize('dtype', ['linear', 'quadratic', 'cubic', 'quartic', 'polynomial', 'trigo', 'log', 'intersection'])
@pytest.mark.parametrize('answers', [False, True])
def test_every_batch_family_forwards_style(dtype, answers, tmp_path, monkeypatch):
    calls = []
    original = layout.render_pages
    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(layout, 'render_pages', record)
    design = kw.PDFStyle(font_size=13, paragraph_spacing=8)
    kw.worksheet(tmp_path/f'{dtype}.pdf', dtype=dtype, equations_per_page=2,
                  get_solutions=answers, style=design, line_height=1.25)
    assert calls[0]['style'] is design and calls[0]['line_height'] == 1.25


def test_sections_captions_spaces_and_long_question(tmp_path):
    sheet = kw.PDFWorksheet('Narrow pages', style=kw.PDFStyle(page_size=(350, 500), margin=35))
    sheet.add_heading('Section one').add_heading('Subsection', level=2)
    sheet.add_exercise(kw.PDFExercise('Long question wraps. '*100, 'test', 'test', solution='Answer'))
    sheet.add_answer_space(height=45, pattern='grid')
    sheet.add_answer_space(height=20, pattern='blank')
    sheet.add_plot(lambda ax: ax.plot([0, 1], [0, 1]), height=100,
                   caption='A caption kept with the plot.')
    sheet.end_page()
    sheet.create(tmp_path/'long.pdf')
    assert len(sheet.pages[1].exercises) == 1  # no headings or writing areas in answer key
    with pytest.raises(ValueError):
        kw.PDFAnswerSpace(pattern='bad')
    with pytest.raises(ValueError):
        kw.PDFHeading('title', level=3)


def test_plot_style_and_ownership():
    import matplotlib.pyplot as plt
    before = dict(plt.rcParams)
    seen = []
    design = kw.PDFStyle(font_size=17, plot_line_width=3)
    block = kw.PDFPlot(lambda ax: seen.append(ax.plot([0, 1], [1, 0])[0].get_linewidth()))
    block.image(style=design)
    assert seen == [3]
    assert dict(plt.rcParams) == before
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1], linewidth=1)
    try:
        kw.PDFPlot(fig).image(style=design)
        assert line.get_linewidth() == 1 and plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('options', [
    {'page_size': 'bad'}, {'page_size': (1,)}, {'page_size': (-10, 500)},
    {'margin': float('inf')}, {'question_spacing': -1}, {'margin_left': True},
    {'title_alignment': 'justify'}, {'math_font': 'unknown'}, {'plot_font': ''},
    {'math_dpi': 12.5}, {'plot_dpi': False}, {'footer': 12}, {'text_color': 2},
])
def test_additional_invalid_styles(options):
    with pytest.raises((ValueError, TypeError)):
        kw.PDFStyle(**options)


def test_export_style_replacement_and_custom_fonts(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(layout.SimpleDocTemplate, 'build', lambda self, story, **kwargs: seen.append(story))
    sheet = kw.PDFWorksheet('Title', style=kw.PDFStyle(font_size=14))
    sheet.add_exercise(kw.PDFExercise('Text', 'test', 'test'))
    sheet.create(tmp_path/'style.pdf', style=kw.PDFStyle(font_name='Times-Roman',
                 heading_font='Times-Bold', font_size=10, keep_questions_together=False), font_size=11)
    assert seen[0][-1].style.fontName == 'Times-Roman'
    assert seen[0][-1].style.fontSize == 11
    assert sheet.style.font_size == 14


def test_oversized_question_does_not_strand_heading(tmp_path, monkeypatch):
    drawn = []
    original = layout.Paragraph.draw
    def record(paragraph):
        drawn.append((paragraph.canv.getPageNumber(), paragraph.text))
        return original(paragraph)
    monkeypatch.setattr(layout.Paragraph, 'draw', record)
    layout.render_pages(tmp_path/'overflow.pdf', ['Title'],
        [[kw.PDFHeading('Section'), PDFParagraph('Long question. '*200, number=1)]],
        style=kw.PDFStyle(page_size=(350, 500), margin=35))
    # Split fragments can have text=None, so ensure the first page draws more
    # than its title/section/footer and does not consist of headings alone.
    first = [text for page, text in drawn if page == 1]
    assert None in first or any(text and 'Long question' in text for text in first)


def test_header_footer_margin_validation_and_disabled_labels(tmp_path):
    with pytest.raises(ValueError, match='Header/footer'):
        layout.render_pages(tmp_path/'crowded.pdf', ['Title'], [['Text']],
                            style=kw.PDFStyle(margin_top=1, header='Header'))
    layout.render_pages(tmp_path/'minimal.pdf', ['Title'], [['Text']],
                        style=kw.PDFStyle(header='', footer='', footer_rule=False,
                                          background_color='#F8F8F8', alignment='center'))
    with pytest.raises(ValueError, match='Answer space'):
        layout.render_pages(tmp_path/'space.pdf', ['Title'], [[kw.PDFAnswerSpace(1000)]])
