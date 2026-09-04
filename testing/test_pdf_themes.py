import pytest
import kiwicalc as kw
from kiwicalc.pdf import layout


EXPECTED = ('academic', 'classroom', 'assessment', 'engineering', 'accessible', 'ink_saver')


def test_built_in_themes_are_available_valid_and_immutable():
    assert kw.available_pdf_themes() == EXPECTED
    for name in EXPECTED:
        theme = kw.get_pdf_theme(name)
        assert theme is kw.PDFTheme.get(name)
        assert isinstance(theme.to_style(), kw.PDFStyle)
    classroom = kw.PDFTheme.get('classroom')
    custom = classroom.with_options(primary='#123456', heading='#234567', body_size=14,
                                    margin=60, heading_spacing=18, name='school')
    assert classroom.colors.primary != custom.colors.primary
    assert custom.name == 'school'
    assert custom.to_style().font_size == 14
    assert custom.to_style().margin == 60
    assert custom.to_style().heading_color == '#234567'
    assert custom.to_style().heading_spacing == 18


def test_friendly_theme_lookup_and_style_shortcut():
    assert kw.get_pdf_theme('Ink Saver').name == 'ink_saver'
    assert kw.get_pdf_theme('ink-saver').name == 'ink_saver'
    style = kw.PDFStyle.theme('academic', page_size='Letter', footer='Sheet {page}')
    assert style.font_name == 'Times-Roman'
    assert style.page_size == 'Letter'
    assert style.footer == 'Sheet {page}'
    with pytest.raises(ValueError, match='Unknown PDF theme'):
        kw.get_pdf_theme('neon')
    with pytest.raises(TypeError, match='theme'):
        kw.get_pdf_theme(3)
    with pytest.raises(TypeError, match='Unknown theme option'):
        kw.PDFTheme.get('academic').with_options(glitter=True)


def test_theme_accessibility_validation():
    with pytest.raises(ValueError, match='muted contrast'):
        kw.PDFTheme('low contrast', colors=kw.PDFThemeColors(muted='#BBBBBB'))
    with pytest.raises(ValueError, match='body_size'):
        kw.PDFTheme.get('classroom').with_options(body_size=0)
    with pytest.raises((TypeError, ValueError), match='colors'):
        kw.PDFTheme('bad', colors='blue')


def test_theme_supported_across_friendly_entry_points(tmp_path, monkeypatch):
    seen = []
    original = layout.render_pages
    def record(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(layout, 'render_pages', record)

    sheet = kw.PDFWorksheet('Friendly', theme='classroom')
    sheet.add_exercise(kw.PDFExercise('Solve x + 2 = 5.', 'test', 'test', solution=3))
    sheet.end_page()
    assert sheet.style.font_size == 12.5
    sheet.create(tmp_path/'sheet.pdf')
    sheet.create(tmp_path/'override.pdf', theme='engineering')
    kw.create_pages(tmp_path/'pages.pdf', 1, ['Title'], [['Text']], theme='accessible')
    kw.PDFDocument(theme='assessment').add(sheet).create(tmp_path/'document.pdf')

    assert seen[0]['style'].font_size == 12.5
    assert seen[1]['theme'] == 'engineering' and 'style' not in seen[1]
    assert seen[2]['theme'] == 'accessible'
    assert seen[3]['style'].font_size == 11.5
    for name in ('sheet.pdf', 'override.pdf', 'pages.pdf', 'document.pdf'):
        assert (tmp_path/name).read_bytes().startswith(b'%PDF')


def test_style_and_theme_are_unambiguous(tmp_path):
    with pytest.raises(ValueError, match='either style or theme'):
        kw.PDFWorksheet(style=kw.PDFStyle(), theme='academic')
    with pytest.raises(ValueError, match='either style or theme'):
        kw.PDFDocument(style=kw.PDFStyle(), theme='academic')
    with pytest.raises(ValueError, match='either style or theme'):
        layout.render_pages(tmp_path/'bad.pdf', ['Title'], [['Text']],
                            style=kw.PDFStyle(), theme='academic')


def test_theme_semantic_colors_reach_plots():
    seen = {}
    style = kw.PDFStyle.theme('classroom')
    def draw(ax):
        seen['line'] = ax.plot([0, 1], [0, 1])[0].get_color()
        seen['surface'] = ax.get_facecolor()
    kw.PDFPlot(draw).image(style=style)
    from reportlab.lib.colors import toColor
    assert seen['line'] == style.primary_color
    expected = toColor(style.surface_color)
    assert seen['surface'][:3] == pytest.approx((expected.red, expected.green, expected.blue))
