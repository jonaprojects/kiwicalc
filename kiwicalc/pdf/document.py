"""Compose KiwiCalc sections before rendering, with one page-number sequence."""
from .style import PDFStyle
from .theme import get_pdf_theme
from .worksheet import PDFWorksheet, PDFPage, create_pages


class PDFDocument:
    """One styled document containing worksheet, page, and report sections.

    Sections start on new pages. The document's style governs all sections;
    section worksheet styles are intentionally not applied. Content is prepared
    at export time so edits and refreshed answer keys are reflected.
    """
    def __init__(self, *, style=None, theme=None):
        if style is not None and not isinstance(style, PDFStyle):
            raise TypeError('style must be a PDFStyle')
        if style is not None and theme is not None:
            raise ValueError('Pass either style or theme, not both')
        self.style = get_pdf_theme(theme).to_style() if theme is not None else style
        self._sections = []

    def add(self, section):
        """Append a PDFWorksheet or PDFPage and return this document."""
        if not isinstance(section, (PDFWorksheet, PDFPage)):
            raise TypeError('section must be a PDFWorksheet or PDFPage, not an exported PDF')
        self._sections.append(section)
        return self

    def add_report(self, polynomial):
        """Append a polynomial report, including its graph."""
        from kiwicalc.expressions.poly import Poly
        if not isinstance(polynomial, Poly):
            raise TypeError('polynomial must be a Poly')
        self._sections.append(polynomial)
        return self

    def create(self, path, **layout_options):
        """Render once; footer numbering includes every physical output page."""
        if not self._sections:
            raise ValueError('Cannot create a document with no sections')
        titles, pages = [], []
        for section in self._sections:
            if isinstance(section, PDFWorksheet):
                section_titles, section_pages = section._render_content()
            elif isinstance(section, PDFPage):
                worksheet = PDFWorksheet(section.title)
                worksheet.pages[0].exercises.extend(section.exercises)
                section_titles, section_pages = worksheet._render_content()
            else:
                section_titles, section_pages = ['Function Report'], [section._pdf_report_content()]
            titles.extend(section_titles)
            pages.extend(section_pages)
        if 'style' not in layout_options and 'theme' not in layout_options and self.style is not None:
            layout_options['style'] = self.style
        create_pages(path, len(titles), titles, pages, **layout_options)


__all__ = ['PDFDocument']
