"""ReportLab flow-based layout, with optional math and figure blocks."""
from io import BytesIO
import math
from xml.sax.saxutils import escape
from .formatting import format_math, PDFText
from .style import PDFStyle
from .footer import PDFFooter
from .blocks import PDFParagraph, PDFHeading, PDFAnswerSpace
from .arrays import PDFArray
from reportlab.platypus.paraparser import ParaParser

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, KeepTogether, Flowable
from reportlab.lib.colors import toColor
from reportlab.pdfbase.pdfmetrics import stringWidth


class PDFMath:
    """A Mathtext formula (a TeX subset), requiring no external TeX installation."""
    def __init__(self, expression, font_size=None):
        self.expression = format_math(expression)
        self.font_size = None if font_size is None else _positive(font_size, 'font_size')

    def image(self, *, font_size=16, fontset='dejavusans', color='black', dpi=200):
        from matplotlib import rc_context
        from matplotlib.mathtext import math_to_image
        from matplotlib.font_manager import FontProperties
        data = BytesIO()
        expression = self.expression.strip()
        if not expression.startswith('$'):
            expression = '$' + expression + '$'
        try:
            with rc_context({'text.usetex': False, 'mathtext.fontset': fontset}):
                math_to_image(expression, data, prop=FontProperties(size=self.font_size or font_size), dpi=dpi, color=color, format='png')
        except ValueError as exc:
            raise ValueError(f'Unsupported Mathtext expression {self.expression!r}: {exc}') from exc
        data.seek(0)
        return data


class PDFPlot:
    """A plot block from a Matplotlib Figure or a callback draw(ax).

    Callbacks render into an isolated figure; no pyplot windows are opened.
    Existing figures are read without closing them. Images are embedded at 180 dpi.
    """
    def __init__(self, source, height=180, *, caption=None):
        from matplotlib.figure import Figure
        if not callable(source) and not isinstance(source, Figure):
            raise TypeError('source must be a Matplotlib Figure or draw(ax) callback')
        self.source, self.height = source, _positive(height, 'height')
        if caption is not None and not isinstance(caption, str):
            raise TypeError('caption must be text')
        self.caption = caption

    def image(self, *, style=None):
        from matplotlib.figure import Figure
        from matplotlib import rc_context
        from cycler import cycler
        data = BytesIO()
        style = style or PDFStyle()
        with rc_context({'text.usetex': False, 'font.family': style.plot_font,
                         'font.size': style.font_size, 'lines.linewidth': style.plot_line_width,
                         'figure.facecolor': style.background_color, 'axes.facecolor': style.surface_color,
                         'text.color': style.text_color, 'axes.labelcolor': style.text_color,
                         'axes.edgecolor': style.muted_color, 'grid.color': style.rule_color,
                         'xtick.color': style.text_color, 'ytick.color': style.text_color,
                         'axes.prop_cycle': cycler(color=(style.primary_color, style.accent_color,
                                                          style.success_color, style.warning_color))}):
            if isinstance(self.source, Figure):
                figure = self.source
            else:
                figure = Figure(figsize=(6, self.height/72), constrained_layout=True)
                self.source(figure.subplots())
            figure.savefig(data, format='png', dpi=style.plot_dpi, bbox_inches='tight')
        data.seek(0)
        return data


class _MathParagraphParser(ParaParser):
    """Resolve only our generated image tokens, entirely in memory."""
    def __init__(self, images):
        super().__init__()
        self.images = images

    def end_img(self):
        self._stack[-1].src = BytesIO(self.images[self._stack[-1].src])
        super().end_img()


class _WritingArea(Flowable):
    def __init__(self, block, width, color):
        super().__init__()
        self.block, self.width, self.height, self.color = block, width, block.height, color

    def draw(self):
        self.canv.setStrokeColor(toColor(self.color))
        self.canv.setLineWidth(.4)
        if self.block.pattern == 'blank':
            return
        y = 0
        while y <= self.height:
            self.canv.line(0, y, self.width, y)
            y += self.block.spacing
        if self.block.pattern == 'grid':
            x = 0
            while x <= self.width:
                self.canv.line(x, 0, x, self.height)
                x += self.block.spacing


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def render_pages(path, titles, pages, *, style=None, theme=None, **overrides):
    """Render logical pages with line height as a font-size multiplier.

    Tall inline formulas may expand a line to avoid clipping. Paragraph spacing
    is separate from line height and is not changed by this option.
    """
    if style is not None and not isinstance(style, PDFStyle):
        raise TypeError('style must be a PDFStyle')
    if style is not None and theme is not None:
        raise ValueError('Pass either style or theme, not both')
    if theme is not None:
        from .theme import get_pdf_theme
        design = get_pdf_theme(theme).to_style(**overrides)
    else:
        design = (style or PDFStyle()).with_changes(**overrides)
    page_size = design.page_size
    font_size, line_height = design.font_size, design.line_height
    if isinstance(page_size, str):
        try:
            size = {'A4': A4, 'LETTER': letter}[page_size.upper()]
        except KeyError as exc:
            raise ValueError('page_size must be A4, Letter, or a (width, height) pair') from exc
    else:
        if len(page_size) != 2:
            raise ValueError('page_size must contain width and height in points')
        size = tuple(_positive(v, 'page dimension') for v in page_size)
    top, right, bottom, left = design.margins
    if size[0]-left-right < 80 or size[1]-top-bottom < 80:
        raise ValueError('Margins leave too little usable page space')
    doc = SimpleDocTemplate(str(path), pagesize=size, leftMargin=left, rightMargin=right,
                            topMargin=top, bottomMargin=bottom)
    alignment = {'left': 0, 'center': 1, 'right': 2}
    body = ParagraphStyle('WorksheetBody', fontName=design.font_name, fontSize=font_size,
                           leading=font_size*line_height, spaceAfter=design.paragraph_spacing,
                           textColor=toColor(design.text_color), alignment=alignment[design.alignment],
                           splitLongWords=True, autoLeading='max', allowWidows=0, allowOrphans=0)
    heading = ParagraphStyle('WorksheetHeading', parent=body, fontName=design.heading_font,
                              fontSize=design.title_size, leading=design.title_size*line_height,
                              textColor=toColor(design.heading_color),
                              alignment=alignment[design.title_alignment],
                              # Inline math can rise above the following text
                              # baseline; reserve half a body line of clearance.
                              spaceAfter=design.heading_spacing+font_size*.5,
                              keepWithNext=True)
    story = []
    for index, (title, lines) in enumerate(zip(titles, pages)):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(escape(str(title)), heading))
        if not lines:
            story.append(Spacer(1, 1))
        previous_was_heading = True
        for line in lines:
            if isinstance(line, PDFAnswerSpace):
                if line.height > doc.height-12:
                    raise ValueError('Answer space is taller than the usable page')
                story.extend([_WritingArea(line, doc.width-12, design.rule_color), Spacer(1, design.block_spacing)])
            elif isinstance(line, PDFHeading):
                heading_size = design.heading_size if line.level == 1 else design.subheading_size
                section = ParagraphStyle('Section', parent=heading, fontSize=heading_size,
                                         leading=heading_size*line_height, spaceBefore=design.heading_spacing)
                story.append(Paragraph(escape(line.text), section))
            elif isinstance(line, PDFText):
                markup = []
                images = {}
                paragraph_style = body
                bullet = None
                if isinstance(line, PDFParagraph):
                    indent = design.solution_indent if line.role == 'solution' else design.question_indent
                    if line.number is not None:
                        bullet = f'{line.number}.'
                        indent = max(indent, stringWidth(bullet, design.font_name, font_size) + 8)
                    else:
                        indent = 0
                    paragraph_style = ParagraphStyle('Numbered', parent=body, leftIndent=indent,
                        bulletIndent=0, bulletFontName=design.font_name, bulletFontSize=font_size,
                        spaceAfter=design.solution_spacing if line.role == 'solution' else design.question_spacing)
                for part in line.parts:
                    if isinstance(part, str):
                        markup.append(escape(part.strip() if len(line.parts) == 1 else part).replace('\n', '<br/>'))
                    else:
                        data = part.image(font_size=font_size, fontset=design.math_font,
                                          color=design.text_color, dpi=design.math_dpi).getvalue()
                        picture = Image(BytesIO(data))
                        width, height = picture.imageWidth*72/design.math_dpi, picture.imageHeight*72/design.math_dpi
                        if width > doc.width-12-paragraph_style.leftIndent:
                            raise ValueError('Inline formula is wider than the page; use add_math() for a display formula')
                        uri = str(len(images))
                        images[uri] = data
                        markup.append(f'<img src="{uri}" width="{width}" height="{height}" valign="middle"/>')
                text = ''.join(markup)
                parsed_style, fragments, _ = _MathParagraphParser(images).parse(text, paragraph_style)
                paragraph = Paragraph(text, parsed_style, frags=fragments, bulletText=bullet)
                # An oversized paragraph must split in place. Wrapping it in
                # KeepTogether would first push it to the next page and strand
                # preceding headings on a mostly empty page.
                paragraph_height = paragraph.wrap(doc.width-12, doc.height)[1]
                # A page heading already keeps its next flowable with it. Let it
                # place a first solution directly: nesting an image-heavy answer
                # in another KeepTogether can pull its bullet into the title.
                if (isinstance(line, PDFParagraph) and design.keep_questions_together
                        and (not previous_was_heading or line.role != 'solution')
                        and paragraph_height <= doc.height-12):
                    story.append(KeepTogether([paragraph]))
                else:
                    story.append(paragraph)
            elif isinstance(line, (PDFMath, PDFArray, PDFPlot)):
                data = line.image(font_size=design.display_math_size, fontset=design.math_font,
                                  color=design.text_color, dpi=design.math_dpi) if isinstance(line, (PDFMath, PDFArray)) else line.image(style=design)
                image = Image(data)
                # PNG pixels -> PDF points, then fit inside the physical frame.
                dpi = design.math_dpi if isinstance(line, (PDFMath, PDFArray)) else design.plot_dpi
                width, height = image.imageWidth*72/dpi, image.imageHeight*72/dpi
                scale = min(1., (doc.width-12)/width, (doc.height-60)/height)
                image.drawWidth, image.drawHeight = width*scale, height*scale
                image.hAlign = (design.math_alignment if isinstance(line, (PDFMath, PDFArray)) else design.plot_alignment).upper()
                if isinstance(line, PDFPlot) and line.caption:
                    caption_style = ParagraphStyle('Caption', parent=body, fontSize=design.caption_size,
                        leading=design.caption_size*line_height, textColor=toColor(design.muted_color),
                        alignment=alignment[design.plot_alignment])
                    caption = Paragraph(escape(line.caption), caption_style)
                    story.append(KeepTogether([image, Spacer(1, 4), caption]))
                else:
                    story.append(image)
                story.append(Spacer(1, design.block_spacing))
            elif str(line).strip():
                story.append(Paragraph(escape(str(line)).replace('\n', '<br/>'), body))
            else:
                story.append(Spacer(1, design.paragraph_spacing))
            previous_was_heading = isinstance(line, PDFHeading)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(toColor(design.background_color))
        canvas.rect(0, 0, size[0], size[1], fill=1, stroke=0)
        canvas.setFont(design.font_name, design.footer_size)
        canvas.setFillColor(toColor(design.muted_color))
        canvas.setStrokeColor(toColor(design.rule_color))
        component = design.footer if isinstance(design.footer, PDFFooter) else PDFFooter(design.footer)
        component.draw(canvas, style=design, page=design.page_start + document.page - 1,
                       title=titles[0] if titles else '', left=left, bottom=bottom, width=doc.width)
        def label(template):
            return template.replace('{page}', str(design.page_start + document.page - 1)).replace('{title}', str(titles[0]) if titles else '')
        def draw_label(template, available, y, align):
            if not template:
                return
            label_style = ParagraphStyle('PageLabel', fontName=design.font_name,
                fontSize=design.footer_size, leading=design.footer_size*1.25,
                textColor=toColor(design.muted_color), alignment=align)
            paragraph = Paragraph(escape(label(template)).replace('\n', '<br/>'), label_style)
            width, height = paragraph.wrap(doc.width, available)
            if height > available:
                raise ValueError('Header/footer does not fit its margin; increase the margin or shorten the text')
            paragraph.drawOn(canvas, left, y-height/2)
        draw_label(design.header, top*.8, size[1]-top/2, 0)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


__all__ = ['PDFMath', 'PDFPlot']
