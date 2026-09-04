"""Reusable page footer, shared by standalone and composed documents."""
from dataclasses import dataclass
import math
from xml.sax.saxutils import escape
from reportlab.lib.colors import toColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


@dataclass(frozen=True)
class PDFFooter:
    """Footer configuration. None values inherit the document style."""
    text: str = 'Page {page}'
    alignment: str = 'right'
    font_size: float | None = None
    color: str | None = None
    rule: bool | None = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError('Footer text must be a string')
        remainder = self.text.replace('{page}', '').replace('{title}', '')
        if '{' in remainder or '}' in remainder:
            raise ValueError('Footer supports only {page} and {title} placeholders')
        if self.alignment not in ('left', 'center', 'right'):
            raise ValueError('Footer alignment must be left, center, or right')
        if self.font_size is not None and (isinstance(self.font_size, bool) or not isinstance(self.font_size, (int, float)) or not math.isfinite(self.font_size) or self.font_size <= 0):
            raise ValueError('Footer font_size must be positive and finite')
        if self.color is not None:
            if not isinstance(self.color, str):
                raise TypeError('Footer color must be a name or hex string')
            toColor(self.color)
        if self.rule is not None and not isinstance(self.rule, bool):
            raise TypeError('Footer rule must be a boolean')

    def draw(self, canvas, *, style, page, title, left, bottom, width):
        if style.footer_rule if self.rule is None else self.rule:
            canvas.setStrokeColor(toColor(style.rule_color))
            canvas.line(left, bottom*.7, left+width, bottom*.7)
        if not self.text:
            return
        size = self.font_size or style.footer_size
        paragraph_style = ParagraphStyle('PageLabel', fontName=style.font_name,
            fontSize=size, leading=size*1.25,
            textColor=toColor(self.color or style.muted_color),
            alignment={'left': 0, 'center': 1, 'right': 2}[self.alignment])
        text = self.text.replace('{page}', str(page)).replace('{title}', str(title))
        paragraph = Paragraph(escape(text).replace('\n', '<br/>'), paragraph_style)
        _, height = paragraph.wrap(width, bottom*.6)
        if height > bottom*.6:
            raise ValueError('Header/footer does not fit its margin; increase the margin or shorten the text')
        paragraph.drawOn(canvas, left, bottom*.3-height/2)


__all__ = ['PDFFooter']
