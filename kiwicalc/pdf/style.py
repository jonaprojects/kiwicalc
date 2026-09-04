"""Validated, reusable document styling. All physical dimensions are points."""
from dataclasses import dataclass, replace
import math
from reportlab.lib.colors import toColor
from reportlab.pdfbase import pdfmetrics
from .footer import PDFFooter


@dataclass(frozen=True)
class PDFStyle:
    """Worksheet design defaults; use with_changes() for reusable variants.

    Font names may be built-in ReportLab names or fonts registered by the caller.
    Individual export keyword options take precedence over a supplied style.
    """
    page_size: object = 'A4'
    margin: float = 50
    margin_top: float | None = None
    margin_right: float | None = None
    margin_bottom: float | None = None
    margin_left: float | None = None
    font_name: str = 'Helvetica'
    heading_font: str = 'Helvetica-Bold'
    font_size: float = 12
    line_height: float = 1.5
    title_size: float = 20
    heading_size: float = 16
    subheading_size: float = 13
    caption_size: float = 10
    footer_size: float = 9
    paragraph_spacing: float = 9.6
    question_spacing: float = 12
    solution_spacing: float = 10
    heading_spacing: float = 12
    block_spacing: float = 10
    question_indent: float = 24
    solution_indent: float = 24
    text_color: str = '#202020'
    heading_color: str = '#202020'
    muted_color: str = '#555555'
    rule_color: str = '#BFC7D1'
    background_color: str = '#FFFFFF'
    primary_color: str = '#1D4E89'
    accent_color: str = '#B45309'
    surface_color: str = '#F5F7FA'
    success_color: str = '#216E39'
    warning_color: str = '#8A4600'
    alignment: str = 'left'
    title_alignment: str = 'left'
    math_alignment: str = 'left'
    plot_alignment: str = 'left'
    math_font: str = 'dejavusans'
    display_math_size: float = 16
    math_dpi: int = 200
    plot_dpi: int = 180
    plot_font: str = 'DejaVu Sans'
    plot_line_width: float = 1.5
    header: str = ''
    footer: str | PDFFooter = 'Page {page}'
    page_start: int = 1
    footer_rule: bool = True
    keep_questions_together: bool = True

    def __post_init__(self):
        if isinstance(self.page_start, bool) or not isinstance(self.page_start, int) or self.page_start < 1:
            raise ValueError('page_start must be a positive integer')
        if isinstance(self.page_size, str):
            if self.page_size.upper() not in ('A4', 'LETTER'):
                raise ValueError('page_size must be A4, Letter, or a (width, height) pair')
        else:
            if not isinstance(self.page_size, (tuple, list)) or len(self.page_size) != 2:
                raise ValueError('page_size must contain width and height in points')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in self.page_size):
                raise ValueError('Page dimensions must be positive and finite')
            object.__setattr__(self, 'page_size', tuple(self.page_size))
        positive = ('margin', 'font_size', 'line_height', 'title_size', 'heading_size',
                    'subheading_size', 'caption_size', 'footer_size', 'display_math_size',
                    'math_dpi', 'plot_dpi', 'plot_line_width')
        nonnegative = ('paragraph_spacing', 'question_spacing', 'solution_spacing',
                       'heading_spacing', 'block_spacing', 'question_indent', 'solution_indent')
        for name in positive + nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f'{name} must be {"positive" if name in positive else "nonnegative"} and finite')
        for name in ('margin_top', 'margin_right', 'margin_bottom', 'margin_left'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be nonnegative and finite')
        for name in ('font_name', 'heading_font'):
            try:
                pdfmetrics.getFont(getattr(self, name))
            except (KeyError, TypeError) as exc:
                raise ValueError(f'{name} must name a registered ReportLab font') from exc
        for name in ('text_color', 'heading_color', 'muted_color', 'rule_color', 'background_color',
                     'primary_color', 'accent_color', 'surface_color', 'success_color', 'warning_color'):
            try:
                value = getattr(self, name)
                if not isinstance(value, str):
                    raise TypeError('Colors must be names or hex strings')
                toColor(value)
            except (ValueError, TypeError, AssertionError) as exc:
                raise ValueError(f'Invalid {name}') from exc
        for name in ('alignment', 'title_alignment', 'math_alignment', 'plot_alignment'):
            if getattr(self, name) not in ('left', 'center', 'right'):
                raise ValueError(f'{name} must be left, center, or right')
        if self.math_font not in ('dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans'):
            raise ValueError('Unsupported Mathtext font set')
        if not isinstance(self.plot_font, str) or not self.plot_font.strip():
            raise ValueError('plot_font must be a nonempty Matplotlib font family')
        for name in ('math_dpi', 'plot_dpi'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        for name in ('footer_rule', 'keep_questions_together'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be a boolean')
        for name in ('header', 'footer'):
            value = getattr(self, name)
            if name == 'footer' and isinstance(value, PDFFooter):
                continue
            if not isinstance(value, str):
                raise TypeError(f'{name} must be text')
            # Fixed substitutions only: never execute arbitrary format fields.
            remainder = value.replace('{page}', '').replace('{title}', '')
            if '{' in remainder or '}' in remainder:
                raise ValueError(f'{name} supports only {{page}} and {{title}} placeholders')

    def with_changes(self, **changes):
        """Return a validated copy without modifying this style."""
        return replace(self, **changes)

    @classmethod
    def theme(cls, name, **overrides):
        """Create a resolved style from a friendly built-in theme name."""
        from .theme import get_pdf_theme
        return get_pdf_theme(name).to_style(**overrides)

    @property
    def margins(self):
        return tuple(self.margin if value is None else value for value in
                     (self.margin_top, self.margin_right, self.margin_bottom, self.margin_left))


__all__ = ['PDFStyle']
