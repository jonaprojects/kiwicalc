"""Friendly, semantic themes for KiwiCalc PDF documents."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from reportlab.lib.colors import toColor

from .style import PDFStyle


def _color(value, name):
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a color name or hex string')
    try:
        return toColor(value)
    except (ValueError, TypeError, AssertionError) as exc:
        raise ValueError(f'Invalid {name}') from exc


def _contrast(first, second):
    def luminance(color):
        channels = []
        for value in (color.red, color.green, color.blue):
            channels.append(value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4)
        return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2]
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + .05) / (dark + .05)


@dataclass(frozen=True)
class PDFThemeColors:
    """Colors named by their role rather than by a particular component."""
    text: str = '#202020'
    heading: str = '#202020'
    muted: str = '#555555'
    primary: str = '#1D4E89'
    accent: str = '#B45309'
    background: str = '#FFFFFF'
    surface: str = '#F5F7FA'
    border: str = '#BFC7D1'
    success: str = '#216E39'
    warning: str = '#8A4600'

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _color(getattr(self, name), name)


@dataclass(frozen=True)
class PDFThemeTypography:
    """A coordinated type scale for a document theme."""
    body_font: str = 'Helvetica'
    heading_font: str = 'Helvetica-Bold'
    plot_font: str = 'DejaVu Sans'
    math_font: str = 'dejavusans'
    body_size: float = 12
    title_size: float = 20
    heading_size: float = 16
    subheading_size: float = 13
    caption_size: float = 10
    footer_size: float = 9
    line_height: float = 1.5

    def __post_init__(self):
        for name in ('body_size', 'title_size', 'heading_size', 'subheading_size',
                     'caption_size', 'footer_size', 'line_height'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class PDFThemeSpacing:
    """A compact spacing scale applied consistently to PDF components."""
    margin: float = 50
    paragraph_spacing: float = 9.6
    question_spacing: float = 12
    solution_spacing: float = 10
    heading_spacing: float = 12
    block_spacing: float = 10
    question_indent: float = 24
    solution_indent: float = 24

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if self.margin == 0:
            raise ValueError('margin must be positive and finite')


@dataclass(frozen=True)
class PDFTheme:
    """High-level visual intent that resolves to a deterministic PDFStyle."""
    name: str
    colors: PDFThemeColors = PDFThemeColors()
    typography: PDFThemeTypography = PDFThemeTypography()
    spacing: PDFThemeSpacing = PDFThemeSpacing()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Theme name must be nonempty text')
        if not isinstance(self.colors, PDFThemeColors):
            raise TypeError('colors must be PDFThemeColors')
        if not isinstance(self.typography, PDFThemeTypography):
            raise TypeError('typography must be PDFThemeTypography')
        if not isinstance(self.spacing, PDFThemeSpacing):
            raise TypeError('spacing must be PDFThemeSpacing')
        background = _color(self.colors.background, 'background')
        for role in ('text', 'heading', 'muted'):
            ratio = _contrast(_color(getattr(self.colors, role), role), background)
            if ratio < 4.5:
                raise ValueError(f'{role} contrast is {ratio:.2f}:1; PDF themes require at least 4.5:1')
        # PDFStyle performs numeric, font, and Mathtext validation in one place.
        self.to_style()

    @classmethod
    def get(cls, name):
        """Return a built-in theme by name."""
        return get_pdf_theme(name)

    def with_options(self, **options):
        """Return a customized theme using friendly, flat option names."""
        groups = {
            **{name: 'colors' for name in PDFThemeColors.__dataclass_fields__},
            **{name: 'typography' for name in PDFThemeTypography.__dataclass_fields__},
            **{name: 'spacing' for name in PDFThemeSpacing.__dataclass_fields__},
        }
        values = {'colors': {}, 'typography': {}, 'spacing': {}}
        new_name = options.pop('name', self.name)
        unknown = sorted(set(options) - set(groups))
        if unknown:
            raise TypeError(f'Unknown theme option: {unknown[0]}')
        for key, value in options.items():
            values[groups[key]][key] = value
        return PDFTheme(new_name,
            replace(self.colors, **values['colors']),
            replace(self.typography, **values['typography']),
            replace(self.spacing, **values['spacing']))

    def to_style(self, **overrides):
        """Resolve semantic theme tokens to the low-level renderer style."""
        colors, typography, spacing = self.colors, self.typography, self.spacing
        return PDFStyle(
            font_name=typography.body_font, heading_font=typography.heading_font,
            plot_font=typography.plot_font, math_font=typography.math_font,
            font_size=typography.body_size, title_size=typography.title_size,
            heading_size=typography.heading_size, subheading_size=typography.subheading_size,
            caption_size=typography.caption_size, footer_size=typography.footer_size,
            line_height=typography.line_height, margin=spacing.margin,
            paragraph_spacing=spacing.paragraph_spacing, question_spacing=spacing.question_spacing,
            solution_spacing=spacing.solution_spacing, heading_spacing=spacing.heading_spacing,
            block_spacing=spacing.block_spacing, question_indent=spacing.question_indent,
            solution_indent=spacing.solution_indent, text_color=colors.text,
            heading_color=colors.heading, muted_color=colors.muted,
            rule_color=colors.border, background_color=colors.background,
            primary_color=colors.primary, accent_color=colors.accent,
            surface_color=colors.surface, success_color=colors.success,
            warning_color=colors.warning,
        ).with_changes(**overrides)


def _theme(name, *, colors=None, typography=None, spacing=None):
    return PDFTheme(name, colors or PDFThemeColors(), typography or PDFThemeTypography(),
                    spacing or PDFThemeSpacing())


_THEMES = {
    'academic': _theme('academic',
        colors=PDFThemeColors(heading='#153A5B', primary='#153A5B', accent='#8A4B08',
                              muted='#4B5563', surface='#F5F7FA', border='#AEB8C4'),
        typography=PDFThemeTypography(body_font='Times-Roman', heading_font='Times-Bold',
                                      math_font='stix', body_size=11.5, line_height=1.45)),
    'classroom': _theme('classroom',
        colors=PDFThemeColors(heading='#153E75', primary='#1D4E89', accent='#9A4D00',
                              muted='#4A5568', surface='#F3F7FC', border='#AEBFD2'),
        typography=PDFThemeTypography(body_size=12.5, title_size=21, heading_size=16,
                                      subheading_size=13.5, line_height=1.5),
        spacing=PDFThemeSpacing(margin=52, paragraph_spacing=10, question_spacing=14,
                                solution_spacing=12, heading_spacing=14, block_spacing=12,
                                question_indent=26, solution_indent=26)),
    'assessment': _theme('assessment',
        colors=PDFThemeColors(heading='#1F2937', primary='#1F2937', accent='#374151',
                              muted='#4B5563', surface='#F7F7F7', border='#9CA3AF'),
        typography=PDFThemeTypography(body_size=11.5, title_size=19, heading_size=15,
                                      subheading_size=12.5, line_height=1.45)),
    'engineering': _theme('engineering',
        colors=PDFThemeColors(heading='#164E63', primary='#164E63', accent='#9A4D00',
                              muted='#475569', surface='#F1F5F7', border='#94A3B8'),
        typography=PDFThemeTypography(body_size=10.5, title_size=19, heading_size=14.5,
                                      subheading_size=12, caption_size=9, footer_size=8.5,
                                      line_height=1.35),
        spacing=PDFThemeSpacing(margin=42, paragraph_spacing=7, question_spacing=9,
                                solution_spacing=8, heading_spacing=10, block_spacing=8,
                                question_indent=22, solution_indent=22)),
    'accessible': _theme('accessible',
        colors=PDFThemeColors(text='#111111', heading='#003B5C', muted='#3D3D3D',
                              primary='#003B5C', accent='#7A3E00', surface='#F4F7F8',
                              border='#626262', success='#145A2A', warning='#713800'),
        typography=PDFThemeTypography(body_size=13, title_size=23, heading_size=18,
                                      subheading_size=15, caption_size=11, footer_size=10,
                                      line_height=1.6),
        spacing=PDFThemeSpacing(margin=56, paragraph_spacing=13, question_spacing=16,
                                solution_spacing=14, heading_spacing=16, block_spacing=14,
                                question_indent=28, solution_indent=28)),
    'ink_saver': _theme('ink_saver',
        colors=PDFThemeColors(text='#111111', heading='#111111', muted='#444444',
                              primary='#111111', accent='#333333', background='#FFFFFF',
                              surface='#FFFFFF', border='#777777', success='#222222', warning='#222222'),
        typography=PDFThemeTypography(body_font='Times-Roman', heading_font='Times-Bold',
                                      math_font='stix', body_size=11, title_size=19,
                                      heading_size=15, subheading_size=12.5, line_height=1.4)),
}

PDF_THEMES = MappingProxyType(_THEMES)


def available_pdf_themes():
    """Return built-in PDF theme names in display order."""
    return tuple(_THEMES)


def get_pdf_theme(theme):
    """Resolve a theme name or return an existing PDFTheme."""
    if isinstance(theme, PDFTheme):
        return theme
    if not isinstance(theme, str):
        raise TypeError('theme must be a PDFTheme or theme name')
    key = theme.strip().lower().replace('-', '_').replace(' ', '_')
    try:
        return _THEMES[key]
    except KeyError as exc:
        choices = ', '.join(_THEMES)
        raise ValueError(f'Unknown PDF theme {theme!r}; choose from {choices}') from exc


__all__ = ['PDFTheme', 'PDFThemeColors', 'PDFThemeTypography', 'PDFThemeSpacing',
           'PDF_THEMES', 'available_pdf_themes', 'get_pdf_theme']
