"""Document-level content blocks independent of ReportLab layout objects."""
from dataclasses import dataclass
import math
from .formatting import PDFText


class PDFParagraph(PDFText):
    """Internal semantic paragraph retaining its public plain-text value."""
    def __new__(cls, content, *, number=None, role='body'):
        parts = list(content.parts) if isinstance(content, PDFText) else [str(content)]
        if parts and isinstance(parts[0], str):
            parts[0] = parts[0].lstrip()
        if parts and isinstance(parts[-1], str):
            parts[-1] = parts[-1].rstrip()
        instance = super().__new__(cls, *parts, plain=(f'{number}. ' if number is not None else '') + str(content))
        instance.number = number
        instance.role = role
        return instance


@dataclass(frozen=True)
class PDFHeading:
    text: str
    level: int = 1

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError('Heading text must not be empty')
        if isinstance(self.level, bool) or self.level not in (1, 2):
            raise ValueError('Heading level must be 1 or 2')


@dataclass(frozen=True)
class PDFAnswerSpace:
    """Writing area, optionally ruled or gridded; dimensions are PDF points."""
    height: float = 72
    pattern: str = 'lines'
    spacing: float = 18

    def __post_init__(self):
        for name in ('height', 'spacing'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.pattern not in ('lines', 'grid', 'blank'):
            raise ValueError('pattern must be lines, grid, or blank')


__all__ = ['PDFHeading', 'PDFAnswerSpace']
