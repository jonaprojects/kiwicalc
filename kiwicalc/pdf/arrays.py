"""Dependency-free matrix and vector notation for KiwiCalc PDF documents."""
from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from io import BytesIO
from numbers import Complex, Real
import math

from .formatting import format_math


_BRACKETS = {
    'square': 'square', 'bracket': 'square', 'brackets': 'square',
    'round': 'round', 'parenthesis': 'round', 'parentheses': 'round',
    'determinant': 'determinant', 'bars': 'determinant', 'bar': 'determinant',
    'none': 'none', None: 'none',
}


def _scalar(value):
    """Convert NumPy scalars while leaving KiwiCalc expressions untouched."""
    if type(value).__module__.startswith('numpy') and hasattr(value, 'item'):
        return value.item()
    return value


def _cell_math(value):
    value = _scalar(value)
    if isinstance(value, complex) and not isinstance(value, Real):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError('Matrix entries must be finite')
        if value.imag == 0:
            return format_math(value.real)
        real = '' if value.real == 0 else format_math(value.real)
        magnitude = abs(value.imag)
        imaginary = 'i' if magnitude == 1 else format_math(magnitude)+'i'
        if value.real == 0:
            return ('-' if value.imag < 0 else '')+imaginary
        return real+('+' if value.imag > 0 else '-')+imaginary
    return format_math(value)


def _sequence(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f'{name} must be a KiwiCalc object, NumPy array, or sequence')
    return list(value)


def _matrix_rows(values):
    if hasattr(values, 'to_list') and callable(values.to_list):
        values = values.to_list()
    elif type(values).__module__.startswith('numpy') and hasattr(values, 'tolist'):
        values = values.tolist()
    outer = _sequence(values, 'matrix')
    if not outer:
        raise ValueError('matrix must not be empty')
    first = outer[0]
    nested = isinstance(first, Iterable) and not isinstance(first, (str, bytes))
    rows = [_sequence(row, 'matrix row') for row in outer] if nested else [outer]
    if not rows[0]:
        raise ValueError('matrix rows must not be empty')
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError('matrix rows must all have the same length')
    if len(rows) > 30 or columns > 30:
        raise ValueError('PDF matrices support at most 30 rows and 30 columns')
    return tuple(tuple(_scalar(value) for value in row) for row in rows)


def _vector_values(values):
    if hasattr(values, 'direction'):
        values = values.direction
    elif hasattr(values, 'to_list') and callable(values.to_list):
        rows = values.to_list()
        if len(rows) == 1:
            values = rows[0]
        elif rows and all(len(row) == 1 for row in rows):
            values = [row[0] for row in rows]
        else:
            raise ValueError('a matrix used as a vector must have one row or one column')
    elif type(values).__module__.startswith('numpy') and hasattr(values, 'tolist'):
        values = values.tolist()
        if values and isinstance(values[0], list):
            if len(values) == 1:
                values = values[0]
            elif all(len(row) == 1 for row in values):
                values = [row[0] for row in values]
            else:
                raise ValueError('an array used as a vector must be one-dimensional')
    result = _sequence(values, 'vector')
    if not result:
        raise ValueError('vector must not be empty')
    if any(isinstance(value, Iterable) and not isinstance(value, (str, bytes)) for value in result):
        raise ValueError('vector entries must be scalar values')
    if len(result) > 30:
        raise ValueError('PDF vectors support at most 30 entries')
    return tuple(_scalar(value) for value in result)


class PDFArray:
    """Structured two-dimensional math block shared by matrices and vectors."""

    def __init__(self, rows, *, brackets='square', font_size=None):
        try:
            brackets = _BRACKETS[brackets]
        except (KeyError, TypeError) as exc:
            raise ValueError('brackets must be square, round, determinant, or none') from exc
        if font_size is not None and (isinstance(font_size, bool)
                                      or not isinstance(font_size, (int, float))
                                      or not math.isfinite(font_size) or font_size <= 0):
            raise ValueError('font_size must be positive and finite')
        self.values = _matrix_rows(rows)
        self.brackets = brackets
        self.font_size = font_size
        # Validate at construction so malformed entries fail close to the caller.
        self.expressions = tuple(tuple(_cell_math(value) for value in row)
                                 for row in self.values)

    @property
    def shape(self):
        return len(self.values), len(self.values[0])

    def __str__(self):
        return '['+'; '.join(', '.join(str(value) for value in row)
                            for row in self.values)+']'

    def image(self, *, font_size=16, fontset='dejavusans', color='black', dpi=200):
        """Render cells and brackets without invoking an external TeX engine."""
        from PIL import Image as PILImage, ImageDraw
        from matplotlib import rc_context
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image

        size = self.font_size or font_size
        images = []
        with rc_context({'text.usetex': False, 'mathtext.fontset': fontset}):
            for row in self.expressions:
                rendered = []
                for expression in row:
                    data = BytesIO()
                    try:
                        math_to_image('$'+expression+'$', data,
                                      prop=FontProperties(size=size), dpi=dpi,
                                      color=color, format='png')
                    except ValueError as exc:
                        raise ValueError(f'Unsupported matrix entry {expression!r}: {exc}') from exc
                    data.seek(0)
                    rendered.append(PILImage.open(data).convert('RGBA').copy())
                images.append(rendered)

        scale = dpi/72
        gap_x, gap_y = max(4, round(size*.55*scale)), max(2, round(size*.25*scale))
        pad_x, pad_y = max(4, round(size*.45*scale)), max(2, round(size*.2*scale))
        column_widths = [max(row[column].width for row in images)
                         for column in range(self.shape[1])]
        row_heights = [max(cell.height for cell in row) for row in images]
        grid_width = sum(column_widths)+gap_x*(self.shape[1]-1)
        grid_height = sum(row_heights)+gap_y*(self.shape[0]-1)
        bracket_width = 0 if self.brackets == 'none' else max(5, round(size*.4*scale))
        width = grid_width+2*pad_x+2*bracket_width
        height = grid_height+2*pad_y
        canvas = PILImage.new('RGBA', (width, height), (255, 255, 255, 0))

        y = pad_y
        for row_index, row in enumerate(images):
            x = pad_x+bracket_width
            for column, cell in enumerate(row):
                cell_x = x+(column_widths[column]-cell.width)//2
                cell_y = y+(row_heights[row_index]-cell.height)//2
                canvas.alpha_composite(cell, (cell_x, cell_y))
                x += column_widths[column]+gap_x
            y += row_heights[row_index]+gap_y

        if self.brackets != 'none':
            draw = ImageDraw.Draw(canvas)
            stroke = max(2, round(1.1*scale))
            left, right = pad_x//2, width-pad_x//2-1
            top, bottom = max(1, pad_y//3), height-max(1, pad_y//3)-1
            hook = max(4, bracket_width//2)
            if self.brackets == 'square':
                draw.line((left+hook, top, left, top, left, bottom, left+hook, bottom),
                          fill=color, width=stroke, joint='curve')
                draw.line((right-hook, top, right, top, right, bottom, right-hook, bottom),
                          fill=color, width=stroke, joint='curve')
            elif self.brackets == 'determinant':
                draw.line((left, top, left, bottom), fill=color, width=stroke)
                draw.line((right, top, right, bottom), fill=color, width=stroke)
            else:
                arc_width = max(hook*2, bracket_width)
                draw.arc((left, top, left+arc_width, bottom), 90, 270, fill=color, width=stroke)
                draw.arc((right-arc_width, top, right, bottom), 270, 90, fill=color, width=stroke)

        output = BytesIO()
        canvas.save(output, format='PNG', dpi=(dpi, dpi))
        output.seek(0)
        return output


class PDFMatrix(PDFArray):
    """A renderable matrix accepting KiwiCalc, NumPy, or nested-sequence data."""

    def __init__(self, values, *, brackets='square', font_size=None):
        super().__init__(_matrix_rows(values), brackets=brackets, font_size=font_size)


class PDFVector(PDFArray):
    """A renderable row or column vector; KiwiCalc vectors use their direction."""

    def __init__(self, values, *, orientation='column', brackets='round', font_size=None):
        if orientation not in ('column', 'row'):
            raise ValueError('orientation must be column or row')
        entries = _vector_values(values)
        rows = tuple((value,) for value in entries) if orientation == 'column' else (entries,)
        super().__init__(rows, brackets=brackets, font_size=font_size)
        self.orientation = orientation


__all__ = ['PDFArray', 'PDFMatrix', 'PDFVector']
