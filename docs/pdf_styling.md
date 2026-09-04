# PDF styling

`PDFStyle` is an immutable, reusable design for manual and generated worksheets.

## Friendly themes

For most documents, start with a coordinated theme instead of configuring every
style value:

```python
sheet = kw.PDFWorksheet('Algebra practice', theme='classroom')
sheet.create('algebra.pdf')
```

Built-in themes are `academic`, `classroom`, `assessment`, `engineering`,
`accessible`, and `ink_saver`. The same `theme=` option works with
`PDFDocument`, `create_pages()`, `create_pdf()`, `worksheet()`, and export-time
overrides:

```python
sheet.create('compact.pdf', theme='engineering')
```

Use semantic options to create a reusable variant without mutating the preset:

```python
school = kw.PDFTheme.get('classroom').with_options(
    name='school', primary='#24543D', heading='#24543D', body_size=13,
)
sheet = kw.PDFWorksheet('Practice', theme=school)
```

`available_pdf_themes()` lists the names. `PDFStyle.theme('academic')` resolves
a theme to a low-level style when exact renderer options are needed. Passing
both `style=` and `theme=` is rejected because their precedence would be
ambiguous. Themes validate their fonts, measurements, colors, and a minimum
4.5:1 contrast for body, heading, and muted text against the page background.
`PDFStyle` remains fully supported for existing code and precise customization.
No external LaTeX installation or additional runtime dependency is required.

```python
import kiwicalc as kw

style = kw.PDFStyle(
    font_size=13, line_height=1.5,
    margin_left=60, margin_right=60,
    heading_color='#243B53',
    header='Algebra practice', footer='Page {page}',
)
sheet = kw.PDFWorksheet('Quadratics', style=style)
sheet.add_heading('Solve the equation')
sheet.add_exercise(kw.PDFQuadraticEquation())
sheet.add_answer_space(height=72, pattern='lines')
sheet.end_page()
sheet.create('worksheet.pdf')
```

## Precedence and compatibility

The default style is used when none is supplied. A worksheet's constructor style
applies to subsequent exports. `sheet.create(path, style=other)` replaces it for
that export only. Explicit export options override the selected style:
`sheet.create(path, font_size=14, line_height=1.25)`.
`style.with_changes(...)` returns a new validated style; it never edits the original.

The same `style=` and option overrides work in `worksheet()`, `create_pdf()`,
`create_pages()`, and equation classes' `random_worksheets()` methods. This includes
all eight batch families. Existing calls without a style continue to work.
Legacy singular `random_worksheet()` helpers, `LinearEquation.adjusted_worksheet()`,
and `Poly.export_report()` also use the same default renderer and accept styling
overrides. Questions and answers are laid out separately, rather than printing
Python tuples or lists. Report plots are rendered in memory; the old
`delete_image` argument remains accepted but is no longer needed. Existing PDF
files must be regenerated to receive these defaults.
`create_pdf()` retains its legacy boolean/error-warning contract; the other export
methods propagate errors. Unknown options are errors, not silently ignored.

## Options

All sizes and spacings below are PDF points (72 points = 1 inch), except
`line_height`, alignment names, font names, and DPI.

| Group | Options and defaults |
| --- | --- |
| Paper | `page_size='A4'`; also `'Letter'` or `(width, height)`; swap dimensions for landscape |
| Margins | `margin=50`; `margin_top`, `margin_right`, `margin_bottom`, `margin_left` default to `None`, inheriting `margin` |
| Text | `font_name='Helvetica'`, `font_size=12`, `line_height=1.5`, `alignment='left'` |
| Headings | `heading_font='Helvetica-Bold'`, `title_size=20`, `heading_size=16`, `subheading_size=13`, `title_alignment='left'` |
| Small text | `caption_size=10`, `footer_size=9` (also used for headers) |
| Vertical spacing | `paragraph_spacing=9.6`, `question_spacing=12`, `solution_spacing=10`, `heading_spacing=12`, `block_spacing=10` |
| Numbering | `question_indent=24`, `solution_indent=24`; automatically increased for wide numbers |
| Colors | `text_color='#202020'`, `heading_color='#202020'`, `muted_color='#555555'`, `rule_color='#BFC7D1'`, `background_color='#FFFFFF'` |
| Math | `math_font='dejavusans'`, `display_math_size=16`, `math_alignment='left'`, `math_dpi=200` |
| Plots | `plot_font='DejaVu Sans'`, `plot_line_width=1.5`, `plot_alignment='left'`, `plot_dpi=180` |
| Page furniture | `header=''`, `footer='Page {page}'`, `footer_rule=True`, `page_start=1` |
| Pagination | `keep_questions_together=True` (applies to answers too) |

Alignment values are `left`, `center`, or `right`. Colors accept named colors or
hex strings. Use colors recognized by both ReportLab and Matplotlib when they
affect math or plots; hex strings are the most portable choice. Fonts for text
and headings must be built-in ReportLab names or registered by the caller with
ReportLab's font registration API. `plot_font` is a Matplotlib font family.
Math font sets: `dejavusans`, `dejavuserif`, `cm`, `stix`, `stixsans`.

Headers and footers support literal text and `{page}` (physical PDF page number),
starting at `page_start` and incrementing automatically for every overflow and
answer page. Separate exports restart at 1 by default. When merging exports,
set the next export's `page_start` to the number of pages already merged plus 1.
Prefer `PDFDocument` below for KiwiCalc content: it requires no offset calculation.
Labels also support `{title}` (the first logical page's title). They wrap within the document
width; a label too tall for its margin raises an error. An empty string disables
the label. The footer rule is independently controlled by `footer_rule`.

## Content and spacing

### Composing a document with continuous numbering

```python
document = kw.PDFDocument()
document.add(first_worksheet).add(second_worksheet)
document.add_report(kw.Poly('x^2-4'))
document.create('combined.pdf')
```

`PDFDocument` renders the combined content once. Every section, answer page,
and overflow page is numbered automatically in one sequence. There is no merge
dependency and no need to calculate offsets. Repeated exports start again at
`page_start` (default 1); they never reuse a counter from the previous export.
Worksheet edits and enabled answer pages are refreshed at export time.

`add()` accepts a `PDFWorksheet` or `PDFPage`; `add_report()` accepts a `Poly`.
Each section begins on a new page. Question numbers retain each worksheet's
numbering, independently of the continuous physical page numbers.
The document's `style=` and export overrides govern every section; individual
worksheet styles are not applied inside a composed document. This API composes
KiwiCalc content, not existing PDF files or mixed page-size PDF imports.

### Reusable footer component

```python
footer = kw.PDFFooter('Practice | Page {page}', alignment='center', rule=True)
style = kw.PDFStyle(footer=footer)
document = kw.PDFDocument(style=style)
```

`PDFFooter` is shared by standalone exports and composed documents. Defaults:
text `Page {page}`, right alignment, and inherited `footer_size`, `muted_color`,
and `footer_rule`. Optional component overrides are `font_size`, `color`, and
`rule`. Existing footer strings still work and are converted into a component
by the shared renderer. Use `PDFFooter('', rule=False)` to hide the whole footer.
The component supports `{page}` and `{title}` and validates its configuration.

### Questions and layout blocks

`add_heading(text, level=1)` adds a section; level 2 adds a subsection. Neither is
numbered or copied to the answer key. Headings are kept with subsequent content.
`add_plot(source, height=180, caption='...')` keeps a caption with its image when
the group fits a page. `add_answer_space(height=72, pattern='lines', spacing=18)`
adds a writing area; patterns are `lines`, `grid`, and `blank`. Writing areas are
not copied to answer pages. Oversized answer areas raise a clear error.

Plain multiline questions and `PDFText` both treat a newline as a line break
inside one paragraph. Separate questions use `question_spacing`; solutions use
`solution_spacing`. Raw `create_pages()` text entries are separate paragraphs
using `paragraph_spacing`. Text is escaped literally, not interpreted as HTML.
Question/answer numbers sit in a hanging-indent column, so wrapped lines align
with the content. Leading/trailing whitespace is removed for rendering only.

Line height is a font-size multiplier, not paragraph spacing. Tall inline formulas
can expand lines to prevent clipping. Generated inline math inherits the body size;
`PDFMath(..., font_size=18)` overrides it for that formula. Unspecified display math
size inherits `display_math_size`. Formula objects and style objects are not mutated.

Short questions/answers stay together where possible; oversized paragraphs can
split across pages. Explicit `next_page()` boundaries are retained. Logical page
counts may differ from physical PDF counts after overflow. Captions or equations
that are larger than the available page may need a smaller size or a larger page;
inline formulas are indivisible and report an error if wider than the frame.

Plot callbacks inherit font, text color, background, and line-width settings;
explicit callback settings take precedence. Caller-supplied Figure objects retain
their own artist styling and are not closed. Export DPI applies to both. Callback
rendering creates no pyplot windows and restores global Matplotlib settings.

## Limitations and verification

This is a styling system, not a LaTeX engine or a PDF/UA accessibility claim.
Math and plots are still raster images. Tagged document structure, accessible
formula alternatives, arbitrary rich-text markup, and vector embedding are not
provided. Use good text/background contrast, and don't rely on color alone.

`examples/pdf_styling.ipynb` is an executable, git-ignored demonstration. Regression
tests cover style validation, overrides, all batch families, math inheritance,
plain/rich parity, hanging numbers, narrow-page overflow, and figure ownership.
