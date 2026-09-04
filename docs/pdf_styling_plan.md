# PDF styling system completion checklist

The active goal is a complete, reusable PDF styling system, not merely new
defaults. Implementation and verification must cover:

- Validated reusable style with typography, colors, sizing, spacing, independent
  margins, alignment, heading hierarchy, and header/footer controls.
- Shared style across manual and batch export APIs; explicit override precedence.
- Inherited math size/font/color, with explicit per-formula overrides preserved.
- Consistent plain/rich newline handling and semantic hanging question/answer indents.
- Section headings, captions, answer-space blocks, keep-together and page-break behavior.
- Style-aware plots without changing caller-owned figures or global plotting state.
- Public documentation and executable notebook examples with all options described.
- Regression tests for validation, overrides, compatibility, pagination and ownership;
  complete test suite and existing coverage gates.
- Rendered PDF examples inspected on every page, including narrow pages, long
  questions, nested fractions, captions, and overflow.

Tagged PDF, accessible math alternatives, full LaTeX, and vector embedding are
separate rendering/accessibility features, not claims made by a styling API.
Document these limitations explicitly; do not imply PDF/UA compliance.

## Verification record

- `style.py`: immutable validated style; options and precedence documented in
  `docs/pdf_styling.md`.
- `worksheet.py` and equation batch methods: style forwarding verified for all
  eight families, with and without answers (`test_every_batch_family_forwards_style`).
- Math inheritance, explicit overrides, plain/rich parity, hanging indents,
  custom fonts, plot ownership and global-state restoration covered in
  `testing/test_pdf_style.py`.
- Section/caption/answer-space rendering and narrow overflow tested. Visual review
  found and fixed an oversized-question heading-stranding defect; a dedicated
  regression now guards it.
- `examples/pdf_styling.ipynb` validated and all code cells executed successfully.
  All five final showcase pages and four narrow overflow pages inspected visually.
- Full suite: 1,826 passed, 11 warnings; line coverage 94.3821%, branch coverage
  90.4075%. Both existing 90% gates passed. `git diff --check` passed.
- No new runtime dependencies, external LaTeX requirement, or PDF/UA claim.
