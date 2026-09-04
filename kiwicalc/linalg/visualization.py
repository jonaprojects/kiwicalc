"""Math-aware explanations and visualizations for linear algebra.

The public entry points live on :class:`~kiwicalc.linalg.Matrix`.  Keeping the
Matplotlib imports here lets ordinary matrix arithmetic remain lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from kiwicalc.linalg.matrix import Matrix


@dataclass(frozen=True)
class RowOperation:
    """One elementary row operation and the matrix it produces."""

    description: str
    before: Matrix
    after: Matrix
    kind: str
    target_row: int
    source_row: Optional[int] = None
    scalar: Any = None


@dataclass(frozen=True)
class LinearAlgebraPlot:
    """A figure plus the numerical data used to build it.

    ``figure`` and ``axes`` provide normal Matplotlib access, while ``data``
    makes a visualization useful in lessons and tests without reading pixels.
    """

    figure: Any
    axes: Tuple[Any, ...]
    artists: Tuple[Any, ...] = ()
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def fig(self):
        """Short alias matching KiwiCalc's graph objects."""
        return self.figure

    @property
    def ax(self):
        """Return the first axes for single-panel visualizations."""
        return self.axes[0]

    def __iter__(self):
        yield self.figure
        yield self.axes

    def save(self, path, **kwargs):
        """Save the figure and return ``path`` for convenient chaining."""
        self.figure.savefig(path, **kwargs)
        return path


@dataclass(frozen=True)
class RowReductionExplanation:
    """A complete, inspectable sequence of elementary row operations."""

    initial: Matrix
    steps: Tuple[RowOperation, ...]
    result: Matrix
    pivot_columns: Tuple[int, ...]
    rank: int

    def as_text(self) -> str:
        """Return a concise human-readable derivation."""
        if not self.steps:
            return "The matrix is already in reduced row-echelon form."
        lines = [f"{index}. {step.description}" for index, step in enumerate(self.steps, 1)]
        lines.append(f"RREF complete: rank {self.rank}, pivot columns {self.pivot_columns}.")
        return "\n".join(lines)

    def plot(self, *, title="Row reduction", theme=None, show=True, columns=3):
        """Display the initial matrix and each elementary row-operation result."""
        import matplotlib.pyplot as plt

        if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
            raise ValueError("columns must be a positive integer")
        snapshots = [("Initial matrix", self.initial)] + [
            (step.description, step.after) for step in self.steps
        ]
        count = len(snapshots)
        rows = int(np.ceil(count / columns))
        fig, raw_axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.8 * rows), squeeze=False)
        axes = tuple(raw_axes.ravel())
        resolved = _resolve_theme(theme)
        artists = []
        for ax, (label, matrix) in zip(axes, snapshots):
            _apply_theme(fig, ax, resolved)
            ax.set_axis_off()
            table = ax.table(
                cellText=[[_format_number(value) for value in row] for row in matrix],
                cellLoc="center", loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(resolved.font_size if resolved else 11)
            table.scale(1, 1.35)
            ax.set_title(label)
            artists.append(table)
        for ax in axes[count:]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return LinearAlgebraPlot(fig, axes[:count], tuple(artists), {"explanation": self})


def explain_rref(matrix: Matrix, tolerance=None) -> RowReductionExplanation:
    """Reduce a matrix while recording every elementary row operation."""
    working = matrix.copy()
    initial = matrix.copy()
    effective_tolerance = Matrix._numeric_tolerance(working.matrix, tolerance)
    numeric = all(isinstance(value, (int, float, complex, np.number)) for value in working.yield_items())
    steps = []
    pivots = []
    pivot_row = 0

    def record(description, kind, target, action, *, source=None, scalar=None):
        before = working.copy()
        action()
        _clean_small_values(working, effective_tolerance)
        steps.append(RowOperation(description, before, working.copy(), kind, target, source, scalar))

    for pivot_column in range(working.num_of_columns):
        if pivot_row >= working.num_of_rows:
            break
        candidates = [
            row for row in range(pivot_row, working.num_of_rows)
            if not Matrix._is_zero(working.matrix[row][pivot_column], effective_tolerance)
        ]
        if not candidates:
            continue
        candidate = max(candidates, key=lambda row: abs(working.matrix[row][pivot_column])) if numeric else candidates[0]
        if candidate != pivot_row:
            record(
                f"R{pivot_row + 1} <-> R{candidate + 1}", "swap", pivot_row,
                lambda a=pivot_row, b=candidate: working.replace_rows(a, b), source=candidate,
            )
        pivot = working.matrix[pivot_row][pivot_column]
        if not Matrix._is_zero(pivot - 1, effective_tolerance):
            record(
                f"R{pivot_row + 1} <- ({_format_number(1 / pivot)})R{pivot_row + 1}",
                "scale", pivot_row,
                lambda value=pivot, row=pivot_row: working.divide_row(value, row), scalar=1 / pivot,
            )
        for row in range(working.num_of_rows):
            factor = working.matrix[row][pivot_column]
            if row == pivot_row or Matrix._is_zero(factor, effective_tolerance):
                continue
            scalar = -factor
            sign = "+" if _is_negative(factor) else "-"
            magnitude = _format_number(abs(factor)) if _supports_abs(factor) else _format_number(factor)
            description = f"R{row + 1} <- R{row + 1} {sign} {magnitude}R{pivot_row + 1}"
            record(
                description, "replace", row,
                lambda target=row, source=pivot_row, value=scalar: working.add_and_mul(target, source, value),
                source=pivot_row, scalar=scalar,
            )
        pivots.append(pivot_column)
        pivot_row += 1
    return RowReductionExplanation(initial, tuple(steps), working.copy(), tuple(pivots), len(pivots))


def visualize_transformation(matrix: Matrix, *, vectors=None, limits=(-2, 2), grid_lines=9,
                             title=None, theme=None, show=True) -> LinearAlgebraPlot:
    """Visualize how a 2×2 or 3×3 matrix transforms space."""
    import matplotlib.pyplot as plt

    values = _real_array(matrix, "visualize_transformation")
    dimension = _visual_dimension(values, "visualize_transformation")
    low, high = _limits(limits)
    if isinstance(grid_lines, bool) or not isinstance(grid_lines, int) or grid_lines < 2:
        raise ValueError("grid_lines must be an integer of at least 2")
    resolved = _resolve_theme(theme)
    projection = "3d" if dimension == 3 else None
    fig = plt.figure(figsize=(12, 5.5))
    axes = tuple(fig.add_subplot(1, 2, index + 1, projection=projection) for index in range(2))
    artists = []
    vector_values = _vectors(vectors, dimension)

    if dimension == 2:
        positions = np.linspace(low, high, grid_lines)
        samples = np.linspace(low, high, 120)
        for position in positions:
            horizontal = np.vstack((samples, np.full_like(samples, position)))
            vertical = np.vstack((np.full_like(samples, position), samples))
            for line in (horizontal, vertical):
                artists.extend(axes[0].plot(line[0], line[1], color="#9aa0a6", alpha=.45, linewidth=.8))
                transformed = values @ line
                artists.extend(axes[1].plot(transformed[0], transformed[1], color="#5f6368", alpha=.55, linewidth=.9))
        basis = np.eye(2)
        _draw_arrows_2d(axes[0], basis, resolved, artists, labels=("e₁", "e₂"))
        _draw_arrows_2d(axes[1], values @ basis, resolved, artists, labels=("Ae₁", "Ae₂"))
        if vector_values.size:
            _draw_arrows_2d(axes[0], vector_values.T, resolved, artists, labels=None, color="#7A3E9D")
            _draw_arrows_2d(axes[1], values @ vector_values.T, resolved, artists, labels=None, color="#7A3E9D")
        original_extent = max(abs(low), abs(high))
        corners = np.array([[low, low, high, high], [low, high, low, high]])
        transformed_extent = max(1.0, float(np.max(np.abs(values @ corners)))) * 1.12
        _configure_2d(axes[0], original_extent, "Original space", resolved)
        _configure_2d(axes[1], transformed_extent, "Transformed space", resolved)
    else:
        cube = _cube_edges(low, high)
        for edge in cube:
            artists.extend(axes[0].plot(*edge, color="#9aa0a6", alpha=.6, linewidth=1))
            transformed = values @ edge
            artists.extend(axes[1].plot(*transformed, color="#5f6368", alpha=.7, linewidth=1))
        _draw_arrows_3d(axes[0], np.eye(3), resolved, artists, labels=("e₁", "e₂", "e₃"))
        _draw_arrows_3d(axes[1], values, resolved, artists, labels=("Ae₁", "Ae₂", "Ae₃"))
        if vector_values.size:
            _draw_arrows_3d(axes[0], vector_values.T, resolved, artists, color="#7A3E9D")
            _draw_arrows_3d(axes[1], values @ vector_values.T, resolved, artists, color="#7A3E9D")
        original_extent = max(abs(low), abs(high))
        transformed_extent = max(1.0, float(np.max(np.abs(values @ np.array(list(_cube_vertices(low, high))).T)))) * 1.12
        _configure_3d(axes[0], original_extent, "Original space", resolved)
        _configure_3d(axes[1], transformed_extent, "Transformed space", resolved)

    fig.suptitle(title or f"Linear transformation by A ({dimension}D)")
    fig.tight_layout()
    if show:
        plt.show()
    data = {"matrix": values.copy(), "vectors": vector_values.copy(), "transformed_vectors": vector_values @ values.T}
    return LinearAlgebraPlot(fig, axes, tuple(artists), data)


def visualize_eigenvectors(matrix: Matrix, *, title=None, theme=None, show=True) -> LinearAlgebraPlot:
    """Plot real eigenvector directions and their transformed vectors."""
    import matplotlib.pyplot as plt

    values = matrix._as_numeric_array()
    dimension = _visual_dimension(values, "visualize_eigenvectors")
    eigenvalues, eigenvectors = np.linalg.eig(values)
    if np.max(np.abs(np.imag(eigenvalues))) > 1e-10 or np.max(np.abs(np.imag(eigenvectors))) > 1e-10:
        raise ValueError("visualize_eigenvectors() requires real eigenvectors; this matrix has complex eigenpairs")
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    transformed = values @ eigenvectors
    resolved = _resolve_theme(theme)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d" if dimension == 3 else None)
    artists = []
    extent = max(1.0, float(np.max(np.abs(np.column_stack((eigenvectors, transformed)))))) * 1.3
    labels = tuple(f"v{index + 1} (λ={_format_number(value)})" for index, value in enumerate(eigenvalues))
    if dimension == 2:
        angle = np.linspace(0, 2 * np.pi, 240)
        artists.extend(ax.plot(np.cos(angle), np.sin(angle), color="#9aa0a6", linewidth=1, label="unit circle"))
        _draw_arrows_2d(ax, eigenvectors, resolved, artists, labels=labels)
        _draw_arrows_2d(ax, transformed, resolved, artists, labels=tuple(f"Av{i + 1}" for i in range(dimension)),
                        linestyle="--", alpha=.65)
        _configure_2d(ax, extent, title or "Eigenvectors: Av = λv", resolved)
    else:
        _draw_arrows_3d(ax, eigenvectors, resolved, artists, labels=labels)
        _draw_arrows_3d(ax, transformed, resolved, artists, labels=tuple(f"Av{i + 1}" for i in range(dimension)),
                        linestyle="--", alpha=.65)
        _configure_3d(ax, extent, title or "Eigenvectors: Av = λv", resolved)
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return LinearAlgebraPlot(fig, (ax,), tuple(artists), {
        "matrix": np.asarray(values).copy(), "eigenvalues": eigenvalues.copy(),
        "eigenvectors": eigenvectors.copy(), "transformed_vectors": transformed.copy(),
    })


def visualize_svd(matrix: Matrix, *, title="How SVD transforms the unit circle", theme=None,
                  show=True) -> LinearAlgebraPlot:
    """Show the four geometric stages ``x → Vᵀx → ΣVᵀx → UΣVᵀx``."""
    import matplotlib.pyplot as plt

    values = _real_array(matrix, "visualize_svd")
    if values.shape != (2, 2):
        raise ValueError("visualize_svd() currently requires a 2x2 matrix")
    u, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    sigma = np.diag(singular_values)
    angle = np.linspace(0, 2 * np.pi, 320)
    circle = np.vstack((np.cos(angle), np.sin(angle)))
    stages = (circle, vt @ circle, sigma @ vt @ circle, u @ sigma @ vt @ circle)
    basis = np.eye(2)
    basis_stages = (basis, vt @ basis, sigma @ vt @ basis, u @ sigma @ vt @ basis)
    names = ("1. Unit circle", "2. Rotate/reflect with Vᵀ", "3. Scale with Σ", "4. Rotate/reflect with U")
    resolved = _resolve_theme(theme)
    fig, raw_axes = plt.subplots(1, 4, figsize=(16, 4.2), squeeze=False)
    axes = tuple(raw_axes.ravel())
    artists = []
    extent = max(1.0, *(float(np.max(np.abs(stage))) for stage in stages)) * 1.15
    for ax, stage, stage_basis, name in zip(axes, stages, basis_stages, names):
        artists.extend(ax.plot(stage[0], stage[1], linewidth=2.2))
        _draw_arrows_2d(ax, stage_basis, resolved, artists, labels=None)
        _configure_2d(ax, extent, name, resolved)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return LinearAlgebraPlot(fig, axes, tuple(artists), {
        "matrix": values.copy(), "u": u, "singular_values": singular_values,
        "vt": vt, "stages": tuple(stage.copy() for stage in stages),
    })


def visualize_least_squares(matrix: Matrix, right_hand_side, *, x=None, title="Least-squares fit",
                            xlabel="x", ylabel="y", theme=None, show=True) -> LinearAlgebraPlot:
    """Plot observations, fitted values, and residuals for a design matrix."""
    import matplotlib.pyplot as plt

    design = _real_array(matrix, "visualize_least_squares")
    response = matrix._as_rhs_array(right_hand_side)
    if response.shape[1] != 1:
        raise ValueError("visualize_least_squares() accepts one response column at a time")
    if np.iscomplexobj(response) and np.max(np.abs(np.imag(response))) > 1e-10:
        raise ValueError("visualize_least_squares() requires real response values")
    response = np.real(response[:, 0]).astype(float)
    if x is None:
        spreads = np.ptp(design, axis=0)
        varying = np.flatnonzero(spreads > 1e-12)
        if len(varying) != 1:
            raise ValueError("Pass x=... when the design matrix does not contain exactly one varying predictor column")
        x_values = design[:, varying[0]]
    else:
        x_values = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
        if x_values.ndim != 1 or len(x_values) != design.shape[0]:
            raise ValueError(f"x must contain exactly {design.shape[0]} values")
        if np.any(~np.isfinite(x_values)):
            raise ValueError("x values must be finite")
    solution = matrix.least_squares(response, return_info=True)
    coefficients = solution.solution.to_numpy(dtype=float).reshape(-1)
    fitted = design @ coefficients
    residuals = response - fitted
    order = np.argsort(x_values)
    resolved = _resolve_theme(theme)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _apply_theme(fig, ax, resolved)
    artists = []
    artists.append(ax.scatter(x_values, response, label="observations", zorder=3))
    artists.extend(ax.plot(x_values[order], fitted[order], label="least-squares fit", linewidth=2.2))
    residual_artist = ax.vlines(x_values, fitted, response, colors="#D14900", alpha=.7, label="residuals")
    artists.append(residual_artist)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return LinearAlgebraPlot(fig, (ax,), tuple(artists), {
        "x": np.asarray(x_values).copy(), "observed": response.copy(), "fitted": fitted,
        "residuals": residuals, "coefficients": coefficients, "solution": solution,
    })


def _resolve_theme(theme):
    from kiwicalc.plotting.themes import get_theme
    return get_theme(theme)


def _apply_theme(fig, ax, theme):
    from kiwicalc.plotting.themes import apply_theme
    apply_theme(fig, ax, theme)


def _format_number(value):
    if isinstance(value, (float, np.floating)):
        if abs(value) < 1e-12:
            return "0"
        return f"{value:.4g}"
    if isinstance(value, (complex, np.complexfloating)):
        return f"{value:.4g}"
    return str(value)


def _supports_abs(value):
    try:
        abs(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_negative(value):
    try:
        return bool(value < 0)
    except (TypeError, ValueError):
        return False


def _clean_small_values(matrix, tolerance):
    if not tolerance:
        return
    for row in range(matrix.num_of_rows):
        for column in range(matrix.num_of_columns):
            if Matrix._is_zero(matrix.matrix[row][column], tolerance):
                matrix.matrix[row][column] = 0.0


def _real_array(matrix, method):
    values = matrix._as_numeric_array()
    if np.iscomplexobj(values):
        if np.max(np.abs(np.imag(values))) > 1e-10:
            raise ValueError(f"{method}() requires a real matrix")
        values = np.real(values)
    return np.asarray(values, dtype=float)


def _visual_dimension(values, method):
    if values.shape not in ((2, 2), (3, 3)):
        raise ValueError(f"{method}() requires a 2x2 or 3x3 matrix")
    return values.shape[0]


def _limits(limits):
    try:
        low, high = (float(value) for value in limits)
    except (TypeError, ValueError) as exc:
        raise ValueError("limits must be a pair of finite numbers") from exc
    if not np.isfinite((low, high)).all() or low >= high:
        raise ValueError("limits must increase from a finite lower bound to a finite upper bound")
    return low, high


def _vectors(vectors, dimension):
    if vectors is None:
        return np.empty((0, dimension))
    if isinstance(vectors, Matrix):
        values = vectors._as_numeric_array()
    else:
        values = np.asarray(vectors, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"vectors must contain rows of {dimension} coordinates")
    if np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("vectors must contain finite real values")
    return np.asarray(values, dtype=float)


def _colors(theme, count):
    palette = theme.color_cycle if theme else ("#0072B2", "#D55E00", "#009E73")
    return tuple(palette[index % len(palette)] for index in range(count))


def _draw_arrows_2d(ax, vectors, theme, artists, labels=None, color=None, linestyle="-", alpha=1):
    for index, vector in enumerate(np.asarray(vectors).T):
        shade = color or _colors(theme, len(np.asarray(vectors).T))[index]
        label = labels[index] if labels else None
        artists.extend(ax.plot([0, vector[0]], [0, vector[1]], color=shade, linestyle=linestyle,
                               linewidth=2, alpha=alpha, label=label))
        artists.append(ax.scatter([vector[0]], [vector[1]], color=shade, s=26, alpha=alpha, zorder=4))


def _draw_arrows_3d(ax, vectors, theme, artists, labels=None, color=None, linestyle="-", alpha=1):
    for index, vector in enumerate(np.asarray(vectors).T):
        shade = color or _colors(theme, len(np.asarray(vectors).T))[index]
        label = labels[index] if labels else None
        artists.extend(ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], color=shade,
                               linestyle=linestyle, linewidth=2, alpha=alpha, label=label))
        artists.append(ax.scatter([vector[0]], [vector[1]], [vector[2]], color=shade, s=28, alpha=alpha))


def _configure_2d(ax, extent, title, theme):
    _apply_theme(ax.figure, ax, theme)
    ax.axhline(0, color="#5f6368", linewidth=.8, alpha=.55)
    ax.axvline(0, color="#5f6368", linewidth=.8, alpha=.55)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal", adjustable="box")


def _configure_3d(ax, extent, title, theme):
    _apply_theme(ax.figure, ax, theme)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent),
           xlabel="x", ylabel="y", zlabel="z", title=title)
    ax.set_box_aspect((1, 1, 1))


def _cube_vertices(low, high):
    for x in (low, high):
        for y in (low, high):
            for z in (low, high):
                yield (x, y, z)


def _cube_edges(low, high):
    vertices = list(_cube_vertices(low, high))
    edges = []
    for first_index, first in enumerate(vertices):
        for second in vertices[first_index + 1:]:
            if sum(a != b for a, b in zip(first, second)) == 1:
                edges.append(np.asarray((first, second), dtype=float).T)
    return edges


__all__ = [
    "RowOperation", "RowReductionExplanation", "LinearAlgebraPlot",
]
