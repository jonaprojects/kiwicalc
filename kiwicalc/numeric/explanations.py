"""Opt-in teaching traces. Normal solver loops do not import or call this module.

Traced root loops preserve the scalar API's arithmetic, evaluation order,
stopping checks, and legacy warnings. Quadrature records are geometric panels
using actual samples, not a claim about the order of floating-point summation.
Keep parity tests in sync when modifying the underlying algorithms.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
from textwrap import fill
import warnings

import numpy as np

from kiwicalc.numeric.api import NumericalResult, _ScalarFunction, _RootFound, _count, _difference


@dataclass(frozen=True)
class NumericalStep:
    index: int
    kind: str
    x: Tuple[float, ...]
    y: Tuple[float, ...]
    estimate: float
    residual: Optional[float]
    formula: str
    decision: str
    bounds: Optional[Tuple[float, float]] = None


class _Recorder:
    def __init__(self, limit):
        self.limit = _count(limit, 'trace_limit')
        self.records = []
        self.total = 0

    def add(self, kind, x, y, estimate, residual, formula, decision, bounds=None):
        self.total += 1
        if len(self.records) < self.limit:
            self.records.append(NumericalStep(self.total, kind, tuple(x), tuple(y),
                                              estimate, residual, formula, decision, bounds))


@dataclass(frozen=True)
class NumericalExplanation:
    """A numerical result plus a bounded, immutable tuple of teaching records.

    explain=True always returns this object, including on non-convergence.
    Callback/input errors still propagate. Plotting may call the original f again;
    use pure callbacks for a reproducible background curve. Saved records and
    result.function_calls never change when rendering.
    """
    result: NumericalResult
    steps: Tuple[NumericalStep, ...]
    total_steps: int
    function: object = field(repr=False, compare=False)

    @property
    def value(self):
        return self.result.value

    @property
    def method(self):
        return self.result.method

    @property
    def converged(self):
        return self.result.converged

    @property
    def residual(self):
        return self.result.residual

    @property
    def function_calls(self):
        return self.result.function_calls

    @property
    def truncated(self):
        return len(self.steps) < self.total_steps

    def _background(self, xlim, samples):
        samples = _count(samples, 'samples')
        if samples < 2:
            raise ValueError('samples must be at least 2')
        if not self.steps:
            raise ValueError('This result has no recorded steps to plot')
        if xlim is None:
            xs = [x for record in self.steps for x in (record.bounds or record.x)]
            low, high = min(xs), max(xs)
            padding = .15*(high-low) if high != low else max(1., abs(low)*.15)
            xlim = low-padding, high+padding
        xlim = np.asarray(xlim, dtype=float)
        if xlim.shape != (2,) or not np.isfinite(xlim).all() or xlim[0] >= xlim[1]:
            raise ValueError('xlim must contain two increasing finite bounds')
        x = np.linspace(*xlim, samples)
        f = _ScalarFunction(self.function)
        y = np.array([f(float(value)) for value in x])
        low, high = min(0., float(y.min())), max(0., float(y.max()))
        padding = .12*max(high-low, 1.)
        return x, y, (low-padding, high+padding)

    def _draw(self, ax, index, background):
        record = self.steps[index]
        x, y, ylim = background
        ax.clear()
        ax.plot(x, y, color='navy', label='f(x)')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.grid(True, alpha=.25)
        if record.kind in ('newton', 'secant'):
            ax.plot(record.x, record.y, 'o--', color='darkorange', label=record.kind)
            ax.axvline(record.estimate, color='crimson', linestyle=':', label='current estimate')
        elif record.kind == 'bisection':
            ax.axvspan(*sorted(record.bounds), color='skyblue', alpha=.3, label='current bracket')
            ax.scatter(record.x, record.y, color='crimson', zorder=3, label='midpoint')
        elif record.kind in ('midpoint', 'trapezoid', 'simpson'):
            left, right = record.bounds
            if record.kind == 'midpoint':
                px, py = [left, right], [record.y[0], record.y[0]]
            elif record.kind == 'trapezoid':
                px, py = record.x, record.y
            else:
                # Lagrange interpolation in local coordinates avoids fitting a
                # polynomial in potentially very large absolute x coordinates.
                u = np.linspace(0, 1, 80)
                px = left + (right-left)*u
                a, b, c = record.y
                py = a*2*(u-.5)*(u-1) - b*4*u*(u-1) + c*2*u*(u-.5)
            ax.fill_between(px, py, color='skyblue', alpha=.45, label='panel approximation')
            ax.plot(px, py, color='darkorange')
            ax.scatter(record.x, record.y, color='crimson', zorder=3, label='actual samples')
        else:
            ax.scatter(record.x, record.y, color='crimson', zorder=3, label=record.kind)
        ax.set(xlim=(x[0], x[-1]), ylim=ylim, xlabel='x', ylabel='f(x)')
        ax.legend(loc='best')
        suffix = ' (trace truncated)' if self.truncated else ''
        ax.set_title(f'{self.method}: record {index+1}/{len(self.steps)}{suffix}')
        current = f'Estimate: {record.estimate:.7g} | residual: {record.residual:.3g}' if record.residual is not None else f'Panel contribution: {record.estimate:.7g}'
        status = f'Final value: {self.value:.7g}'
        if self.converged is not None:
            status += f' | converged: {self.converged}'
        formula = fill(record.formula, width=60)
        ax.text(.02, .98, f'{formula}\n{record.decision}\n{current}\n{status}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='lightgray'))
        return ax

    def plot_steps(self, step=-1, *, ax=None, show=True, xlim=None, samples=400):
        """Plot a zero-based saved step (last by default); return the Axes."""
        import matplotlib.pyplot as plt
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError('step must be an integer index')
        if not -len(self.steps) <= step < len(self.steps):
            raise ValueError('step is outside the recorded trace')
        background = self._background(xlim, samples)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))
        self._draw(ax, step % len(self.steps), background)
        if show:
            plt.show()
        return ax

    def plot_convergence(self, *, ax=None, show=True):
        """Plot recorded root residuals, not unknown true root errors."""
        import matplotlib.pyplot as plt
        records = [r for r in self.steps if r.residual is not None]
        if not records:
            raise ValueError('Convergence plots require root residual records')
        if ax is None:
            _, ax = plt.subplots()
        ax.plot([r.index for r in records], [r.residual for r in records], 'o-')
        ax.set_yscale('symlog', linthresh=1e-15)
        ax.set(xlabel='Recorded step', ylabel='|f(x)|', title=f'{self.method}: residual history')
        ax.grid(True, alpha=.3)
        if show:
            plt.show()
        return ax

    def animate(self, *, interval=700, xlim=None, samples=400):
        """Create a player; to_jshtml() provides notebook playback controls."""
        return NumericalAnimation(self, interval, self._background(xlim, samples))


class NumericalAnimation:
    """Lazy visualization with notebook play/pause, step, and speed controls.

    Native GUI backends can use play/pause/next/previous/set_speed. For inline
    notebooks display HTML(player.to_jshtml()); its controls require no widgets.
    """
    def __init__(self, explanation, interval, background):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from kiwicalc.numeric.api import _positive
        self.interval = _positive(interval, 'interval')
        self.explanation, self.background = explanation, background
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.index = 0
        self.animation = FuncAnimation(self.fig, self._frame,
                                       frames=len(explanation.steps), interval=self.interval,
                                       init_func=lambda: self.ax.lines, repeat=False, blit=False)
        self._frame(0)

    def _frame(self, index):
        self.index = index
        self.explanation._draw(self.ax, index, self.background)
        return self.ax.lines

    def next(self):
        self.pause()
        self._frame(min(self.index+1, len(self.explanation.steps)-1))
        self.fig.canvas.draw_idle()

    def previous(self):
        self.pause()
        self._frame(max(0, self.index-1))
        self.fig.canvas.draw_idle()

    def play(self):
        self.animation.frame_seq = iter(range(self.index, len(self.explanation.steps)))
        self.animation.event_source.start()

    def pause(self):
        self.animation.event_source.stop()

    def set_speed(self, multiplier):
        from kiwicalc.numeric.api import _positive
        interval = self.interval/_positive(multiplier, 'multiplier')
        self.animation._interval = interval
        self.animation.event_source.interval = interval

    def to_jshtml(self):
        import matplotlib.pyplot as plt
        html = self.animation.to_jshtml()
        plt.close(self.fig)  # Prevent an extra static figure in inline notebooks.
        return html


def trace_root(f, derivative, method, bracket, x0, x1, tolerance, nmax, limit):
    if method not in ('bisection', 'newton', 'secant'):
        raise ValueError('explain=True supports bisection, newton, and secant; choose method explicitly')
    recorder = _Recorder(limit)
    first = _ScalarFunction(derivative) if derivative is not None else None
    df = first if first is not None else lambda x: _difference(f, x, 'central', None)
    last = [None, None]
    last_y = [None]

    def evaluate(x):
        value = f(x)
        last[:] = [float(x), abs(value)]
        last_y[0] = value
        if abs(value) <= tolerance:
            recorder.add('result', [x], [value], x, abs(value), '|f(x)| <= tolerance', 'Residual tolerance satisfied')
            raise _RootFound(float(x), abs(value))
        return value

    converged = False
    candidate = None
    try:
        if method != 'bisection':
            evaluate(x0)
        if method == 'bisection':
            a, b = sorted(bracket)
            fa, fb = evaluate(a), evaluate(b)
            if not (fa < 0 < fb or fb < 0 < fa):
                raise ValueError('a and b must be of opposite signs')
            for _ in range(nmax):
                c = (a+b)/2
                fc = evaluate(c)
                right = fc*fa > 0
                recorder.add('bisection', [c], [fc], c, abs(fc), 'c = (a + b) / 2',
                             'Keep the right half' if right else 'Keep the left half', (a, b))
                if right:
                    a, fa = c, fc
                else:
                    b = c
        elif method == 'newton':
            current = x0
            if df(current) == 0:
                recorder.add('adjustment', [current], [last_y[0]], current, last[1], 'x = x + 0.1',
                             'Legacy initial zero-derivative adjustment')
                current += .1
            for _ in range(nmax):
                fx = evaluate(current)
                slope = df(current)
                if slope == 0:
                    warnings.warn('Newton-Raphson failed because the derivative is zero')
                    break
                next_x = current-fx/slope
                recorder.add('newton', [current, next_x], [fx, 0.], current, abs(fx),
                             "x_next = x - f(x) / f'(x)", f'Tangent intercept: {next_x:.7g}')
                current = next_x
            else:
                warnings.warn('The solution might have not converged properly')
            candidate = current
        else:
            a, b = x0, x1
            fa, fb = evaluate(a), evaluate(b)
            for _ in range(nmax):
                denominator = fa-fb
                if denominator == 0:
                    warnings.warn('The secant method failed because the function values are equal')
                    break
                d = (a-b)/denominator*fa
                next_x = a-d
                recorder.add('secant', [b, a, next_x], [fb, fa, 0.], a, abs(fa),
                             'x_next = x - f(x)*(x - previous)/(f(x) - f(previous))',
                             f'Secant intercept: {next_x:.7g}')
                b, a = a, next_x
                if abs(d) <= 0:
                    break
                fb, fa = fa, evaluate(a)
            else:
                warnings.warn('The solution might have not converged properly')
            candidate = a
        if candidate is not None:
            value = evaluate(candidate)
            recorder.add('stopped', [candidate], [value], candidate, abs(value), 'Residual tolerance not satisfied',
                         'Solver stopped without convergence')
    except _RootFound:
        converged = True
    message = 'Residual tolerance satisfied' if converged else 'Root solver did not satisfy the residual tolerance'
    result = NumericalResult(last[0], method, f.calls+(first.calls if first else 0), converged, last[1], message)
    return NumericalExplanation(result, tuple(recorder.records), recorder.total, f.function)


def trace_integral(f, a, b, method, intervals, limit):
    if method not in ('midpoint', 'trapezoid', 'simpson'):
        raise ValueError('explain=True supports midpoint, trapezoid, and simpson')
    recorder = _Recorder(limit)
    if a == b:
        return NumericalExplanation(NumericalResult(0., method, 0), (), 0, f.function)
    if method == 'simpson':
        intervals += intervals % 2
    dx = (b-a)/intervals

    if method == 'midpoint':
        def samples():
            for i in range(intervals):
                x = a+(i+.5)*dx
                y = f(x)
                recorder.add(method, [x], [y], dx*y, None, 'panel = width * f(midpoint)',
                             f'Signed panel contribution: {dx*y:.7g}', (a+i*dx, a+(i+1)*dx))
                yield y
        value = dx*sum(samples())
    elif method == 'trapezoid':
        def samples():
            for i in range(1, intervals+1):
                right, left = a+i*dx, a+(i-1)*dx
                yr, yl = f(right), f(left)
                contribution = .5*dx*(yr+yl)
                recorder.add(method, [left, right], [yl, yr], contribution, None,
                             'panel = width * (f(left) + f(right)) / 2',
                             f'Signed panel contribution: {contribution:.7g}', (left, right))
                yield yr+yl
        value = .5*dx*sum(samples())
    else:
        saved = {}
        def sample(i):
            x = a+i*dx
            # Legacy Simpson evaluates its final endpoint as b, not a+n*dx.
            x = b if i == intervals else x
            y = f(x)
            if i <= 2*recorder.limit:
                saved[i] = (x, y)
            return y
        odd = sum(sample(i) for i in range(1, intervals, 2))
        even = sum(sample(i) for i in range(2, intervals, 2))
        value = dx/3*(sample(0)+4*odd+2*even+sample(intervals))
        for i in range(min(intervals//2, recorder.limit)):
            nodes = [saved[2*i+j] for j in range(3)]
            xs, ys = zip(*nodes)
            contribution = dx/3*(ys[0]+4*ys[1]+ys[2])
            recorder.add(method, xs, ys, contribution, None,
                         'panel = width * (f(left) + 4*f(mid) + f(right)) / 6',
                         f'Signed panel contribution: {contribution:.7g}', (xs[0], xs[-1]))
        recorder.total = intervals//2
    from kiwicalc.numeric.api import _real
    result = NumericalResult(_real(value, 'integral'), method, f.calls)
    return NumericalExplanation(result, tuple(recorder.records), recorder.total, f.function)


__all__ = ['NumericalStep', 'NumericalExplanation', 'NumericalAnimation']
