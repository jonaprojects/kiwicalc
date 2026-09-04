"""Explicit initial-value ODE solvers for non-stiff, real-valued problems."""
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from kiwicalc.numeric.api import _count, _method, _positive, _real
from kiwicalc.numeric.advanced import _array


@dataclass(frozen=True)
class ODEEvent:
    """A zero-crossing callback event(t, state).

    terminal stops at the first matching crossing. direction is -1, 0, or +1,
    measured along integration order (including backward integration). An exact
    initial zero is recorded regardless of direction. Tangencies and multiple
    crossings within a step can be missed; reduce max_step to resolve them.
    """
    function: Callable
    terminal: bool = False
    direction: int = 0

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError('Event function must be callable')
        if not isinstance(self.terminal, bool):
            raise TypeError('terminal must be a boolean')
        if isinstance(self.direction, bool) or self.direction not in (-1, 0, 1):
            raise ValueError('direction must be -1, 0, or 1')


@dataclass(frozen=True)
class ODESolution:
    """Time-first results: y.shape is (times,) or (times, components).

    success is True on reaching the endpoint or a terminal event. status is
    'finished', 'event', or 'failed'. steps counts accepted steps; rejected_steps
    counts adaptive rejections. function_calls includes event localization's
    right-hand-side evaluations; event_calls counts event callback evaluations.
    t_events/y_events contain one array per supplied event, including empty ones.
    """
    t: np.ndarray
    y: np.ndarray
    method: str
    success: bool
    status: str
    message: str
    steps: int
    rejected_steps: int
    function_calls: int
    event_calls: int
    t_events: Tuple[np.ndarray, ...]
    y_events: Tuple[np.ndarray, ...]

    def to_graph(self, *, components=None, labels=None):
        """Build an unrendered Graph2D with sampled time-series curves."""
        from kiwicalc.geometry.curves import Curve2D
        from kiwicalc.plotting.plots import Graph2D

        if not self.t.size:
            raise ValueError('There are no output samples to plot')
        data = self.y[:, None] if self.y.ndim == 1 else self.y
        selected = list(range(data.shape[1])) if components is None else list(components)
        if not selected or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                               or not 0 <= i < data.shape[1] for i in selected):
            raise ValueError('components must contain valid component indices')
        if labels is None:
            labels = ['y' if self.y.ndim == 1 else f'y[{i}]' for i in selected]
        elif isinstance(labels, str):
            labels = [labels]
        else:
            labels = list(labels)
        if len(labels) != len(selected):
            raise ValueError('labels must match the selected components')

        class TimeSeries(Curve2D):
            def __init__(self, times, values):
                super().__init__(samples=max(2, len(times)))
                self.times, self.values = times.copy(), values.copy()

            def sample(self, samples=None):
                return self.times.copy(), self.values.copy()

        graph = Graph2D()
        for i, label in zip(selected, labels):
            graph.add(TimeSeries(self.t, data[:, i]), label=label)
        return graph

    def plot(self, *, components=None, labels=None, **options):
        """Render through Graph2D and return that graph; accepts show=False."""
        components = None if components is None else list(components)
        graph = self.to_graph(components=components, labels=labels)
        options.setdefault('xlabel', 't')
        options.setdefault('ylabel', 'state')
        options.setdefault('legend', True)
        options.setdefault('grid', True)
        low, high = float(min(self.t)), float(max(self.t))
        padding = .05 * max(1., abs(low)) if low == high else 0.
        options.setdefault('xlim', (low-padding, high+padding))
        plotted = self.y if self.y.ndim == 1 or components is None else self.y[:, components]
        low, high = float(np.min(plotted)), float(np.max(plotted))
        padding = .05 * max(high-low, abs(high), 1.)
        options.setdefault('ylim', (low-padding, high+padding))
        graph.plot(**options)
        return graph


# Dormand-Prince 5(4) tableau. The fifth-order solution advances the state;
# the difference from the embedded fourth-order formula controls local error.
_C = (0., 1/5, 3/10, 4/5, 8/9, 1., 1.)
_A = ((), (1/5,), (3/40, 9/40), (44/45, -56/15, 32/9),
      (19372/6561, -25360/2187, 64448/6561, -212/729),
      (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
      (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84))
_B4 = np.array((5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40))
_ERROR = np.array((*_A[-1], 0.)) - _B4


class _StepFailure(Exception):
    """Numerical breakdown, distinct from invalid callback output."""


def _step(rhs, t, y, h, method):
    if method == 'rk4':
        k1 = rhs(t, y)
        k2 = rhs(t+h/2, y+h*k1/2)
        k3 = rhs(t+h/2, y+h*k2/2)
        k4 = rhs(t+h, y+h*k3)
        return y+h*(k1+2*k2+2*k3+k4)/6, np.zeros_like(y)
    stages = []
    for c, row in zip(_C, _A):
        state = y + h * sum((a*k for a, k in zip(row, stages)), np.zeros_like(y))
        stages.append(rhs(t+c*h, state))
    advanced = y + h*sum((a*k for a, k in zip(_A[-1], stages)), np.zeros_like(y))
    error = h*sum((a*k for a, k in zip(_ERROR, stages)), np.zeros_like(y))
    return advanced, error


def solve_ivp(f, t_span, initial, *, method='rk45', step=None, rtol=1e-6,
              atol=1e-9, max_step=None, max_steps=100000, t_eval=None,
              events=None, event_tolerance=1e-10, raise_on_failure=True):
    """Solve y'=f(t,y) on a finite interval, without a SciPy dependency.

    A scalar initial value gets scalar callbacks; a nonempty 1D initial state
    gets copied arrays and must return the same shape. Callback errors propagate.
    RK4 uses step (default interval/100); RK45 adapts it (default interval/100
    for the first attempt). rtol is nonnegative, atol positive scalar/per-state.
    RK45 accepts steps with max(abs(error)/(atol+rtol*max(abs(y),abs(y_new))))<=1.
    These are local error estimates, not guarantees of global solution accuracy.

    max_step caps positive step magnitude; max_steps bounds attempted steps.
    t_eval must be ordered in integration direction within t_span. Steps land
    directly on requested times, not interpolated samples; this may alter the
    step sequence. Without t_eval, all accepted endpoints are returned. A terminal
    event is appended even when it is not in t_eval. Failure returns partial
    requested samples only if raise_on_failure=False (otherwise RuntimeError).

    Events are callables or ODEEvent objects, located by bisection with substeps
    of the selected RK method, to event_tolerance in time (subject to roundoff).
    Backward integration and zero-length intervals are supported. Stiff problems,
    complex states, dense output, and boundary-value problems are not supported.
    """
    if not callable(f):
        raise TypeError('f must be callable as f(t, state)')
    method = _method(method, ('rk4', 'rk45'))
    span = _array(t_span, 't_span')
    if span.shape != (2,) or not np.isfinite(span[1]-span[0]):
        raise ValueError('t_span must contain two bounds with finite width')
    start, end = map(float, span)
    state = _array(initial, 'initial')
    if state.ndim > 1 or state.size == 0:
        raise ValueError('initial must be scalar or a nonempty one-dimensional state')
    scalar = state.ndim == 0
    state = state.reshape(-1)
    rtol = _real(rtol, 'rtol')
    absolute = _array(atol, 'atol')
    if rtol < 0 or absolute.shape not in ((), state.shape) or np.any(absolute <= 0):
        raise ValueError('rtol must be nonnegative; atol must be positive scalar or match the state')
    limit = _count(max_steps, 'max_steps')
    cap = abs(end-start) if max_step is None else _positive(max_step, 'max_step')
    h_next = abs(end-start)/100 if step is None else _positive(step, 'step')
    event_tolerance = _positive(event_tolerance, 'event_tolerance')
    if not isinstance(raise_on_failure, bool):
        raise TypeError('raise_on_failure must be a boolean')
    direction = 1 if end >= start else -1
    requested = None if t_eval is None else _array(t_eval, 't_eval')
    if requested is not None:
        if (requested.ndim != 1 or requested.size == 0
                or np.any(direction*np.diff(requested) <= 0)
                or np.any(requested < min(start, end)) or np.any(requested > max(start, end))):
            raise ValueError('t_eval must be nonempty, strictly ordered, and inside t_span')
    if events is None:
        event_list = []
    elif isinstance(events, ODEEvent) or callable(events):
        event_list = [events]
    else:
        event_list = list(events)
    event_list = [e if isinstance(e, ODEEvent) else ODEEvent(e) for e in event_list]
    calls, event_calls = 0, 0

    def external(y):
        return float(y[0]) if scalar else y.copy()

    def rhs(t, y):
        nonlocal calls
        if not np.isfinite(y).all():
            raise _StepFailure('Non-finite internal state; reduce step or use a stiff solver')
        calls += 1
        value = _array(f(float(t), external(y)), 'derivative output')
        expected = () if scalar else state.shape
        if value.shape != expected:
            raise ValueError('Derivative output must match the initial state shape')
        return value.reshape(-1)

    def event_value(index, t, y):
        nonlocal event_calls
        event_calls += 1
        return _real(event_list[index].function(float(t), external(y)), 'event output')

    def locate(index, t, y, right, yr, gl, gr):
        if gr == 0:
            return right, yr.copy()
        left = t
        for _ in range(100):
            middle = left + (right-left)/2
            if middle in (left, right):
                break
            ym, _ = _step(rhs, t, y, middle-t, method)
            gm = event_value(index, middle, ym)
            if gm == 0 or abs(right-left) <= event_tolerance:
                return middle, ym
            if np.signbit(gm) == np.signbit(gl):
                left, gl = middle, gm
            else:
                right, yr = middle, ym
        else:
            raise _StepFailure('Event localization did not converge; reduce max_step')
        return right, yr.copy()

    times, values = [], []
    event_times = [[] for _ in event_list]
    event_states = [[] for _ in event_list]
    position = 0
    if requested is None or requested[0] == start:
        times.append(start)
        values.append(state.copy())
        position = 1 if requested is not None else 0
    current = start
    old_events = [event_value(i, current, state) for i in range(len(event_list))]
    status, message = 'finished', 'Reached the integration endpoint'
    for i, value in enumerate(old_events):
        if value == 0:
            event_times[i].append(current)
            event_states[i].append(state.copy())
            if event_list[i].terminal:
                status, message = 'event', 'Terminal event at the initial time'
    accepted, rejected, attempts = 0, 0, 0
    while current != end and status == 'finished':
        if attempts >= limit:
            status, message = 'failed', 'Maximum attempted steps reached'
            break
        target = end
        if requested is not None and position < len(requested):
            target = float(requested[position])
        magnitude = min(h_next, cap, abs(target-current))
        next_time = target if magnitude == abs(target-current) else current + direction*magnitude
        if next_time == current:
            status, message = 'failed', 'Step is too small to advance time'
            break
        h = next_time-current
        attempts += 1
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                candidate, error = _step(rhs, current, state, h, method)
        except _StepFailure as exc:
            status, message = 'failed', str(exc)
            break
        if not np.isfinite(candidate).all() or not np.isfinite(error).all():
            status, message = 'failed', 'Non-finite numerical step'
            break
        if method == 'rk45':
            scale = absolute + rtol*np.maximum(abs(state), abs(candidate))
            ratio = float(np.max(abs(error)/scale))
            factor = 5. if ratio == 0 else min(5., max(.2, .9*ratio**(-.2)))
            h_next = abs(h)*factor
            if ratio > 1:
                rejected += 1
                continue
        new_events = [event_value(i, next_time, candidate) for i in range(len(event_list))]
        crossings = []
        for i, (gl, gr) in enumerate(zip(old_events, new_events)):
            # Excluding gl==0 avoids recording the same endpoint twice.
            crosses = gl != 0 and (gr == 0 or np.signbit(gl) != np.signbit(gr))
            matches = event_list[i].direction == 0 or (1 if gr > gl else -1) == event_list[i].direction
            if crosses and matches:
                try:
                    te, ye = locate(i, current, state, next_time, candidate, gl, gr)
                except _StepFailure as exc:
                    status, message = 'failed', str(exc)
                    break
                crossings.append((te, i, ye))
        if status == 'failed':
            break
        crossings.sort(key=lambda item: direction*item[0])
        terminal_time = None
        for te, i, ye in crossings:
            if terminal_time is not None and direction*(te-terminal_time) > event_tolerance:
                break
            event_times[i].append(te)
            event_states[i].append(ye.copy())
            if event_list[i].terminal and terminal_time is None:
                terminal_time = te
                next_time, candidate = te, ye
                status, message = 'event', 'Terminal event detected'
        current, state = next_time, candidate
        old_events = new_events
        accepted += 1
        if requested is None or status == 'event' or (position < len(requested) and current == requested[position]):
            times.append(current)
            values.append(state.copy())
            if requested is not None and position < len(requested) and current == requested[position]:
                position += 1
    if status == 'event' and (not times or times[-1] != current):
        times.append(current)
        values.append(state.copy())
    shape = (-1,) if scalar else (-1, len(state))
    result = ODESolution(np.asarray(times, dtype=float), np.asarray(values, dtype=float).reshape(shape),
                         method, status != 'failed', status, message, accepted, rejected, calls, event_calls,
                         tuple(np.asarray(v, dtype=float) for v in event_times),
                         tuple(np.asarray(v, dtype=float).reshape(shape) for v in event_states))
    if not result.success and raise_on_failure:
        raise RuntimeError(message)
    return result


__all__ = ['solve_ivp', 'ODESolution', 'ODEEvent']
