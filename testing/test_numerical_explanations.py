import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.mark.parametrize('options', [
    {'method': 'bisection', 'bracket': (0, 2)},
    {'method': 'bisection', 'bracket': (2, 0)},
    {'method': 'newton', 'x0': 1},
    {'method': 'newton', 'x0': 0, 'derivative': lambda x: 2*x},
    {'method': 'secant', 'x0': 1, 'x1': 2},
    {'method': 'newton', 'x0': 1, 'max_iterations': 1},
    {'method': 'bisection', 'bracket': (0, 2), 'max_iterations': 1},
    {'method': 'secant', 'x0': 1, 'x1': 2, 'max_iterations': 1},
])
def test_root_exact_parity_including_callback_order(options):
    sequences = []
    results = []
    warning_lists = []
    before = plt.get_fignums()
    for explain in (False, True):
        calls = []
        def f(x):
            calls.append(x)
            return x*x-2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results.append(kw.find_root(f, explain=explain, return_info=True, **options))
        sequences.append(calls)
        warning_lists.append([str(w.message) for w in caught])
    assert sequences[0] == sequences[1]
    assert results[0] == results[1].result
    assert warning_lists[0] == warning_lists[1]
    assert plt.get_fignums() == before


@pytest.mark.parametrize('method', ['midpoint', 'trapezoid', 'simpson'])
@pytest.mark.parametrize('bounds', [(0, 2), (2, -1), (1, 1)])
def test_quadrature_exact_parity_and_actual_samples(method, bounds):
    runs = []
    for explain in (False, True):
        calls = []
        def f(x):
            calls.append(x)
            return x*x+len(calls)*1e-5  # Stateful output exposes changed evaluation order.
        result = kw.integrate(f, *bounds, intervals=7, method=method, explain=explain, return_info=True)
        runs.append((result, calls))
    assert runs[0][0] == runs[1][0].result
    assert runs[0][1] == runs[1][1]
    for step in runs[1][0].steps:
        assert all(x in runs[1][1] for x in step.x)


def test_trace_limit_does_not_change_computation():
    result = kw.find_root(lambda x: x*x-2, bracket=(0, 2), method='bisection', explain=True, trace_limit=2)
    assert len(result.steps) == 2 < result.total_steps
    assert result.truncated and result.converged
    integral = kw.integrate(math.sin, 0, 1, intervals=1000, explain=True, trace_limit=2)
    assert len(integral.steps) == 2 and integral.total_steps == 500
    assert integral.value == kw.integrate(math.sin, 0, 1)
    with pytest.raises(ValueError, match='trace_limit'):
        kw.integrate(math.sin, 0, 1, explain=True, trace_limit=0)


def test_unsupported_methods_and_invalid_brackets():
    with pytest.raises(ValueError, match='explicitly'):
        kw.find_root(lambda x: x, bracket=(-1, 1), explain=True)
    with pytest.raises(ValueError, match='supports'):
        kw.integrate(math.sin, 0, 1, method='adaptive_simpson', explain=True)
    with pytest.raises(ValueError, match='opposite signs'):
        kw.find_root(lambda x: x*x+1, method='bisection', bracket=(-1, 1), explain=True)


@pytest.mark.parametrize('options', [
    {'method': 'newton', 'x0': 0, 'derivative': lambda x: 0},
    {'method': 'secant', 'x0': 0, 'x1': 1},
])
def test_stalled_roots_remain_explainable(options):
    with pytest.warns(UserWarning):
        result = kw.find_root(lambda x: -1., explain=True, **options)
    assert not result.converged and result.steps[-1].kind == 'stopped'
    if result.method == 'newton':
        assert result.steps[0].y == (-1.,)


def test_initial_roots_and_callback_errors():
    result = kw.find_root(lambda x: x, x0=0, explain=True)
    assert result.value == 0 and result.steps[0].kind == 'result'
    with pytest.raises(TypeError, match='callback failure'):
        kw.find_root(lambda x: (_ for _ in ()).throw(TypeError('callback failure')), x0=1, explain=True)


@pytest.mark.parametrize('method', ['bisection', 'newton', 'secant', 'midpoint', 'trapezoid', 'simpson'])
def test_static_plots_and_animation_use_recorded_steps(method):
    if method in ('midpoint', 'trapezoid', 'simpson'):
        result = kw.integrate(lambda x: x*x-2, 0, 2, intervals=4, method=method, explain=True)
    else:
        options = {'bracket': (0, 2)} if method == 'bisection' else {'x0': 1}
        if method == 'secant':
            options['x1'] = 2
        result = kw.find_root(lambda x: x*x-2, method=method, explain=True, **options)
    count = result.function_calls
    ax = result.plot_steps(0, show=False, samples=20)
    assert method in ax.get_title()
    plt.close(ax.figure)
    player = result.animate(samples=20)
    player.next()
    assert player.index == min(1, len(result.steps)-1)
    player.previous()
    assert player.index == 0
    player.set_speed(2)
    assert player.animation.event_source.interval == 350
    player.play()
    player.pause()
    plt.close(player.fig)
    assert result.function_calls == count


def test_convergence_and_validation():
    result = kw.find_root(lambda x: x*x-2, x0=1, explain=True)
    ax = result.plot_convergence(show=False)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [r.residual for r in result.steps])
    plt.close(ax.figure)
    for options in ({'step': 999}, {'step': True}, {'xlim': (1, 0)}, {'samples': 1}):
        with pytest.raises((TypeError, ValueError)):
            result.plot_steps(show=False, **options)
    integral = kw.integrate(math.sin, 0, 1, explain=True)
    with pytest.raises(ValueError, match='residual'):
        integral.plot_convergence(show=False)
    with pytest.raises(ValueError, match='no recorded'):
        kw.integrate(math.sin, 1, 1, explain=True).animate()


def test_html_animation_has_controls_and_no_extra_figure():
    result = kw.integrate(lambda x: x*x, 0, 1, intervals=2, method='midpoint', explain=True)
    player = result.animate(samples=10)
    html = player.to_jshtml()
    assert 'button' in html and 'slider' in html
    assert not plt.fignum_exists(player.fig.number)


def test_normal_path_never_calls_trace_helpers(monkeypatch):
    from kiwicalc.numeric import explanations
    def forbidden(*args, **kwargs):
        pytest.fail('Normal path entered tracing')
    monkeypatch.setattr(explanations, 'trace_root', forbidden)
    monkeypatch.setattr(explanations, 'trace_integral', forbidden)
    assert kw.find_root(lambda x: x*x-2, x0=1) == pytest.approx(math.sqrt(2))
    assert kw.integrate(lambda x: x*x, 0, 1) == pytest.approx(1/3)
