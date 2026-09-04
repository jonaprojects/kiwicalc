import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.mark.parametrize('method', ['rk4', 'rk45'])
@pytest.mark.parametrize('backward', [False, True])
def test_scalar_decay_and_requested_times(method, backward):
    times = np.linspace(0, 3, 31)
    if backward:
        times = times[::-1]
    result = kw.solve_ivp(lambda t, y: -.5*y, (times[0], times[-1]),
                          2*math.exp(-.5*times[0]), method=method, step=.03, t_eval=times)
    assert result.success and result.status == 'finished'
    np.testing.assert_array_equal(result.t, times)
    np.testing.assert_allclose(result.y, 2*np.exp(-.5*times), rtol=2e-7)
    assert result.y.shape == times.shape
    assert result.function_calls == (4 if method == 'rk4' else 7)*(result.steps+result.rejected_steps)


@pytest.mark.parametrize('method', ['rk4', 'rk45'])
def test_vector_oscillator_and_time_dependent_rhs(method):
    initial = np.array([1., 0.])
    solution = kw.solve_ivp(lambda t, y: [y[1], -y[0]], (0, 2*np.pi), initial,
                            method=method, max_step=.04, atol=[1e-9, 1e-9])
    assert solution.y.shape == (len(solution.t), 2)
    np.testing.assert_allclose(solution.y[:, 0], np.cos(solution.t), atol=2e-6)
    np.testing.assert_allclose(solution.y[:, 1], -np.sin(solution.t), atol=2e-6)
    np.testing.assert_array_equal(initial, [1, 0])
    nonautonomous = kw.solve_ivp(lambda t, y: 2*t, (1, 4), 1, method=method)
    assert nonautonomous.y[-1] == pytest.approx(16)


def test_rk4_order_and_rk45_tolerance_refinement():
    errors = [abs(kw.solve_ivp(lambda t, y: y, (0, 1), 1., method='rk4', step=h).y[-1]-math.e)
              for h in (.2, .1)]
    assert 12 < errors[0]/errors[1] < 20
    loose = kw.solve_ivp(lambda t, y: y, (0, 5), 1., rtol=1e-3, step=3)
    tight = kw.solve_ivp(lambda t, y: y, (0, 5), 1., rtol=1e-9, atol=1e-12, step=3)
    assert tight.rejected_steps > 0
    assert tight.steps > loose.steps
    assert abs(tight.y[-1]-math.exp(5)) < abs(loose.y[-1]-math.exp(5))/100


def test_terminal_event_and_unsorted_events():
    events = [kw.ODEEvent(lambda t, y: y-3),
              kw.ODEEvent(lambda t, y: y-2, terminal=True),
              kw.ODEEvent(lambda t, y: y-1)]
    result = kw.solve_ivp(lambda t, y: 1., (0, 4), 0., events=events, step=4,
                          t_eval=[0, 4])
    assert result.status == 'event' and result.success
    assert result.t[-1] == pytest.approx(2, abs=1e-9)
    assert result.y[-1] == pytest.approx(2, abs=1e-9)
    assert result.t_events[0].size == 0
    np.testing.assert_allclose(result.t_events[1], [2], atol=1e-9)
    np.testing.assert_allclose(result.t_events[2], [1], atol=1e-9)


@pytest.mark.parametrize('method', ['rk4', 'rk45'])
def test_nonlinear_event_localization(method):
    event = kw.ODEEvent(lambda t, y: y-.5, terminal=True, direction=-1)
    result = kw.solve_ivp(lambda t, y: -y, (0, 2), 1., events=event,
                          method=method, max_step=.03)
    assert result.t[-1] == pytest.approx(math.log(2), abs=2e-7)
    assert result.y_events[0][0] == pytest.approx(.5, abs=1e-8)


def test_event_directions_endpoints_and_initial_zero():
    result = kw.solve_ivp(lambda t, y: 1., (0, 2), 0., method='rk4', step=1,
                          events=[lambda t, y: t, kw.ODEEvent(lambda t, y: t-1, direction=1),
                                  kw.ODEEvent(lambda t, y: t-1, direction=-1)])
    np.testing.assert_array_equal(result.t_events[0], [0])
    np.testing.assert_array_equal(result.t_events[1], [1])
    assert result.t_events[2].size == 0
    backward = kw.solve_ivp(lambda t, y: 1., (2, 0), 2., events=kw.ODEEvent(lambda t, y: t-1, direction=-1))
    np.testing.assert_allclose(backward.t_events[0], [1], atol=1e-9)
    initial = kw.solve_ivp(lambda t, y: pytest.fail('must not evaluate'), (0, 1), 0.,
                           events=kw.ODEEvent(lambda t, y: y, terminal=True), t_eval=[1])
    assert initial.steps == 0 and initial.status == 'event'
    np.testing.assert_array_equal(initial.t, [0])


def test_periodic_events_and_vector_event_state():
    result = kw.solve_ivp(lambda t, y: [y[1], -y[0]], (0, 7), [1., 0.],
                          max_step=.1, events=lambda t, y: y[0])
    np.testing.assert_allclose(result.t_events[0], [np.pi/2, 3*np.pi/2], atol=2e-6)
    assert result.y_events[0].shape == (2, 2)
    np.testing.assert_allclose(result.y_events[0][:, 0], [0, 0], atol=1e-8)


def test_zero_span_caps_partial_results_and_no_figures():
    before = plt.get_fignums()
    empty = kw.solve_ivp(lambda t, y: pytest.fail('must not evaluate'), (3, 3), [1, 2])
    np.testing.assert_array_equal(empty.t, [3])
    np.testing.assert_array_equal(empty.y, [[1, 2]])
    assert empty.function_calls == 0
    capped = kw.solve_ivp(lambda t, y: 1., (0, 1), 0., step=1, max_step=.2)
    assert np.max(np.diff(capped.t)) <= .2+1e-15
    with pytest.raises(RuntimeError, match='steps'):
        kw.solve_ivp(lambda t, y: 1., (0, 1), 0., max_steps=1)
    partial = kw.solve_ivp(lambda t, y: 1., (0, 1), 0., max_steps=1,
                           t_eval=[1], raise_on_failure=False)
    assert not partial.success and partial.t.size == 0 and partial.y.shape == (0,)
    with pytest.raises(ValueError, match='samples'):
        partial.to_graph()
    stalled = kw.solve_ivp(lambda t, y: 1., (1, 2), 0., step=1e-30, raise_on_failure=False)
    assert not stalled.success and 'small' in stalled.message
    assert plt.get_fignums() == before


@pytest.mark.parametrize('options', [
    {'method': 'bad'}, {'step': 0}, {'max_step': -1}, {'rtol': -1}, {'atol': 0},
    {'atol': [1, 2]}, {'max_steps': 0}, {'raise_on_failure': 1},
    {'t_eval': []}, {'t_eval': [0, 0]}, {'t_eval': [1, 0]}, {'t_eval': [2]},
    {'events': [3]}, {'event_tolerance': 0},
])
def test_option_validation(options):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_ivp(lambda t, y: y, (0, 1), 1., **options)


@pytest.mark.parametrize('initial', [[], [[1]], True, 1j, np.nan])
def test_state_validation(initial):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_ivp(lambda t, y: y, (0, 1), initial)


def test_callback_contracts_and_mutation_safety():
    with pytest.raises(ValueError, match='shape'):
        kw.solve_ivp(lambda t, y: [1], (0, 1), 1.)
    with pytest.raises(ValueError, match='shape'):
        kw.solve_ivp(lambda t, y: 1, (0, 1), [1.])
    with pytest.raises(ValueError, match='finite'):
        kw.solve_ivp(lambda t, y: np.nan, (0, 1), 1.)
    calls = []
    def broken(t, y):
        calls.append(t)
        raise TypeError('inside callback')
    with pytest.raises(TypeError, match='inside callback'):
        kw.solve_ivp(broken, (0, 1), 1.)
    assert len(calls) == 1
    def mutating(t, y):
        y[:] = 999
        return [0., 0.]
    initial = np.array([1., 2.])
    result = kw.solve_ivp(mutating, (0, 1), initial)
    np.testing.assert_array_equal(result.y[-1], initial)


def test_numerical_overflow_returns_partial_failure():
    result = kw.solve_ivp(lambda t, y: 1e308, (0, 10), 1e308,
                          method='rk4', step=10, raise_on_failure=False)
    assert not result.success and 'Non-finite' in result.message
    np.testing.assert_array_equal(result.t, [0])
    with pytest.raises(RuntimeError, match='Non-finite'):
        kw.solve_ivp(lambda t, y: 1e308, (0, 10), 1e308, method='rk4', step=10)
    final_overflow = kw.solve_ivp(lambda t, y: 1e308, (0, 1), 0., method='rk4',
                                  step=1, raise_on_failure=False)
    assert not final_overflow.success and 'numerical step' in final_overflow.message


def test_event_localization_budget_does_not_report_false_success():
    result = kw.solve_ivp(lambda t, y: 0., (0, 1e100), 0., step=1e100,
                          events=lambda t, y: t-1e-200, raise_on_failure=False)
    assert not result.success and 'localization' in result.message
    assert result.t_events[0].size == 0


@pytest.mark.parametrize('f,span', [(3, (0, 1)), (lambda t, y: y, (0,)),
                                  (lambda t, y: y, (0, np.inf))])
def test_rhs_and_span_validation(f, span):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_ivp(f, span, 1.)


def test_plotting_and_public_exports():
    solution = kw.solve_ivp(lambda t, y: [1., -1.], (0, 1), [100., 200.])
    before = plt.get_fignums()
    graph = solution.to_graph(labels=['first', 'second'])
    assert plt.get_fignums() == before
    graph = solution.plot(show=False, title='ODE example', labels=['first', 'second'])
    assert len(graph.artists) == 2
    np.testing.assert_array_equal(graph.artists[0].get_xdata(), solution.t)
    assert graph.ax.get_ylim()[1] > 200
    plt.close(graph.fig)
    selected = solution.plot(components=(i for i in [0]), show=False)
    assert len(selected.artists) == 1 and selected.ax.get_ylim()[1] < 150
    plt.close(selected.fig)
    for options in ({'components': [2]}, {'components': []}, {'labels': ['one']}):
        with pytest.raises(ValueError):
            solution.to_graph(**options)
    scalar = kw.solve_ivp(lambda t, y: 0., (3, 3), 4.)
    graph = scalar.plot(show=False, labels='constant')
    assert graph.ax.get_xlim()[0] < 3 < graph.ax.get_xlim()[1]
    plt.close(graph.fig)
    for name in ('solve_ivp', 'ODESolution', 'ODEEvent'):
        assert name in kw.__all__


@pytest.mark.parametrize('options', [{'function': 1}, {'function': lambda t, y: y, 'terminal': 1},
                                    {'function': lambda t, y: y, 'direction': 2}])
def test_event_validation(options):
    with pytest.raises((TypeError, ValueError)):
        kw.ODEEvent(**options)
