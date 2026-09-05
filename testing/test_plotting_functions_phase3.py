import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_phase_portrait_renders_field_trajectories_and_equilibrium():
    result = kw.plot_phase_portrait(
        lambda x, y: -y, lambda x, y: x,
        x_range=(-2, 2), y_range=(-2, 2), density=21,
        initial_conditions=[(1, 0), (0.5, 0)], trajectory_steps=101,
        t_span=(0, 2), show=False,
    )

    assert isinstance(result, kw.PhasePortraitResult)
    assert len(result.trajectories) == 2
    assert result.equilibria is not None
    assert result.trajectories[0].kiwicalc_trajectory.shape == (101, 2)


def test_phase_portrait_accepts_combined_system_callable():
    result = kw.plot_phase_portrait(
        lambda x, y: (-y, x), density=11,
        initial_conditions=[(1, 0)], trajectory_steps=20, show=False,
    )

    assert len(result.trajectories) == 1


@pytest.mark.parametrize('mode', ['domain', 'magnitude', 'phase', 'real', 'imaginary'])
def test_complex_function_modes(mode):
    image = kw.plot_complex_function(
        lambda z: z*z, samples=25, mode=mode,
        colorbar=mode != 'domain', show=False,
    )

    assert image.kiwicalc_values.shape == (25, 25)
    assert image.kiwicalc_mode == mode
    assert (image.kiwicalc_colorbar is None) == (mode == 'domain')


def test_complex_function_scalar_fallback_handles_nonvectorized_callable():
    image = kw.plot_complex_function(complex.__abs__, samples=8, mode='magnitude', show=False)

    assert np.isfinite(image.kiwicalc_values).all()


def test_convergence_accepts_values_and_numerical_explanation():
    line = kw.plot_convergence([1, 0.1, 0.01], show=False)
    explanation = kw.find_root(
        lambda x: x*x - 2, method='bisection', bracket=(0, 2),
        explain=True, tolerance=1e-8,
    )
    traced = kw.plot_convergence(explanation, show=False)

    assert list(line.get_ydata()) == [1, 0.1, 0.01]
    assert len(traced.get_ydata()) > 1
    assert traced.axes.get_yscale() == 'log'


def test_transform_compares_points_under_homogeneous_matrix():
    matrix = np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]], dtype=float)
    result = kw.plot_transform(
        [(0, 0), (1, 0)], matrix, connectors=2, show=False,
    )

    assert isinstance(result, kw.TransformPlotResult)
    assert np.allclose(result.transformed_points, [[2, 3], [2, 4]])
    assert len(result.connectors) == 2


def test_transform_accepts_curves_and_callable_transform():
    curve = kw.ParametricCurve2D(lambda t: t, lambda t: t*t, t_range=(0, 1), samples=20)
    result = kw.plot_transform(
        curve, lambda points: points + np.array([1, -1]), samples=20, show=False,
    )

    assert len(result.original.get_xdata()) == 20
    assert result.transformed_points[0] == pytest.approx(result.original_points[0] + [1, -1])


def test_bifurcation_is_bounded_and_retains_generated_data():
    artist = kw.plot_bifurcation(
        lambda state, parameter: parameter*state*(1-state),
        parameter_range=(2.8, 4), parameter_samples=30,
        burn_in=40, keep=10, show=False,
    )

    assert 0 < len(artist.kiwicalc_states) <= 300
    assert len(artist.kiwicalc_parameters) == len(artist.kiwicalc_states)
    assert artist.kiwicalc_iterations == 1500


@pytest.mark.parametrize(
    'call, error, message',
    [
        (lambda: kw.plot_phase_portrait(1, density=10, show=False), TypeError, 'system'),
        (lambda: kw.plot_complex_function(lambda z: z, mode='unknown', show=False), ValueError, 'mode'),
        (lambda: kw.plot_convergence([1, -0.1], show=False), ValueError, 'non-negative'),
        (lambda: kw.plot_transform([(0, 0)], np.eye(2), show=False), ValueError, 'transform'),
        (lambda: kw.plot_bifurcation(lambda x, p: x, parameter_range=(0, 1), parameter_samples=20, keep=10, max_points=100, show=False), ValueError, 'max_points'),
    ],
)
def test_phase3_validation(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_expensive_plotters_enforce_computation_caps():
    with pytest.raises(ValueError, match='max_field_points'):
        kw.plot_phase_portrait(lambda x, y: (-y, x), density=20,
                               max_field_points=100, show=False)
    with pytest.raises(ValueError, match='max_trajectory_points'):
        kw.plot_phase_portrait(lambda x, y: (-y, x), density=5,
                               initial_conditions=[(1, 0)] * 5,
                               trajectory_steps=30, max_trajectory_points=100,
                               show=False)
    with pytest.raises(ValueError, match='max_points'):
        kw.plot_complex_function(lambda z: z, samples=20, max_points=100,
                                 show=False)
    with pytest.raises(ValueError, match='max_iterations'):
        kw.plot_bifurcation(lambda x, p: x, parameter_range=(0, 1),
                            parameter_samples=20, burn_in=20, keep=5,
                            max_iterations=100, show=False)


def test_exact_zero_convergence_uses_symlog_scale():
    line = kw.plot_convergence([1, 0.1, 0], show=False)

    assert line.axes.get_yscale() == 'symlog'
