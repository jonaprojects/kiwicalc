import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_piecewise_draws_lines_and_open_closed_endpoint_markers():
    result = kw.plot_piecewise(
        [((-2, 0), lambda x: -x, 'closed-open'),
         ((0, 2), lambda x: x*x, 'closed-closed')],
        samples=11, show=False,
    )

    assert isinstance(result, kw.PiecewisePlotResult)
    assert len(result.lines) == 2
    assert len(result.endpoint_markers) == 4
    assert all(sample.point_count == 11 for sample in result.samples)


def test_piecewise_supports_adaptive_sampling_per_piece():
    result = kw.plot_piecewise(
        [((0, 0.2), lambda x: np.sin(100*np.pi*x), '[]')],
        samples=21, sampling='adaptive', show=False,
    )

    assert result.samples[0].point_count > 21
    assert result.lines[0].kiwicalc_sample is result.samples[0]


def test_region_shades_between_functions_and_retains_coordinates():
    artist = kw.plot_region(
        lambda x: x*x, lambda x: 2*x + 3, interval=(-1, 3),
        samples=31, where=lambda x, low, high: high >= low, show=False,
    )

    assert artist.kiwicalc_sample.point_count == 31
    assert artist.kiwicalc_lower.shape == artist.kiwicalc_upper.shape
    assert len(artist.get_paths()) >= 1


def test_string_inequality_and_callable_predicate_are_supported():
    disk = kw.plot_inequality(
        'x^2 + y^2 <= 4', x_range=(-3, 3), y_range=(-3, 3),
        samples=31, show=False,
    )
    half_plane = kw.plot_inequality(
        lambda x, y: x > y, samples=20, boundary=False, show=False,
    )

    assert disk.kiwicalc_mask.shape == (31, 31)
    assert disk.kiwicalc_mask[15, 15]
    assert not disk.kiwicalc_mask[0, 0]
    assert half_plane.kiwicalc_boundary is None


def test_parametric_and_polar_convenience_functions_return_lines():
    parametric = kw.plot_parametric(
        lambda t: np.cos(t), lambda t: np.sin(t), samples=41,
        equal_aspect=True, show=False,
    )
    polar = kw.plot_polar(lambda theta: 1 + np.cos(theta), samples=51, show=False)

    assert len(parametric.get_xdata()) == 41
    assert len(polar.get_xdata()) == 51
    assert isinstance(parametric.kiwicalc_curve, kw.ParametricCurve2D)
    assert isinstance(polar.kiwicalc_curve, kw.PolarCurve2D)


def test_sequence_accepts_kiwicalc_sequence_callable_and_values():
    arithmetic = kw.ArithmeticProg([2], difference=3)
    first = kw.plot_sequence(arithmetic, start=1, stop=4, show=False)
    second = kw.plot_sequence(lambda n: n*n, indices=[1, 3, 5], show=False)
    third = kw.plot_sequence([5, 8], indices=[10, 20], show=False)

    assert list(first.kiwicalc_values) == [2, 5, 8, 11]
    assert list(second.kiwicalc_values) == [1, 9, 25]
    assert list(third.kiwicalc_indices) == [10, 20]


def test_error_band_supports_callable_uncertainty_and_adaptive_sampling():
    band = kw.plot_error_band(
        lambda x: np.sin(100*np.pi*x), lambda x: 0.1 + 0*x,
        start=0, stop=0.2, samples=21, sampling='adaptive', show=False,
    )

    assert band.kiwicalc_sample.point_count > 21
    assert np.allclose(band.kiwicalc_error, 0.1)
    assert band.kiwicalc_line.kiwicalc_sample is band.kiwicalc_sample


@pytest.mark.parametrize(
    'call, message',
    [
        (lambda: kw.plot_piecewise([], show=False), 'must not be empty'),
        (lambda: kw.plot_piecewise([((1, 0), lambda x: x)], show=False), 'increasing'),
        (lambda: kw.plot_inequality('x + y', show=False), 'must contain'),
        (lambda: kw.plot_sequence([1], indices=[1, 2], show=False), 'match'),
        (lambda: kw.plot_error_band(lambda x: x, -1, show=False), 'non-negative'),
    ],
)
def test_phase2_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()
