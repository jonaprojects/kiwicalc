import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_ecdf_aggregates_duplicates_and_reports_missing_values():
    result = kw.ecdf([3, 1, 1, 2, np.nan])

    assert isinstance(result, kw.ECDFResult)
    np.testing.assert_array_equal(result.values, [1, 2, 3])
    np.testing.assert_array_equal(result.counts, [2, 1, 1])
    np.testing.assert_allclose(result.probabilities, [0.5, 0.75, 1])
    assert result.sample_size == 4
    assert result.missing == 1


def test_ecdf_nan_and_data_validation():
    with pytest.raises(ValueError, match='NaN'):
        kw.ecdf([1, np.nan], nan_policy='raise')
    with pytest.raises(ValueError, match='nan_policy'):
        kw.ecdf([1, 2], nan_policy='propagate')
    with pytest.raises(ValueError, match='finite'):
        kw.ecdf([1, math.inf])
    with pytest.raises(ValueError, match='at least'):
        kw.ecdf([])


def test_qq_data_fits_normal_by_default_and_accepts_distribution():
    sample = [-2, -1, 0, 1, 2]
    fitted = kw.qq_data(sample)
    specified = kw.qq_data(sample, kw.Normal())

    assert isinstance(fitted, kw.QQData)
    assert isinstance(fitted.distribution, kw.Normal)
    assert fitted.fitted is True
    assert specified.fitted is False
    np.testing.assert_array_equal(fitted.observed, sample)
    assert np.all(np.diff(fitted.theoretical) > 0)
    assert math.isfinite(fitted.reference_slope)
    assert math.isfinite(fitted.reference_intercept)


def test_pp_data_uses_plotting_positions_and_distribution_cdf():
    sample = [-1, 0, 1, 2]
    result = kw.pp_data(sample, kw.Normal())

    assert isinstance(result, kw.PPData)
    np.testing.assert_allclose(result.empirical, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(result.theoretical, kw.Normal().cdf(sample))
    assert np.all((result.theoretical >= 0) & (result.theoretical <= 1))


def test_quantile_diagnostic_validation_is_clear():
    with pytest.raises(ValueError, match='non-constant'):
        kw.qq_data([2, 2, 2])
    with pytest.raises(TypeError, match='continuous'):
        kw.qq_data([1, 2, 3], kw.Binomial(2, 0.5))
    with pytest.raises(ValueError, match="'normal'"):
        kw.pp_data([1, 2, 3], 'uniform')


def test_assumption_summary_reports_shape_outliers_and_context():
    result = kw.assumption_summary([1, 1, 1, 2, 2, 100, np.nan])

    assert isinstance(result, kw.AssumptionSummary)
    assert result.count == 6
    assert result.missing == 1
    assert result.outlier_count == 1
    assert result.outlier_fraction == pytest.approx(1 / 6)
    assert result.skewness > 1
    assert result.has_messages
    assert any('subject-matter judgment' in message for message in result.messages)
    assert result.as_dict()['count'] == 6


def test_constant_assumption_summary_is_defined():
    result = kw.assumption_summary([4, 4, 4, 4])

    assert result.constant
    assert result.standard_deviation == 0
    assert math.isnan(result.skewness)
    assert math.isnan(result.excess_kurtosis)


def test_ecdf_plot_supports_result_objects_and_themes():
    result = kw.ecdf([3, 1, 2, 2])
    axes = result.plot(show=False, theme='classroom', color='navy')

    assert isinstance(axes, Axes)
    assert axes.get_title() == 'Empirical cumulative distribution'
    assert axes.get_ylim()[1] >= 1
    np.testing.assert_allclose(axes.lines[0].get_ydata(), result.probabilities)


def test_qq_and_pp_plots_include_reference_lines():
    sample = [-1.2, -0.7, -0.1, 0.3, 0.8, 1.4]
    qq_axes = kw.qq_plot(sample, show=False)
    pp_axes = kw.pp_plot(sample, show=False)

    assert isinstance(qq_axes, Axes)
    assert 'Q-Q plot' in qq_axes.get_title()
    assert len(qq_axes.collections) == 1
    assert len(qq_axes.lines) == 1
    assert pp_axes.get_xlim()[0] < 0 and pp_axes.get_xlim()[1] > 1
    np.testing.assert_allclose(pp_axes.lines[0].get_xdata(), [0, 1])
    np.testing.assert_allclose(pp_axes.lines[0].get_ydata(), [0, 1])


def test_histogram_can_overlay_fitted_or_supplied_density():
    sample = [-2, -1, -0.5, 0, 0.2, 0.8, 1, 2]
    fitted = kw.histogram_plot(sample, fit='normal', bins=4, show=False)
    supplied = kw.histogram_plot(sample, fit=kw.Normal(), bins=4, show=False)

    assert len(fitted.patches) == 4
    assert len(fitted.lines) == 1
    assert fitted.lines[0].get_label() == 'Fitted Normal'
    assert supplied.lines[0].get_label() == 'Normal'
    assert fitted.get_ylabel() == 'Density'


def test_confidence_interval_plot_handles_multiple_and_array_results():
    first = kw.ConfidenceInterval(1, 3, 0.95, 2, method='Method A')
    second = kw.ConfidenceInterval(2, 6, 0.95, 4, method='Method B')
    horizontal = kw.confidence_interval_plot(
        [first, second], reference=0, show=False,
    )
    array_interval = kw.ConfidenceInterval(
        np.array([0, 1]), np.array([2, 3]), 0.95, np.array([1, 2]),
    )
    vertical = kw.confidence_interval_plot(
        array_interval, labels=['A', 'B'], orientation='vertical', show=False,
    )

    assert [tick.get_text() for tick in horizontal.get_yticklabels()] == [
        'Method A', 'Method B',
    ]
    assert horizontal.get_title() == 'Confidence intervals'
    assert [tick.get_text() for tick in vertical.get_xticklabels()] == ['A', 'B']
    assert vertical.get_ylabel() == 'Estimate'


def test_confidence_interval_plot_validation():
    invalid = kw.ConfidenceInterval(3, 4, 0.95, 2)
    with pytest.raises(ValueError, match='within'):
        kw.confidence_interval_plot(invalid, show=False)
    with pytest.raises(ValueError, match='labels'):
        kw.confidence_interval_plot(
            kw.ConfidenceInterval(1, 3, 0.95, 2), labels=['A', 'B'], show=False,
        )
    with pytest.raises(ValueError, match='orientation'):
        kw.confidence_interval_plot(
            kw.ConfidenceInterval(1, 3, 0.95, 2), orientation='diagonal', show=False,
        )


def test_diagnostic_dashboard_has_four_named_panels():
    sample = [-2, -1, -0.5, 0, 0.4, 0.9, 1.3, 2]
    figure, axes = kw.diagnostic_plots(sample, show=False, theme='publication')

    assert axes.shape == (2, 2)
    assert figure._suptitle.get_text() == 'Statistical diagnostics'
    assert 'density' in axes[0, 0].get_title().lower()
    assert axes[0, 1].get_title() == 'Empirical cumulative distribution'
    assert 'Q-Q' in axes[1, 0].get_title()
    assert 'P-P' in axes[1, 1].get_title()


def test_statistical_diagnostic_exports():
    for name in (
        'ECDFResult', 'QQData', 'PPData', 'AssumptionSummary', 'ecdf',
        'qq_data', 'pp_data', 'assumption_summary', 'plot_ecdf', 'qq_plot',
        'pp_plot', 'histogram_plot', 'confidence_interval_plot',
        'diagnostic_plots',
    ):
        assert hasattr(kw, name)
