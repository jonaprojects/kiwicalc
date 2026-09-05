import math

import numpy as np
import pytest

import kiwicalc as kw


def test_basic_location_statistics_accept_lists_and_arrays():
    assert kw.mean([1, 2, 3, 4]) == 2.5
    assert kw.median(np.array([1, 2, 3, 4])) == 2.5
    assert kw.mode(['red', 'blue', 'red']) == 'red'


def test_mode_breaks_ties_by_first_appearance_and_supports_weights():
    assert kw.mode(['b', 'a']) == 'b'
    assert kw.mode(['a', 'b'], weights=[1, 3]) == 'b'


def test_extrema_and_range_ignore_zero_weight_observations():
    data = [1, 5, 100]
    weights = [1, 1, 0]
    assert kw.minimum(data, weights=weights) == 1
    assert kw.maximum(data, weights=weights) == 5
    assert kw.data_range(data, weights=weights) == 4
    assert kw.min(data) == 1
    assert kw.max(data) == 100


def test_quantiles_percentiles_quartiles_and_iqr_match_numpy():
    data = np.arange(1, 9)
    np.testing.assert_allclose(kw.quantile(data, [0.25, 0.5, 0.75]),
                               np.quantile(data, [0.25, 0.5, 0.75]))
    np.testing.assert_allclose(kw.percentile(data, [25, 50, 75]),
                               np.percentile(data, [25, 50, 75]))
    np.testing.assert_allclose(kw.quartiles(data), [2.75, 4.5, 6.25])
    assert kw.iqr(data) == 3.5


def test_weighted_quantiles_and_equal_weight_compatibility():
    assert kw.median([0, 10], weights=[3, 1]) == 0
    assert kw.median([1, 2, 3, 4], weights=[2, 2, 2, 2]) == 2.5


def test_axis_reductions_have_numpy_compatible_shapes():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(kw.mean(data, axis=0), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(kw.median(data, axis=1), [2, 5])
    np.testing.assert_allclose(
        kw.quartiles(data, axis=1),
        np.quantile(data, [0.25, 0.5, 0.75], axis=1),
    )


def test_one_dimensional_axis_weights_broadcast_over_other_dimensions():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(kw.mean(data, axis=1, weights=[1, 1, 2]), [2.25, 5.25])


def test_population_and_sample_variance_families():
    data = [1, 2, 3, 4]
    assert kw.variance(data) == pytest.approx(np.var(data))
    assert kw.population_variance(data) == pytest.approx(np.var(data))
    assert kw.sample_variance(data) == pytest.approx(np.var(data, ddof=1))
    assert kw.standard_deviation(data) == pytest.approx(np.std(data))
    assert kw.std(data) == pytest.approx(np.std(data))
    assert kw.population_standard_deviation(data) == pytest.approx(np.std(data))
    assert kw.sample_standard_deviation(data) == pytest.approx(np.std(data, ddof=1))


def test_weighted_variance_uses_total_weight_minus_ddof():
    data = np.array([1.0, 2.0, 8.0])
    weights = np.array([1.0, 2.0, 1.0])
    center = np.average(data, weights=weights)
    expected = np.sum(weights * (data - center) ** 2) / (weights.sum() - 1)
    assert kw.variance(data, weights=weights, ddof=1) == pytest.approx(expected)


def test_absolute_deviations():
    assert kw.mean_absolute_deviation([1, 2, 3]) == pytest.approx(2 / 3)
    assert kw.median_absolute_deviation([1, 2, 100]) == 1


def test_geometric_harmonic_trimmed_and_weighted_means():
    assert kw.geometric_mean([1, 4, 16]) == pytest.approx(4)
    assert kw.harmonic_mean([1, 2, 4]) == pytest.approx(12 / 7)
    assert kw.trimmed_mean([0, 1, 2, 3, 100], 0.2) == 2
    assert kw.weighted_mean([1, 3], [1, 3]) == 2.5


@pytest.mark.parametrize('function', [kw.geometric_mean, kw.harmonic_mean])
def test_positive_mean_domains_are_explicit(function):
    with pytest.raises(ValueError, match='positive'):
        function([0, 1])


def test_skewness_kurtosis_and_coefficient_of_variation():
    symmetric = [-2, -1, 0, 1, 2]
    assert kw.skewness(symmetric) == pytest.approx(0)
    assert kw.kurtosis(symmetric, excess=False) == pytest.approx(1.7)
    assert kw.kurtosis(symmetric) == pytest.approx(-1.3)
    assert kw.coefficient_of_variation([1, 2, 3]) == pytest.approx(
        np.std([1, 2, 3]) / 2
    )


@pytest.mark.parametrize('function', [kw.skewness, kw.kurtosis])
def test_standardized_moments_reject_constant_data(function):
    with pytest.raises(ValueError, match='constant'):
        function([3, 3, 3])


def test_coefficient_of_variation_rejects_zero_mean():
    with pytest.raises(ValueError, match='mean is zero'):
        kw.coefficient_of_variation([-1, 1])


def test_frequency_and_proportion_tables_preserve_first_seen_order():
    table = kw.frequency_table(['pear', 'apple', 'pear'])
    assert table.values.tolist() == ['pear', 'apple']
    assert table.counts.tolist() == [2, 1]
    np.testing.assert_allclose(table.proportions, [2 / 3, 1 / 3])
    assert table.to_dict() == {'pear': 2, 'apple': 1}
    assert kw.proportion_table(['a', 'b', 'a']) == {'a': 2 / 3, 'b': 1 / 3}


def test_weighted_frequency_table():
    table = kw.frequency_table(['a', 'b', 'a'], weights=[0.5, 2, 1.5])
    assert table.to_dict() == {'a': 2.0, 'b': 2.0}
    np.testing.assert_allclose(table.proportions, [0.5, 0.5])


def test_frequency_tables_support_axis():
    tables = kw.frequency_table([[1, 1, 2], [3, 3, 3]], axis=1)
    assert tables.shape == (2,)
    assert tables[0].to_dict() == {1: 2, 2: 1}
    assert tables[1].to_dict() == {3: 3}


def test_five_number_summary_and_describe():
    summary = kw.five_number_summary([1, 2, 3, 4, 100])
    assert summary.as_dict() == {'min': 1, 'q1': 2, 'median': 3, 'q3': 4, 'max': 100}
    described = kw.describe([1, 2, 3, 4])
    assert described.count == 4
    assert described.mean == 2.5
    assert described.std == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert described['iqr'] == 1.5


def test_outlier_fences_and_detection():
    data = np.array([1, 2, 2, 3, 100])
    fences = kw.outlier_fences(data)
    assert tuple(fences) == (0.5, 4.5)
    np.testing.assert_array_equal(kw.detect_outliers(data), [False, False, False, False, True])


def test_outlier_detection_broadcasts_axis_fences():
    data = np.array([[1, 2, 3, 100], [10, 11, 12, 13]])
    mask = kw.detect_outliers(data, axis=1, factor=1)
    assert mask.shape == data.shape
    assert mask[0, -1]
    assert not np.any(mask[1])


def test_covariance_and_pearson_match_numpy():
    x = np.array([1, 2, 4, 8], dtype=float)
    y = np.array([3, 2, 7, 9], dtype=float)
    assert kw.covariance(x, y) == pytest.approx(np.cov(x, y, ddof=1)[0, 1])
    assert kw.pearson_correlation(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    matrix = np.column_stack([x, y])
    np.testing.assert_allclose(kw.covariance(matrix), np.cov(matrix, rowvar=False))
    np.testing.assert_allclose(kw.pearson_correlation(matrix), np.corrcoef(matrix, rowvar=False))


def test_paired_statistics_support_axis():
    x = np.array([[1, 2, 3], [2, 4, 8]])
    y = np.array([[2, 4, 6], [8, 4, 2]])
    expected = [np.corrcoef(x[index], y[index])[0, 1] for index in range(2)]
    np.testing.assert_allclose(kw.pearson(x, y, axis=1), expected)


def test_spearman_uses_average_ranks_for_ties():
    assert kw.spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1)
    assert kw.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


@pytest.mark.parametrize('function', [kw.pearson_correlation, kw.spearman_correlation])
def test_correlations_reject_constant_data(function):
    with pytest.raises(ValueError, match='constant'):
        function([1, 1, 1], [1, 2, 3])


def test_weighted_covariance_and_correlation():
    assert kw.covariance([1, 2, 4], [2, 3, 8], weights=[1, 2, 1], ddof=0) == pytest.approx(2.5)
    assert -1 <= kw.pearson([1, 2, 4], [2, 3, 8], weights=[1, 2, 1]) <= 1


def test_contingency_table_counts_and_proportions():
    table = kw.contingency_table(
        ['yes', 'yes', 'no', 'no'],
        ['A', 'B', 'A', 'A'],
    )
    assert table.to_dict() == {'yes': {'A': 1, 'B': 1}, 'no': {'A': 2, 'B': 0}}
    assert table.proportions.sum() == pytest.approx(1)


def test_contingency_tables_support_weights_axis_and_missing_policies():
    tables = kw.contingency_table(
        [['yes', 'yes'], ['no', 'yes']],
        [['A', 'B'], ['A', 'A']],
        axis=1,
        weights=[1, 2],
    )
    assert tables.shape == (2,)
    assert tables[0].to_dict() == {'yes': {'A': 1.0, 'B': 2.0}}
    assert tables[1].to_dict() == {'no': {'A': 1.0}, 'yes': {'A': 2.0}}
    assert kw.contingency_table(['yes', None], ['A', 'B']) is None
    omitted = kw.contingency_table(
        ['yes', None], ['A', 'B'], nan_policy='omit'
    )
    assert omitted.to_dict() == {'yes': {'A': 1}}


def test_nan_policy_raise_omit_and_propagate():
    data = [1, np.nan, 3]
    with pytest.raises(ValueError, match='missing'):
        kw.mean(data, nan_policy='raise')
    assert kw.mean(data, nan_policy='omit') == 2
    assert math.isnan(kw.mean(data, nan_policy='propagate'))


def test_nan_policy_applies_per_axis_slice():
    data = np.array([[1, np.nan], [2, 4]])
    np.testing.assert_allclose(kw.mean(data, axis=1, nan_policy='propagate'), [np.nan, 3],
                               equal_nan=True)
    np.testing.assert_allclose(kw.mean(data, axis=1, nan_policy='omit'), [1, 3])


def test_paired_omit_removes_rows_pairwise():
    assert kw.covariance([1, np.nan, 3], [2, 100, 6], ddof=0,
                         nan_policy='omit') == pytest.approx(2)


def test_frequency_propagation_is_explicit_none():
    assert kw.frequency_table(['a', np.nan], nan_policy='propagate') is None
    assert kw.proportion_table(['a', np.nan], nan_policy='propagate') is None


@pytest.mark.parametrize('function', [kw.mean, kw.median, kw.variance, kw.frequency_table])
def test_empty_data_raises_clear_error(function):
    with pytest.raises(ValueError, match='at least one observation'):
        function([])


def test_omit_that_removes_every_observation_raises():
    with pytest.raises(ValueError, match='no observations remain'):
        kw.mean([np.nan], nan_policy='omit')


@pytest.mark.parametrize('weights', [[1, -1], [0, 0], [1, math.inf]])
def test_invalid_weights_raise(weights):
    with pytest.raises(ValueError):
        kw.mean([1, 2], weights=weights)


def test_weight_shape_and_axis_validation():
    with pytest.raises(ValueError, match='broadcastable'):
        kw.mean([[1, 2], [3, 4]], axis=1, weights=[1, 2, 3])
    with pytest.raises(TypeError, match='axis'):
        kw.mean([1, 2], axis=True)


@pytest.mark.parametrize('nan_policy', ['drop', None, 1, []])
def test_invalid_nan_policy_raises(nan_policy):
    with pytest.raises(ValueError, match='nan_policy'):
        kw.mean([1, 2], nan_policy=nan_policy)


def test_invalid_ddof_and_too_large_ddof_raise():
    with pytest.raises(ValueError, match='ddof'):
        kw.variance([1, 2], ddof=-1)
    with pytest.raises(ValueError, match='smaller'):
        kw.variance([1, 2], ddof=2)


def test_invalid_quantile_percentile_trim_and_fence_inputs():
    with pytest.raises(ValueError, match='quantiles'):
        kw.quantile([1, 2], 1.1)
    with pytest.raises(ValueError, match='percentiles'):
        kw.percentile([1, 2], 101)
    with pytest.raises(ValueError, match='proportion'):
        kw.trimmed_mean([1, 2], 0.5)
    with pytest.raises(ValueError, match='factor'):
        kw.outlier_fences([1, 2], factor=-1)


def test_all_descriptive_exports_are_available_from_package():
    namespace = {}
    exec('from kiwicalc import *', namespace)
    for name in ('mean', 'describe', 'contingency_table', 'pearson', 'FrequencyTable'):
        assert name in namespace


def test_scalar_inputs_and_table_length():
    assert kw.mean(4) == 4
    assert kw.mode('green') == 'green'
    assert len(kw.frequency_table(['a', 'a', 'b'])) == 2


def test_quantile_missing_propagation_with_axis():
    assert math.isnan(kw.quantile([1, np.nan], 0.5))
    np.testing.assert_allclose(
        kw.quantile([[1, 2], [np.nan, 4]], [0.25, 0.75], axis=1),
        [[1.25, np.nan], [1.75, np.nan]],
        equal_nan=True,
    )


def test_weighted_advanced_location_and_shape_statistics():
    assert kw.geometric_mean([1, 4], weights=[1, 1]) == pytest.approx(2)
    assert kw.harmonic_mean([1, 2], weights=[1, 1]) == pytest.approx(4 / 3)
    assert kw.trimmed_mean([0, 1, 2, 100], 0.25, weights=[1, 1, 1, 1]) == 1.5
    assert math.isfinite(kw.skewness([1, 2, 4], weights=[1, 2, 1]))
    assert math.isfinite(kw.kurtosis([1, 2, 4], weights=[1, 2, 1]))


def test_proportion_tables_support_axis():
    proportions = kw.proportion_table([[1, 1], [1, 2]], axis=1)
    assert proportions[0] == {1: 1.0}
    assert proportions[1] == {1: 0.5, 2: 0.5}


def test_spearman_matrix_form():
    matrix = np.array([[1, 9], [2, 7], [3, 8]])
    np.testing.assert_allclose(kw.spearman_correlation(matrix), [[1, -0.5], [-0.5, 1]])


def test_mode_propagates_missing_values():
    assert math.isnan(kw.mode([1, np.nan]))


def test_paired_missing_and_validation_paths():
    assert math.isnan(kw.covariance([1, np.nan], [2, 3]))
    with pytest.raises(ValueError, match='missing'):
        kw.covariance([1, np.nan], [2, 3], nan_policy='raise')
    with pytest.raises(ValueError, match='paired observations'):
        kw.covariance([np.nan], [2], nan_policy='omit')
    with pytest.raises(ValueError, match='broadcastable'):
        kw.covariance([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match='finite'):
        kw.covariance([1, 2], [2, 3], weights=[1, np.inf])
    with pytest.raises(ValueError, match='negative'):
        kw.covariance([1, 2], [2, 3], weights=[1, -1])
    with pytest.raises(ValueError, match='ddof'):
        kw.covariance([1], [2], ddof=1)


def test_single_variable_relationship_forms():
    data = [1, 2, 4]
    assert kw.covariance(data) == pytest.approx(np.var(data, ddof=1))
    assert kw.covariance([[1, 2], [3, 4]], axis=None) == pytest.approx(
        np.var([1, 2, 3, 4], ddof=1)
    )
    assert kw.pearson_correlation(data) == pytest.approx(1)
    assert kw.spearman_correlation(data) == pytest.approx(1)


def test_contingency_validation_and_missing_rows():
    assert kw.contingency_table(['yes', None], ['A', 'B']) is None
    omitted = kw.contingency_table(['yes', None], ['A', 'B'], nan_policy='omit')
    assert omitted.to_dict() == {'yes': {'A': 1}}
    with pytest.raises(ValueError, match='broadcastable'):
        kw.contingency_table(['yes', 'no'], ['A', 'B', 'C'])


def test_negative_and_invalid_axes_and_input_types():
    np.testing.assert_allclose(kw.mean([[1, 2], [3, 4]], axis=-1), [1.5, 3.5])
    with pytest.raises(ValueError, match='out of bounds'):
        kw.mean([1, 2], axis=2)
    with pytest.raises(TypeError, match='numeric'):
        kw.mean(['not', 'numeric'])
    with pytest.raises(TypeError, match='weights'):
        kw.mean([1, 2], weights=['heavy', 'light'])


def test_more_invalid_ddof_and_fence_cases():
    with pytest.raises(TypeError, match='finite real'):
        kw.variance([1, 2], ddof=True)
    with pytest.raises(ValueError, match='smaller'):
        kw.skewness([1], ddof=1)
    with pytest.raises(ValueError, match='factor'):
        kw.outlier_fences([1, 2], factor=np.inf)
