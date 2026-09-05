import math

import numpy as np
import pytest

import kiwicalc as kw


def test_confidence_interval_result_is_friendly():
    interval = kw.ConfidenceInterval(1, 5, 0.95, 3, 1, "example")

    assert tuple(interval) == (1, 5)
    assert interval.as_tuple() == (1, 5)
    assert interval.margin_of_error == 2
    assert interval.width == 4
    assert interval.contains(3) is True
    assert interval.contains(7) is False
    assert np.array_equal(interval.contains([0, 2, 6]), [False, True, False])


def test_student_mean_confidence_interval_matches_reference_value():
    interval = kw.mean_confidence_interval([1, 2, 3, 4, 5])

    assert interval.method == "Student t confidence interval"
    assert interval.estimate == 3
    assert interval.standard_error == pytest.approx(math.sqrt(2.5 / 5))
    assert interval.lower == pytest.approx(1.0367568385)
    assert interval.upper == pytest.approx(4.9632431615)


def test_known_sigma_interval_and_alias_support_axis():
    data = [[10, 12, 14], [20, 22, 24]]
    interval = kw.confidence_interval(data, sigma=2, axis=1, confidence=0.9)

    assert interval.method == "Normal z confidence interval"
    assert np.allclose(interval.estimate, [12, 22])
    assert np.allclose(interval.standard_error, 2 / math.sqrt(3))
    assert np.all(interval.contains([12, 22]))


def test_mean_test_matches_reference_and_result_helpers():
    result = kw.mean_test([1, 2, 3, 4, 5])

    assert isinstance(result, kw.TestResult)
    assert result.method == "One-sample Student t test"
    assert result.statistic == pytest.approx(4.2426406871)
    assert result.p_value == pytest.approx(0.01323559956)
    assert result.pvalue == result.p_value
    assert result.degrees_of_freedom == 4
    assert result.significant()
    assert result.reject_null(0.02)
    assert result.as_dict()["confidence_interval"] is result.confidence_interval


@pytest.mark.parametrize(
    "alternative, expected_p, infinite_side",
    [("greater", 0.00661779978, "upper"), ("less", 0.9933822002, "lower")],
)
def test_directional_mean_tests_have_one_sided_intervals(alternative, expected_p,
                                                          infinite_side):
    result = kw.mean_test([1, 2, 3, 4, 5], alternative=alternative, confidence=0.95)

    assert result.p_value == pytest.approx(expected_p)
    bound = getattr(result.confidence_interval, infinite_side)
    assert math.isinf(bound)


def test_mean_z_test_and_array_nan_omission():
    z_result = kw.mean_test([9, 10, 11, 12], expected=10, sigma=2)
    array_result = kw.mean_test(
        [[1, 2, 3], [2, 3, np.nan]], expected=[2, 2], axis=1, nan_policy="omit"
    )

    assert z_result.method == "One-sample z test"
    assert z_result.statistic == pytest.approx(0.5)
    assert np.array_equal(array_result.degrees_of_freedom, [2, 1])
    assert np.allclose(array_result.estimate, [2, 2.5])


def test_constant_mean_tests_have_defined_limiting_results():
    equal = kw.mean_test([3, 3, 3], expected=3)
    different = kw.mean_test([3, 3, 3], expected=2)

    assert equal.statistic == 0 and equal.p_value == 1 and equal.effect_size == 0
    assert different.statistic == math.inf and different.p_value == 0
    assert different.effect_size == math.inf


def test_welch_pooled_and_paired_mean_comparisons():
    first, second = [1, 2, 3], [2, 3, 4]
    welch = kw.compare_means(first, second)
    pooled = kw.two_sample_t_test(first, second, equal_variance=True)
    paired = kw.compare_means(first, second, paired=True)

    assert welch.method == "Welch two-sample t test"
    assert welch.statistic == pytest.approx(-1.2247448714)
    assert welch.degrees_of_freedom == pytest.approx(4)
    assert pooled.method == "Independent pooled Student t test"
    assert paired.method == "Paired Student t test"
    assert paired.statistic == -math.inf
    assert paired.p_value == 0


def test_equal_constant_groups_do_not_create_nan_welch_result():
    result = kw.compare_means([2, 2, 2], [2, 2, 2])

    assert result.statistic == 0
    assert result.p_value == 1
    assert result.degrees_of_freedom == 4


def test_wilson_and_wald_proportion_intervals():
    wilson = kw.proportion_confidence_interval(50, 100)
    wald = kw.proportion_confidence_interval([5, 8], [10, 10], method="wald")

    assert wilson.lower == pytest.approx(0.4038315304)
    assert wilson.upper == pytest.approx(0.5961684696)
    assert wilson.method == "Wilson score interval"
    assert np.allclose(wald.estimate, [0.5, 0.8])
    assert np.all((wald.lower >= 0) & (wald.upper <= 1))


def test_one_and_two_sample_proportion_tests():
    one = kw.proportion_test(60, 100, expected=0.5)
    two = kw.compare_proportions(60, 100, 45, 100, alternative="greater")

    assert one.statistic == pytest.approx(2)
    assert one.p_value == pytest.approx(0.0455002639)
    assert one.confidence_interval.method == "Wilson score interval"
    assert two.estimate == pytest.approx(0.15)
    assert two.p_value < 0.02
    assert math.isinf(two.confidence_interval.upper)


def test_chi_square_goodness_of_fit_counts_probabilities_and_default():
    default = kw.chi_square_test([20, 30, 50])
    probabilities = kw.chi_square_test([20, 30, 50], expected=[0.2, 0.3, 0.5])

    assert default.statistic == pytest.approx(14)
    assert default.p_value == pytest.approx(math.exp(-7))
    assert default.degrees_of_freedom == 2
    assert np.allclose(default.details["expected"], [100 / 3] * 3)
    assert probabilities.statistic == pytest.approx(0)
    assert probabilities.p_value == pytest.approx(1)


def test_chi_square_independence_accepts_arrays_and_contingency_tables():
    array_result = kw.chi_square_independence([[10, 20], [30, 40]])
    table = kw.contingency_table(
        ["A"] * 30 + ["B"] * 70,
        ["yes"] * 10 + ["no"] * 20 + ["yes"] * 30 + ["no"] * 40,
    )
    table_result = kw.chi_square_independence(table, correction=True)

    assert array_result.statistic == pytest.approx(0.79365079365)
    assert array_result.p_value == pytest.approx(0.3729984836)
    assert table_result.statistic < array_result.statistic
    assert table_result.details["expected"].shape == (2, 2)


def test_one_way_anova_matches_reference_and_accepts_group_collection():
    groups = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    result = kw.one_way_anova(*groups)
    collected = kw.anova(groups)

    assert result.statistic == pytest.approx(27)
    assert result.p_value == pytest.approx(0.001)
    assert result.degrees_of_freedom == (2, 6)
    assert result.effect_size == pytest.approx(0.9)
    assert collected.statistic == result.statistic


def test_anova_nan_policies_and_zero_variance_cases():
    omitted = kw.anova([1, 2, np.nan], [3, 4, 5], nan_policy="omit")
    propagated = kw.anova([1, 2, np.nan], [3, 4, 5], nan_policy="propagate")
    equal = kw.anova([2, 2], [2, 2])
    different = kw.anova([1, 1], [2, 2])

    assert math.isfinite(omitted.statistic)
    assert math.isnan(propagated.statistic) and math.isnan(propagated.p_value)
    assert equal.statistic == 0 and equal.p_value == 1 and equal.effect_size == 0
    assert different.statistic == math.inf and different.p_value == 0


def test_pearson_and_spearman_correlation_tests():
    pearson = kw.correlation_test([1, 2, 3, 4], [1, 2, 4, 3])
    spearman = kw.correlation_test(
        [1, 2, 3, 4, 5], [1, 3, 2, 5, 4], method="SPEARMAN", alternative="upper"
    )

    assert pearson.estimate == pytest.approx(0.8)
    assert pearson.p_value == pytest.approx(0.2)
    assert pearson.degrees_of_freedom == 2
    assert spearman.method == "Spearman correlation t approximation"
    assert spearman.alternative == "greater"


def test_correlation_nan_handling():
    omitted = kw.correlation_test(
        [1, 2, 3, 4], [1, np.nan, 3, 5], nan_policy="omit"
    )
    propagated = kw.correlation_test(
        [1, 2, 3, 4], [1, np.nan, 3, 5], nan_policy="propagate"
    )

    assert omitted.estimate == pytest.approx(0.9819805061)
    assert math.isnan(propagated.estimate)
    assert math.isnan(propagated.p_value)


@pytest.mark.parametrize(
    "call, exception, message",
    [
        (lambda: kw.confidence_interval([], 0.95), ValueError, "cannot be empty"),
        (lambda: kw.confidence_interval([1], 0.95), ValueError, "at least 2"),
        (lambda: kw.confidence_interval([1, 2], confidence=1), ValueError, "strictly"),
        (lambda: kw.confidence_interval([1, 2], sigma=0), ValueError, "positive finite"),
        (lambda: kw.mean_test([1, 2], alternative="middle"), ValueError, "alternative"),
        (lambda: kw.mean_test([1, 2], alternative=2), TypeError, "text"),
        (lambda: kw.mean_test([1, 2], expected="x"), TypeError, "numeric"),
        (lambda: kw.mean_test([1, 2], expected=math.inf), ValueError, "finite"),
        (lambda: kw.mean_test([[1, 2]], axis=3), ValueError, "out of bounds"),
        (lambda: kw.mean_test([[1, 2]], axis=True), TypeError, "axis"),
        (lambda: kw.mean_test([1, np.nan], nan_policy="raise"), ValueError, "contains NaN"),
        (lambda: kw.mean_test([1, np.nan], nan_policy="omit"), ValueError, "non-missing"),
        (lambda: kw.mean_test([1, math.inf]), ValueError, "infinite"),
        (lambda: kw.compare_means([1, 2], [1, 2, 3], paired=True), ValueError, "same shape"),
        (lambda: kw.compare_means([1, 2], [2, 3], paired=1), TypeError, "Boolean"),
        (lambda: kw.proportion_confidence_interval(1.5, 2), TypeError, "integers"),
        (lambda: kw.proportion_confidence_interval(3, 2), ValueError, "0 <= successes"),
        (lambda: kw.proportion_confidence_interval(1, 2, method="exact"), ValueError, "wilson"),
        (lambda: kw.proportion_test(1, 2, expected=0), ValueError, "strictly"),
        (lambda: kw.chi_square_test([1]), ValueError, "at least two"),
        (lambda: kw.chi_square_test([1, math.nan]), ValueError, "non-negative vector"),
        (lambda: kw.chi_square_test([1, 2], [1, 1]), ValueError, "must total"),
        (lambda: kw.chi_square_test([1, 2], [1, 0]), ValueError, "positive"),
        (lambda: kw.chi_square_independence([[1, 2]]), ValueError, "at least two"),
        (lambda: kw.chi_square_independence([[1, 0], [0, 0]]), ValueError, "positive total"),
        (lambda: kw.chi_square_independence([[1, 2], [2, 1]], correction=1), TypeError, "Boolean"),
        (lambda: kw.anova([1, 2]), ValueError, "at least two groups"),
        (lambda: kw.anova([1, np.nan], [2, 3], nan_policy="raise"), ValueError, "contains NaN"),
        (lambda: kw.anova([1, np.nan], [2, 3], nan_policy="omit"), ValueError, "at least two"),
        (lambda: kw.anova([1, math.inf], [2, 3]), ValueError, "infinite"),
        (lambda: kw.correlation_test([1, 2], [1, 2]), ValueError, "at least three"),
        (lambda: kw.correlation_test([1, 2, 3], [1, 2]), ValueError, "same shape"),
        (lambda: kw.correlation_test([1, 2, 3], [1, 2, 3], method="kendall"), ValueError, "method"),
        (lambda: kw.correlation_test([1, 2, 3], [1, 2, 3], method=1), TypeError, "text"),
        (lambda: kw.correlation_test([1, 2, 3], [1, np.nan, 3], nan_policy="raise"), ValueError, "contain NaN"),
        (lambda: kw.correlation_test([1, 2, math.inf], [1, 2, 3]), ValueError, "infinite"),
    ],
)
def test_inference_validation_errors(call, exception, message):
    with pytest.raises(exception, match=message):
        call()


def test_invalid_alpha_and_nan_policy_are_rejected():
    result = kw.mean_test([1, 2, 3])
    with pytest.raises(ValueError, match="alpha"):
        result.significant(0)
    with pytest.raises(ValueError, match="nan_policy"):
        kw.mean_test([1, 2], nan_policy="ignore")
