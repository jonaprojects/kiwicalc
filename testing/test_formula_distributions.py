import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


def test_var_implements_lightweight_variable_contract():
    x = kw.Var('x')
    assert isinstance(x, kw.IVariable)
    assert x.name == 'x'
    assert (2 * x + 1).variables == {'x'}
    assert x.name == 'x'


def test_expression_and_string_formulas_are_equivalent():
    x = kw.Var('x')
    algebraic = kw.distribution(2 * x, between=(0, 1))
    explicit = kw.distribution('2*x', between=(0, 1))
    implicit = kw.distribution('2x', between=(0, 1))
    values = np.linspace(0, 1, 9)
    np.testing.assert_allclose(algebraic.pdf(values), explicit.pdf(values))
    np.testing.assert_allclose(algebraic.pdf(values), implicit.pdf(values))


def test_continuous_formula_distribution_reference_values():
    x = kw.Var('x')
    result = kw.distribution(2 * x, between=(0, 1), name='Triangle')
    assert isinstance(result, kw.ContinuousFormulaDistribution)
    assert isinstance(result, kw.ContinuousDistribution)
    assert result.support == (0.0, 1.0)
    assert result.variable == 'x'
    assert result.parameters == {}
    assert result.normalization_constant == pytest.approx(1)
    assert result.was_normalized
    assert result.pdf([-1, 0.5, 2]) == pytest.approx([0, 1, 0])
    assert result.cdf(0.5) == pytest.approx(0.25)
    assert result.ppf(0.25) == pytest.approx(0.5, abs=1e-8)
    assert result.quantile([0, 1]).tolist() == pytest.approx([0, 1])
    assert result.mean == pytest.approx(2 / 3)
    assert result.variance == pytest.approx(1 / 18)
    assert result.name == 'Triangle'
    assert 'Triangle' in repr(result)


def test_automatic_normalization_and_constant_density():
    x = kw.Var('x')
    result = kw.distribution(x, between=(0, 1))
    assert result.normalization_constant == pytest.approx(0.5)
    assert not result.was_normalized
    assert '/ 0.5' in result.normalized_formula
    assert result.pdf(0.5) == pytest.approx(1)

    uniform = kw.distribution(1, between=(-2, 2))
    assert uniform.variable == 'x'
    assert uniform.pdf(0) == pytest.approx(0.25)
    assert uniform.mean == pytest.approx(0)
    assert uniform.variance == pytest.approx(4 / 3)


def test_parameters_are_explicit_bound_and_defensively_copied():
    parameters = {'a': 2}
    result = kw.distribution(
        'a*x', variable='x', parameters=parameters, between=(0, 1),
    )
    parameters['a'] = 100
    returned = result.parameters
    returned['a'] = 50
    assert result.parameters == {'a': 2.0}
    assert result.pdf(0.5) == pytest.approx(1)


def test_source_expression_is_snapshotted():
    x = kw.Var('x')
    formula = 2 * x
    result = kw.distribution(formula, between=(0, 1))
    formula.coefficient = 100
    assert result.pdf(0.5) == pytest.approx(1)


def test_math_functions_constants_powers_and_implicit_multiplication():
    trigonometric = kw.distribution('sin(x)', between=(0, math.pi))
    assert trigonometric.normalization_constant == pytest.approx(2)
    assert trigonometric.pdf(math.pi / 2) == pytest.approx(0.5)
    polynomial = kw.distribution('2(x+1)', between=(0, 1))
    adjacent = kw.distribution('(x+1)(x+1)', between=(0, 1))
    assert polynomial.pdf(0.5) > 0
    assert adjacent.pdf(0.5) > 0


def test_documented_safe_math_function_vocabulary():
    formula = (
        'abs(x)+sqrt(x)+exp(x)+ln(x)+log(x)+log(x,10)+log10(x)+log2(x)'
        '+sin(x)+cos(x)+tan(x)+asin(x/2)+acos(x/2)+atan(x)'
        '+sinh(x)+cosh(x)+tanh(x)+asinh(x)+acosh(x)+atanh((x-1)/2)'
        '+pi+e+tau+x^2'
    )
    result = kw.distribution(formula, between=(1, 1.2))
    assert math.isfinite(result.normalization_constant)
    assert result.pdf([1, 1.1, 1.2]).shape == (3,)


def test_discrete_formula_distribution_reference_values():
    x = kw.Var('x')
    result = kw.distribution(x, over=range(1, 7), name='Weighted die')
    assert isinstance(result, kw.DiscreteFormulaDistribution)
    assert isinstance(result, kw.DiscreteDistribution)
    assert result.values == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert result.support == (1.0, 6.0)
    assert result.normalization_constant == pytest.approx(21)
    assert result.pmf(3) == pytest.approx(1 / 7)
    assert result.pmf([0, 1, 6, 7]) == pytest.approx([0, 1 / 21, 2 / 7, 0])
    assert result.cdf(3.5) == pytest.approx(6 / 21)
    assert result.ppf(result.cdf(3)) == 3
    assert result.mean == pytest.approx(13 / 3)
    assert result.probability_between(2, 4) == pytest.approx(9 / 21)
    assert result.probability_between(2, 4, inclusive='neither') == pytest.approx(3 / 21)
    np.testing.assert_array_equal(
        result.sample(12, random_state=123), result.rvs(12, random_state=123),
    )


def test_sparse_decimal_discrete_support_and_plotting():
    result = kw.distribution('x+1', over=[2.5, -0.5, 10], name='Sparse law')
    assert result.values == (-0.5, 2.5, 10.0)
    assert result.cdf(2.5) == pytest.approx(sum(result.probabilities[:2]))
    artist = result.plot(show=False)
    assert artist.patches
    axes = artist.patches[0].axes
    assert axes.get_title() == 'Sparse law PMF'
    plt.close(axes.figure)


@pytest.mark.parametrize('formula', [
    "__import__('os')",
    '(1).__class__',
    'lambda x: x',
    '[x for x in (1, 2)]',
    'open(x)',
    'sin x',
    'x[0]',
    'x < 2',
])
def test_string_formulas_reject_unsafe_or_unsupported_syntax(formula):
    with pytest.raises(ValueError):
        kw.distribution(formula, between=(0, 1))


@pytest.mark.parametrize('factory, error, message', [
    (lambda: kw.distribution('x'), ValueError, 'exactly one'),
    (lambda: kw.distribution('x', between=(0, 1), over=[1]), ValueError, 'exactly one'),
    (lambda: kw.distribution('x+y', between=(0, 1)), ValueError, 'multiple unbound'),
    (lambda: kw.distribution('a*x', variable='x', between=(0, 1)), ValueError, 'unresolved'),
    (lambda: kw.distribution('x', parameters={'a': 2}, between=(0, 1)), ValueError, 'unknown'),
    (lambda: kw.distribution('-x', between=(0, 1)), ValueError, 'nonnegative'),
    (lambda: kw.distribution('1/(x-0.5)', between=(0, 1)), ValueError, 'finite'),
    (lambda: kw.distribution(0, between=(0, 1)), ValueError, 'positive finite integral'),
    (lambda: kw.distribution('x', between=(1, 1)), ValueError, 'smaller'),
    (lambda: kw.distribution('x', over=[]), ValueError, 'empty'),
    (lambda: kw.distribution('x', over=[1, 1]), ValueError, 'distinct'),
    (lambda: kw.distribution('-x', over=[1, 2]), ValueError, 'nonnegative'),
])
def test_formula_distribution_validation(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


def test_array_shapes_nan_and_sampling_are_consistent():
    continuous = kw.distribution('2*x', between=(0, 1))
    values = continuous.cdf(np.array([[0.0, 0.5], [1.0, np.nan]]))
    assert values.shape == (2, 2)
    assert math.isnan(values[1, 1])
    assert continuous.sample((2, 3), random_state=4).shape == (2, 3)

    discrete = kw.distribution('x', over=[1, 2, 3])
    probabilities = discrete.pmf(np.array([[1, 2], [4, np.nan]]))
    assert probabilities.shape == (2, 2)
    assert math.isnan(probabilities[1, 1])
    assert discrete.sample((2, 3), random_state=4).shape == (2, 3)


def test_formula_distribution_exports():
    for name in (
        'IVariable', 'distribution', 'ContinuousFormulaDistribution',
        'DiscreteFormulaDistribution',
    ):
        assert hasattr(kw, name)
