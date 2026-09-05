import math

import numpy as np
import pytest

import kiwicalc as kw


def test_probability_laws():
    assert kw.complement_probability(0.3) == pytest.approx(0.7)
    assert kw.addition_rule(0.6, 0.5, intersection=0.2) == pytest.approx(0.9)
    assert kw.independent_intersection(0.5, 0.4, 0.25) == pytest.approx(0.05)
    assert kw.independent_union(0.5, 0.4) == pytest.approx(0.7)


def test_conditional_and_joint_probability_are_inverse_operations():
    joint = kw.joint_probability(0.4, 0.5)
    assert joint == pytest.approx(0.2)
    assert kw.conditional_probability(joint, 0.5) == pytest.approx(0.4)


def test_total_probability_and_alias():
    expected = 0.9 * 0.2 + 0.1 * 0.8
    assert kw.total_probability([0.9, 0.1], [0.2, 0.8]) == pytest.approx(expected)
    assert kw.law_of_total_probability([0.9, 0.1], [0.2, 0.8]) == pytest.approx(expected)


def test_bayes_with_direct_evidence():
    assert kw.bayes_theorem(0.2, 0.9, 0.26) == pytest.approx(9 / 13)


def test_bayes_computes_evidence_from_alternatives():
    posterior = kw.bayes(
        prior=0.01,
        likelihood=0.95,
        alternatives=[(0.99, 0.05)],
    )
    assert posterior == pytest.approx(0.0095 / 0.059)


def test_independence_and_exclusivity_helpers():
    assert kw.are_independent(0.5, 0.4, 0.2)
    assert not kw.are_independent(0.5, 0.4, 0.1)
    assert kw.are_mutually_exclusive(0)
    assert not kw.are_mutually_exclusive(0.01)
    with pytest.raises(ValueError, match='inconsistent'):
        kw.are_independent(0.1, 0.1, 0.5)


def test_odds_round_trip():
    assert kw.odds(0.75) == pytest.approx((0.75, 0.25))
    assert kw.odds(0.75, against=True) == pytest.approx((0.25, 0.75))
    assert kw.probability_from_odds(3, 1) == pytest.approx(0.75)
    assert kw.probability_from_odds(1, 3, against=True) == pytest.approx(0.75)
    assert kw.odds(1) == (math.inf, 1.0)
    assert kw.probability_from_odds(math.inf) == 1
    assert kw.probability_from_odds(1, math.inf) == 0


def test_counting_methods():
    assert kw.permutations(5, 2) == 20
    assert kw.permutations(5) == math.factorial(5)
    assert kw.permutations(3, 4) == 0
    assert kw.permutations(3, 4, repetition=True) == 81
    assert kw.combinations(5, 2) == 10
    assert kw.combinations(2, 3) == 0
    assert kw.combinations(3, 2, repetition=True) == 6
    assert kw.multinomial(2, 2, 1) == 30
    assert kw.multinomial([2, 2, 1]) == 30


def test_zero_choice_counting_edges():
    assert kw.permutations(0, 0, repetition=True) == 1
    assert kw.permutations(0, 2, repetition=True) == 0
    assert kw.combinations(0, 0, repetition=True) == 1
    assert kw.combinations(0, 2, repetition=True) == 0


@pytest.fixture
def die():
    return kw.SampleSpace(range(1, 7))


def test_uniform_sample_space(die):
    assert len(die) == 6
    assert tuple(die) == (1, 2, 3, 4, 5, 6)
    assert 3 in die
    assert die.probabilities[1] == pytest.approx(1 / 6)
    assert die.probability([1, 2]) == pytest.approx(1 / 3)
    assert die.probability_of(lambda outcome: outcome % 2 == 0) == pytest.approx(0.5)


def test_weighted_sample_space_mapping_and_normalization():
    coin = kw.SampleSpace({'heads': 0.6, 'tails': 0.4})
    assert coin.probability('heads') == 0.6
    normalized = kw.SampleSpace(['a', 'b'], [2, 3], normalize=True)
    assert normalized.probability('a') == pytest.approx(0.4)


def test_probability_mapping_can_be_supplied_separately():
    space = kw.SampleSpace(['a', 'b'], {'b': 0.7, 'a': 0.3})
    assert space.probability('a') == 0.3


def test_events_support_set_algebra(die):
    even = die.event(lambda value: value % 2 == 0)
    high = die.event([4, 5, 6])
    assert list(even) == [2, 4, 6]
    assert even.probability == pytest.approx(0.5)
    assert (even & high).outcomes == frozenset({4, 6})
    assert (even | high).outcomes == frozenset({2, 4, 5, 6})
    assert (even - high).outcomes == frozenset({2})
    assert (even ^ high).outcomes == frozenset({2, 5})
    assert (~even).outcomes == frozenset({1, 3, 5})
    assert even.complement == ~even


def test_event_conditional_probability(die):
    even = die.event([2, 4, 6])
    high = die.event([4, 5, 6])
    assert even.conditional_probability(high) == pytest.approx(2 / 3)
    assert die.conditional_probability(even, high) == pytest.approx(2 / 3)


def test_event_relationships(die):
    even = die.event([2, 4, 6])
    low = die.event([1, 2])
    one = die.event(1)
    six = die.event(6)
    assert even.is_independent(low)
    assert one.is_mutually_exclusive(six)
    assert not one.is_independent(six)


def test_event_identity_and_repr(die):
    event = die.event([1, 2])
    assert die.event(event) is event
    assert 1 in event
    assert len(event) == 2
    assert 'probability=' in repr(event)
    assert 'SampleSpace' in repr(die)
    assert [1] not in die


def test_cross_space_events_are_rejected():
    first = kw.SampleSpace([1, 2]).event(1)
    second = kw.SampleSpace([1, 2]).event(1)
    with pytest.raises(ValueError, match='different sample spaces'):
        _ = first | second
    with pytest.raises(ValueError, match='different sample space'):
        first.sample_space.event(second)


def test_empty_event_conditioning_is_undefined(die):
    with pytest.raises(ValueError, match='zero'):
        die.conditional_probability(die.event(1), die.event([]))


def test_discrete_random_variable_for_die(die):
    variable = die.random_variable(lambda outcome: outcome, name='roll')
    assert isinstance(variable, kw.RandomVariable)
    assert variable.name == 'roll'
    assert variable.support == (1, 2, 3, 4, 5, 6)
    assert variable.expectation == pytest.approx(3.5)
    assert variable.mean == pytest.approx(3.5)
    assert variable.variance == pytest.approx(35 / 12)
    assert variable.standard_deviation == pytest.approx(math.sqrt(35 / 12))
    assert variable.std == variable.standard_deviation
    assert variable.pmf(2) == pytest.approx(1 / 6)
    assert variable.pmf(9) == 0
    assert variable.cdf(3) == pytest.approx(0.5)


def test_random_variable_aggregates_equal_values(die):
    parity = kw.RandomVariable(die, lambda outcome: outcome % 2)
    assert parity.distribution == {1: 0.5, 0: 0.5}
    assert parity.probability(1) == pytest.approx(0.5)
    assert parity.probability([0, 1]) == 1
    assert parity.probability(lambda value: value == 0) == pytest.approx(0.5)
    assert parity.event([1]).outcomes == frozenset({1, 3, 5})


def test_string_random_variable_value_is_treated_as_one_value():
    space = kw.SampleSpace(['a', 'b'])
    label = kw.RandomVariable(space, {'a': 'yes', 'b': 'no'})
    assert label.event('yes').outcomes == frozenset({'a'})
    with pytest.raises(TypeError, match='comparable'):
        label.cdf(1)


def test_random_variable_sequence_and_mapping_construction(die):
    sequence = kw.RandomVariable(die, [1, 2, 3, 4, 5, 6])
    mapping = kw.RandomVariable(die, {value: value for value in die})
    assert sequence.distribution == mapping.distribution


def test_random_variable_moments_and_transformation(die):
    roll = kw.RandomVariable(die, lambda outcome: outcome)
    assert roll.moment(1) == pytest.approx(3.5)
    assert roll.moment(2, central=True) == pytest.approx(roll.variance)
    square = roll.transform(lambda value: value ** 2, name='square')
    assert square.expectation == pytest.approx(91 / 6)
    assert square.name == 'square'


def test_random_variable_covariance_and_correlation(die):
    roll = kw.RandomVariable(die, lambda outcome: outcome)
    reverse = kw.RandomVariable(die, lambda outcome: 7 - outcome)
    assert roll.covariance(reverse) == pytest.approx(-roll.variance)
    assert roll.correlation(reverse) == pytest.approx(-1)


def test_expected_value_and_probability_variance():
    distribution = {0: 0.25, 2: 0.75}
    assert kw.expected_value(distribution) == pytest.approx(1.5)
    assert kw.expectation([0, 2], [0.25, 0.75]) == pytest.approx(1.5)
    assert kw.probability_variance(distribution) == pytest.approx(0.75)
    assert kw.expected_value([1, 1, 3]) == pytest.approx(5 / 3)
    assert kw.expected_value(distribution, transform=lambda value: value ** 2) == 3


@pytest.mark.parametrize('value', [-0.1, 1.1, math.inf, math.nan, True, '0.2'])
def test_probability_laws_validate_probabilities(value):
    error = TypeError if value is True or isinstance(value, str) else ValueError
    with pytest.raises(error):
        kw.complement_probability(value)


def test_probability_law_invalid_relationships():
    with pytest.raises(ValueError, match='inconsistent'):
        kw.addition_rule(0.2, 0.3, intersection=0.4)
    with pytest.raises(ValueError, match='exceed'):
        kw.conditional_probability(0.4, 0.3)
    with pytest.raises(ValueError, match='zero'):
        kw.conditional_probability(0, 0)
    with pytest.raises(ValueError, match='at least one'):
        kw.independent_union()
    with pytest.raises(ValueError, match='at least one'):
        kw.independent_intersection()


def test_total_probability_validation():
    with pytest.raises(ValueError, match='empty'):
        kw.total_probability([], [])
    with pytest.raises(ValueError, match='same length'):
        kw.total_probability([0.2], [0.4, 0.6])
    with pytest.raises(ValueError, match='sum to one'):
        kw.total_probability([0.2, 0.3], [0.4, 0.4])


def test_bayes_validation():
    with pytest.raises(ValueError, match='evidence or alternative'):
        kw.bayes(0.5, 0.5)
    with pytest.raises(ValueError, match='not both'):
        kw.bayes(0.5, 0.5, 0.5, alternatives=[(0.5, 0.5)])
    with pytest.raises(ValueError, match='pair'):
        kw.bayes(0.5, 0.5, alternatives=[(0.5,)])
    with pytest.raises(ValueError, match='zero'):
        kw.bayes(0, 0, evidence=0)
    with pytest.raises(ValueError, match='smaller'):
        kw.bayes(0.8, 0.8, evidence=0.5)
    with pytest.raises(TypeError, match='pairs'):
        kw.bayes(0.5, 0.5, alternatives=1)


def test_odds_validation():
    with pytest.raises(ValueError, match='0:0'):
        kw.probability_from_odds(0, 0)
    with pytest.raises(ValueError, match='infinite'):
        kw.probability_from_odds(math.inf, math.inf)
    with pytest.raises(ValueError, match='non-negative'):
        kw.probability_from_odds(-1, 2)
    with pytest.raises(TypeError, match='real'):
        kw.probability_from_odds('3', 2)


def test_counting_validation():
    with pytest.raises(TypeError, match='integer'):
        kw.permutations(3.5, 2)
    with pytest.raises(ValueError, match='at least'):
        kw.combinations(-1, 2)
    with pytest.raises(ValueError, match='at least one'):
        kw.multinomial()
    with pytest.raises(TypeError, match='Boolean'):
        kw.permutations(3, 2, repetition=1)
    with pytest.raises(TypeError, match='Boolean'):
        kw.combinations(3, 2, repetition='yes')


def test_sample_space_validation():
    with pytest.raises(ValueError, match='at least one'):
        kw.SampleSpace([])
    with pytest.raises(ValueError, match='unique'):
        kw.SampleSpace([1, 1])
    with pytest.raises(TypeError, match='hashable'):
        kw.SampleSpace([[1], [2]])
    with pytest.raises(ValueError, match='sum to one'):
        kw.SampleSpace(['a', 'b'], [0.2, 0.2])
    with pytest.raises(ValueError, match='same length'):
        kw.SampleSpace(['a', 'b'], [1])
    with pytest.raises(ValueError, match='outside'):
        kw.SampleSpace([1, 2]).event([1, 3])
    with pytest.raises(TypeError, match='Boolean'):
        kw.SampleSpace([1, 2], normalize=1)
    with pytest.raises(ValueError, match='do not pass'):
        kw.SampleSpace({'a': 1}, [1])
    with pytest.raises(TypeError, match='finite iterable'):
        kw.SampleSpace('abc')
    with pytest.raises(ValueError, match='every outcome'):
        kw.SampleSpace(['a', 'b'], {'a': 1})
    with pytest.raises(TypeError, match='real number'):
        kw.SampleSpace(['a'], ['certain'])
    with pytest.raises(ValueError, match='non-negative'):
        kw.SampleSpace(['a', 'b'], [-1, 2])
    with pytest.raises(ValueError, match='positive total'):
        kw.SampleSpace(['a', 'b'], [0, 0])
    with pytest.raises(ValueError, match='unknown outcome'):
        kw.SampleSpace([1, 2]).event(3)
    with pytest.raises(TypeError, match='hashable'):
        kw.SampleSpace([1, 2]).event([[1]])


def test_random_variable_validation(die):
    with pytest.raises(ValueError, match='every'):
        kw.RandomVariable(die, {1: 1})
    with pytest.raises(ValueError, match='number'):
        kw.RandomVariable(die, [1, 2])
    with pytest.raises(TypeError, match='hashable'):
        kw.RandomVariable(die, lambda value: [value])
    with pytest.raises(TypeError, match='callable'):
        kw.RandomVariable(die, range(6)).transform(1)
    with pytest.raises(TypeError, match='iterable'):
        kw.RandomVariable(die, 1)
    with pytest.raises(TypeError, match='Boolean'):
        kw.RandomVariable(die, range(6)).moment(2, central=1)
    with pytest.raises(TypeError, match='SampleSpace'):
        kw.RandomVariable([1, 2], [1, 2])
    with pytest.raises(TypeError, match='name'):
        kw.RandomVariable(die, range(6), name=1)


def test_random_variable_relationship_validation(die):
    variable = kw.RandomVariable(die, lambda outcome: outcome)
    foreign = kw.RandomVariable(kw.SampleSpace([1, 2]), [1, 2])
    with pytest.raises(ValueError, match='same sample space'):
        variable.covariance(foreign)
    with pytest.raises(TypeError, match='RandomVariable'):
        variable.correlation([1, 2])
    constant = kw.RandomVariable(die, [1] * 6)
    with pytest.raises(ValueError, match='constant'):
        constant.correlation(variable)


def test_distribution_validation():
    with pytest.raises(ValueError, match='empty'):
        kw.expected_value([])
    with pytest.raises(ValueError, match='same length'):
        kw.expected_value([1, 2], [1])
    with pytest.raises(ValueError, match='sum to one'):
        kw.expected_value([1, 2], [0.2, 0.2])
    with pytest.raises(TypeError, match='numeric'):
        kw.expected_value(['a', 'b'])
    with pytest.raises(TypeError, match='callable'):
        kw.expected_value([1, 2], transform=3)
    with pytest.raises(TypeError, match='mapping or iterable'):
        kw.expected_value(1)
    variable = kw.RandomVariable(kw.SampleSpace([1, 2]), [1, 2])
    assert kw.expected_value(variable) == pytest.approx(1.5)
    with pytest.raises(ValueError, match='random variable'):
        kw.expected_value(variable, [0.5, 0.5])
    with pytest.raises(ValueError, match='mapping'):
        kw.expected_value({1: 1}, [1])
    with pytest.raises(TypeError, match='hashable'):
        kw.expected_value([[1], [2]])


def test_event_and_random_variable_type_edges(die):
    event = die.event(1)
    assert event != 1
    assert hash(event)
    for operation in (
        lambda: event | 1,
        lambda: event & 1,
        lambda: event - 1,
        lambda: event ^ 1,
    ):
        with pytest.raises(TypeError):
            operation()
    with pytest.raises(TypeError, match='Event'):
        event.is_independent(1)
    with pytest.raises(TypeError, match='Event'):
        event.is_mutually_exclusive(1)
    with pytest.raises(TypeError, match='Event'):
        event.conditional_probability(1)
    with pytest.raises(TypeError, match='SampleSpace'):
        kw.Event([1, 2], [1])
    with pytest.raises(ValueError, match='outside'):
        kw.Event(die, [9])

    variable = kw.RandomVariable(die, range(6))
    assert variable.probability(99) == 0
    assert 'DiscreteRandomVariable' in repr(variable)


def test_tolerance_validation():
    with pytest.raises(TypeError, match='tolerance'):
        kw.are_mutually_exclusive(0, tolerance=True)
    with pytest.raises(ValueError, match='non-negative'):
        kw.are_independent(0.5, 0.5, 0.25, tolerance=-1)


def test_boolean_option_validation():
    with pytest.raises(TypeError, match='Boolean'):
        kw.odds(0.5, against=1)
    with pytest.raises(TypeError, match='Boolean'):
        kw.probability_from_odds(1, 2, against='no')


def test_probability_theory_exports():
    for name in (
        'SampleSpace', 'Event', 'RandomVariable', 'bayes', 'combinations',
        'expected_value',
    ):
        assert hasattr(kw, name)
