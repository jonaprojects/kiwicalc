"""Core probability theory for finite experiments and events."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np


def _probability(value, name='probability'):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and between 0 and 1')
    return value


def _tolerance(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError('tolerance must be a real number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('tolerance must be finite and non-negative')
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def complement_probability(probability):
    """Return ``P(not A)``."""
    return 1 - _probability(probability)


def addition_rule(first, second, intersection=0):
    """Return ``P(A or B) = P(A) + P(B) - P(A and B)``."""
    first = _probability(first, 'first probability')
    second = _probability(second, 'second probability')
    intersection = _probability(intersection, 'intersection probability')
    lower = max(0.0, first + second - 1)
    upper = min(first, second)
    if not lower <= intersection <= upper:
        raise ValueError('intersection is inconsistent with the event probabilities')
    return first + second - intersection


def independent_intersection(*probabilities):
    """Return the intersection probability assuming mutual independence."""
    if not probabilities:
        raise ValueError('provide at least one probability')
    result = 1.0
    for index, probability in enumerate(probabilities):
        result *= _probability(probability, f'probability {index}')
    return result


def independent_union(*probabilities):
    """Return the union probability assuming mutual independence."""
    if not probabilities:
        raise ValueError('provide at least one probability')
    complement = 1.0
    for index, probability in enumerate(probabilities):
        complement *= 1 - _probability(probability, f'probability {index}')
    return 1 - complement


def conditional_probability(intersection, given):
    """Return ``P(A | B)`` from ``P(A and B)`` and ``P(B)``."""
    intersection = _probability(intersection, 'intersection probability')
    given = _probability(given, 'given probability')
    if given == 0:
        raise ValueError('conditional probability is undefined when P(given) is zero')
    if intersection > given:
        raise ValueError('intersection probability cannot exceed P(given)')
    return intersection / given


def joint_probability(conditional, given):
    """Return ``P(A and B) = P(A | B) P(B)``."""
    return _probability(conditional, 'conditional probability') * _probability(
        given, 'given probability'
    )


def total_probability(conditionals, priors, tolerance=1e-12):
    """Apply the law of total probability to a finite partition."""
    tolerance = _tolerance(tolerance)
    conditional_values = tuple(conditionals)
    prior_values = tuple(priors)
    if not conditional_values:
        raise ValueError('conditionals and priors cannot be empty')
    if len(conditional_values) != len(prior_values):
        raise ValueError('conditionals and priors must have the same length')
    conditional_values = tuple(
        _probability(value, f'conditional {index}')
        for index, value in enumerate(conditional_values)
    )
    prior_values = tuple(
        _probability(value, f'prior {index}')
        for index, value in enumerate(prior_values)
    )
    if not math.isclose(sum(prior_values), 1.0, abs_tol=tolerance, rel_tol=0):
        raise ValueError('priors must sum to one')
    return float(sum(conditional * prior for conditional, prior in zip(
        conditional_values, prior_values
    )))


law_of_total_probability = total_probability


def bayes_theorem(prior, likelihood, evidence=None, *, alternatives=None,
                  tolerance=1e-12):
    """Return a posterior probability using Bayes' theorem.

    Supply ``evidence`` directly, or omit it and provide ``alternatives`` as
    ``(prior, likelihood)`` pairs.  In the latter form the target hypothesis is
    included automatically in the evidence total.
    """
    prior = _probability(prior, 'prior')
    likelihood = _probability(likelihood, 'likelihood')
    if evidence is not None and alternatives is not None:
        raise ValueError('provide evidence or alternatives, not both')
    if evidence is None:
        if alternatives is None:
            raise ValueError('provide evidence or alternative hypotheses')
        terms = [(prior, likelihood)]
        try:
            terms.extend(tuple(pair) for pair in alternatives)
        except (TypeError, ValueError) as exc:
            raise TypeError('alternatives must contain (prior, likelihood) pairs') from exc
        if any(len(pair) != 2 for pair in terms):
            raise ValueError('each alternative must be a (prior, likelihood) pair')
        priors = [_probability(pair[0], f'prior {index}') for index, pair in enumerate(terms)]
        likelihoods = [
            _probability(pair[1], f'likelihood {index}') for index, pair in enumerate(terms)
        ]
        evidence = total_probability(likelihoods, priors, tolerance=tolerance)
    else:
        evidence = _probability(evidence, 'evidence')
    if evidence == 0:
        raise ValueError('posterior is undefined when evidence is zero')
    numerator = prior * likelihood
    if numerator > evidence and not math.isclose(numerator, evidence, abs_tol=tolerance):
        raise ValueError('evidence cannot be smaller than prior times likelihood')
    return min(1.0, numerator / evidence)


bayes = bayes_theorem


def are_independent(first, second, intersection, tolerance=1e-12):
    """Test whether ``P(A and B)`` equals ``P(A)P(B)``."""
    first = _probability(first, 'first probability')
    second = _probability(second, 'second probability')
    intersection = _probability(intersection, 'intersection probability')
    tolerance = _tolerance(tolerance)
    addition_rule(first, second, intersection)
    return math.isclose(intersection, first * second, abs_tol=tolerance, rel_tol=tolerance)


def are_mutually_exclusive(intersection, tolerance=1e-12):
    """Test whether an intersection probability is zero."""
    return math.isclose(
        _probability(intersection, 'intersection probability'), 0.0,
        abs_tol=_tolerance(tolerance), rel_tol=0,
    )


def odds(probability, *, against=False):
    """Return odds for an event, or odds against it when requested.

    Odds are returned as a ``(numerator, denominator)`` tuple. Infinite odds use
    ``math.inf`` as the numerator and ``1.0`` as the denominator.
    """
    if not isinstance(against, bool):
        raise TypeError('against must be a Boolean')
    probability = _probability(probability)
    numerator = 1 - probability if against else probability
    denominator = probability if against else 1 - probability
    if denominator == 0:
        return math.inf, 1.0
    return numerator, denominator


def probability_from_odds(numerator, denominator=1, *, against=False):
    """Convert non-negative odds to probability."""
    if not isinstance(against, bool):
        raise TypeError('against must be a Boolean')
    for value, name in ((numerator, 'numerator'), (denominator, 'denominator')):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{name} must be a real number')
        if math.isnan(float(value)) or value < 0:
            raise ValueError(f'{name} must be non-negative')
    numerator, denominator = float(numerator), float(denominator)
    if math.isinf(numerator) and math.isinf(denominator):
        raise ValueError('odds cannot have two infinite terms')
    if numerator == 0 and denominator == 0:
        raise ValueError('odds cannot be 0:0')
    if math.isinf(numerator):
        probability = 1.0
    elif math.isinf(denominator):
        probability = 0.0
    else:
        probability = numerator / (numerator + denominator)
    return 1 - probability if against else probability


def permutations(n, r=None, *, repetition=False):
    """Count ordered selections of ``r`` items from ``n`` choices."""
    if not isinstance(repetition, bool):
        raise TypeError('repetition must be a Boolean')
    n = _integer(n, 'n')
    r = n if r is None else _integer(r, 'r')
    if repetition:
        if n == 0 and r == 0:
            return 1
        if n == 0:
            return 0
        return n ** r
    if r > n:
        return 0
    return math.factorial(n) // math.factorial(n - r)


def combinations(n, r, *, repetition=False):
    """Count unordered selections of ``r`` items from ``n`` choices."""
    if not isinstance(repetition, bool):
        raise TypeError('repetition must be a Boolean')
    n = _integer(n, 'n')
    r = _integer(r, 'r')
    if repetition:
        if n == 0:
            return 1 if r == 0 else 0
        return math.comb(n + r - 1, r)
    if r > n:
        return 0
    return math.comb(n, r)


def multinomial(*counts):
    """Count arrangements with repeated groups of the supplied sizes."""
    if len(counts) == 1 and not isinstance(counts[0], Integral):
        counts = tuple(counts[0])
    if not counts:
        raise ValueError('provide at least one group size')
    counts = tuple(_integer(value, f'count {index}') for index, value in enumerate(counts))
    result = math.factorial(sum(counts))
    for count in counts:
        result //= math.factorial(count)
    return result


class SampleSpace:
    """A finite sample space with uniform or explicitly weighted outcomes."""

    def __init__(self, outcomes, probabilities=None, *, normalize=False,
                 tolerance=1e-12):
        tolerance = _tolerance(tolerance)
        if not isinstance(normalize, bool):
            raise TypeError('normalize must be a Boolean')
        if isinstance(outcomes, Mapping):
            if probabilities is not None:
                raise ValueError('do not pass probabilities when outcomes is a mapping')
            outcome_values = tuple(outcomes.keys())
            probability_values = tuple(outcomes.values())
        else:
            if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Iterable):
                raise TypeError('outcomes must be a finite iterable or mapping')
            outcome_values = tuple(outcomes)
            probability_values = probabilities
        if not outcome_values:
            raise ValueError('a sample space needs at least one outcome')
        try:
            unique_count = len(set(outcome_values))
        except TypeError as exc:
            raise TypeError('sample-space outcomes must be hashable') from exc
        if unique_count != len(outcome_values):
            raise ValueError('sample-space outcomes must be unique')
        if probability_values is None:
            chance = 1.0 / len(outcome_values)
            probability_values = (chance,) * len(outcome_values)
        elif isinstance(probability_values, Mapping):
            if set(probability_values) != set(outcome_values):
                raise ValueError('probability mapping must contain every outcome exactly once')
            probability_values = tuple(probability_values[value] for value in outcome_values)
        else:
            probability_values = tuple(probability_values)
        if len(probability_values) != len(outcome_values):
            raise ValueError('outcomes and probabilities must have the same length')
        checked = []
        for index, value in enumerate(probability_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'probability {index} must be a real number')
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'probability {index} must be finite and non-negative')
            checked.append(value)
        total = sum(checked)
        if total <= 0:
            raise ValueError('probabilities must have a positive total')
        if normalize:
            checked = [value / total for value in checked]
        elif not math.isclose(total, 1.0, abs_tol=tolerance, rel_tol=0):
            raise ValueError('sample-space probabilities must sum to one')
        self._outcomes = outcome_values
        self._probabilities = MappingProxyType(dict(zip(outcome_values, checked)))
        self._outcome_set = frozenset(outcome_values)

    @property
    def outcomes(self):
        return self._outcomes

    @property
    def probabilities(self):
        return self._probabilities

    def __len__(self):
        return len(self._outcomes)

    def __iter__(self):
        return iter(self._outcomes)

    def __contains__(self, outcome):
        try:
            return outcome in self._outcome_set
        except TypeError:
            return False

    def __repr__(self):
        return f'SampleSpace({dict(self._probabilities)!r})'

    def event(self, outcomes):
        """Create an event from outcomes or a predicate over outcomes."""
        if isinstance(outcomes, Event):
            if outcomes.sample_space is not self:
                raise ValueError('event belongs to a different sample space')
            return outcomes
        if callable(outcomes):
            chosen = frozenset(value for value in self if outcomes(value))
        else:
            try:
                is_single_outcome = outcomes in self._outcome_set
            except TypeError:
                is_single_outcome = False
            if is_single_outcome:
                chosen = frozenset((outcomes,))
            elif isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Iterable):
                raise ValueError(f'unknown outcome {outcomes!r}')
            else:
                try:
                    chosen = frozenset(outcomes)
                except TypeError as exc:
                    raise TypeError('event outcomes must be hashable') from exc
        unknown = chosen - self._outcome_set
        if unknown:
            raise ValueError(f'event contains outcomes outside the sample space: {unknown!r}')
        return Event(self, chosen)

    def probability(self, event):
        event = self.event(event)
        return float(sum(self._probabilities[outcome] for outcome in event))

    probability_of = probability

    def conditional_probability(self, event, given):
        event, given = self.event(event), self.event(given)
        return conditional_probability(self.probability(event & given), given.probability)

    def random_variable(self, values, name=None):
        return DiscreteRandomVariable(self, values, name=name)


class Event:
    """An immutable subset of a :class:`SampleSpace`."""

    def __init__(self, sample_space, outcomes):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError('sample_space must be a SampleSpace')
        self._sample_space = sample_space
        self._outcomes = frozenset(outcomes)
        if not self._outcomes <= sample_space._outcome_set:
            raise ValueError('event contains outcomes outside the sample space')

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def outcomes(self):
        return self._outcomes

    @property
    def probability(self):
        return self._sample_space.probability(self)

    @property
    def complement(self):
        return Event(self._sample_space, self._sample_space._outcome_set - self._outcomes)

    def __len__(self):
        return len(self._outcomes)

    def __iter__(self):
        return (outcome for outcome in self._sample_space if outcome in self._outcomes)

    def __contains__(self, outcome):
        return outcome in self._outcomes

    def __repr__(self):
        return f'Event({list(self)!r}, probability={self.probability!r})'

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.sample_space is other.sample_space and self.outcomes == other.outcomes

    def __hash__(self):
        return hash((id(self.sample_space), self.outcomes))

    def _other(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        if other.sample_space is not self.sample_space:
            raise ValueError('events belong to different sample spaces')
        return other

    def __or__(self, other):
        other = self._other(other)
        if other is NotImplemented:
            return NotImplemented
        return Event(self.sample_space, self.outcomes | other.outcomes)

    def __and__(self, other):
        other = self._other(other)
        if other is NotImplemented:
            return NotImplemented
        return Event(self.sample_space, self.outcomes & other.outcomes)

    def __sub__(self, other):
        other = self._other(other)
        if other is NotImplemented:
            return NotImplemented
        return Event(self.sample_space, self.outcomes - other.outcomes)

    def __xor__(self, other):
        other = self._other(other)
        if other is NotImplemented:
            return NotImplemented
        return Event(self.sample_space, self.outcomes ^ other.outcomes)

    def __invert__(self):
        return self.complement

    def is_independent(self, other, tolerance=1e-12):
        other = self._other(other)
        if other is NotImplemented:
            raise TypeError('other must be an Event')
        return are_independent(
            self.probability, other.probability, (self & other).probability, tolerance
        )

    def is_mutually_exclusive(self, other, tolerance=1e-12):
        other = self._other(other)
        if other is NotImplemented:
            raise TypeError('other must be an Event')
        return are_mutually_exclusive((self & other).probability, tolerance)

    def conditional_probability(self, given):
        given = self._other(given)
        if given is NotImplemented:
            raise TypeError('given must be an Event')
        return self.sample_space.conditional_probability(self, given)


class DiscreteRandomVariable:
    """A numeric value assigned to every outcome in a finite sample space."""

    def __init__(self, sample_space, values, name=None):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError('sample_space must be a SampleSpace')
        if name is not None and not isinstance(name, str):
            raise TypeError('name must be text or None')
        if callable(values):
            mapping = {outcome: values(outcome) for outcome in sample_space}
        elif isinstance(values, Mapping):
            if set(values) != sample_space._outcome_set:
                raise ValueError('values must contain every sample-space outcome exactly once')
            mapping = {outcome: values[outcome] for outcome in sample_space}
        else:
            try:
                sequence = tuple(values)
            except TypeError as exc:
                raise TypeError('values must be a callable, mapping, or iterable') from exc
            if len(sequence) != len(sample_space):
                raise ValueError('values must match the number of sample-space outcomes')
            mapping = dict(zip(sample_space, sequence))
        try:
            for value in mapping.values():
                hash(value)
        except TypeError as exc:
            raise TypeError('random-variable values must be hashable') from exc
        self._sample_space = sample_space
        self._values = MappingProxyType(mapping)
        self._name = name

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values

    @property
    def distribution(self):
        distribution = OrderedDict()
        for outcome in self.sample_space:
            value = self._values[outcome]
            distribution[value] = distribution.get(value, 0.0) + self.sample_space.probabilities[outcome]
        return MappingProxyType(dict(distribution))

    @property
    def support(self):
        return tuple(self.distribution)

    @property
    def expectation(self):
        return expected_value(self.distribution)

    mean = expectation

    @property
    def variance(self):
        return probability_variance(self.distribution)

    @property
    def standard_deviation(self):
        return math.sqrt(self.variance)

    std = standard_deviation

    def pmf(self, value):
        return float(self.distribution.get(value, 0.0))

    def cdf(self, value):
        try:
            return float(sum(probability for candidate, probability in self.distribution.items()
                             if candidate <= value))
        except TypeError as exc:
            raise TypeError('cdf requires ordered, comparable values') from exc

    def probability(self, values):
        if callable(values):
            return float(sum(probability for value, probability in self.distribution.items()
                             if values(value)))
        try:
            is_single_value = values in self.distribution
        except TypeError:
            is_single_value = False
        if is_single_value:
            return self.pmf(values)
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            return 0.0
        selected = set(values)
        return float(sum(self.pmf(value) for value in selected))

    def event(self, predicate):
        if not callable(predicate):
            try:
                is_single_value = predicate in self.distribution
            except TypeError:
                is_single_value = False
            if is_single_value or not isinstance(predicate, Iterable):
                selected = {predicate}
            else:
                selected = set(predicate)
            predicate = lambda value: value in selected
        return self.sample_space.event(lambda outcome: predicate(self._values[outcome]))

    def moment(self, order, *, central=False):
        order = _integer(order, 'order')
        if not isinstance(central, bool):
            raise TypeError('central must be a Boolean')
        center = self.expectation if central else 0.0
        return expected_value(
            {value: probability for value, probability in self.distribution.items()},
            transform=lambda value: (value - center) ** order,
        )

    def transform(self, function, name=None):
        if not callable(function):
            raise TypeError('function must be callable')
        return DiscreteRandomVariable(
            self.sample_space,
            {outcome: function(value) for outcome, value in self.values.items()},
            name=name,
        )

    def covariance(self, other):
        other = self._compatible(other)
        expected_product = sum(
            self.sample_space.probabilities[outcome]
            * self.values[outcome] * other.values[outcome]
            for outcome in self.sample_space
        )
        return float(expected_product - self.expectation * other.expectation)

    def correlation(self, other):
        other = self._compatible(other)
        denominator = self.standard_deviation * other.standard_deviation
        if math.isclose(denominator, 0.0, abs_tol=1e-15, rel_tol=0):
            raise ValueError('correlation is undefined for a constant random variable')
        return self.covariance(other) / denominator

    def _compatible(self, other):
        if not isinstance(other, DiscreteRandomVariable):
            raise TypeError('other must be a DiscreteRandomVariable')
        if other.sample_space is not self.sample_space:
            raise ValueError('random variables must use the same sample space')
        return other

    def __repr__(self):
        label = '' if self.name is None else f', name={self.name!r}'
        return f'DiscreteRandomVariable({dict(self.values)!r}{label})'


RandomVariable = DiscreteRandomVariable


def _distribution(values, probabilities=None, tolerance=1e-12):
    tolerance = _tolerance(tolerance)
    if isinstance(values, DiscreteRandomVariable):
        if probabilities is not None:
            raise ValueError('do not pass probabilities with a random variable')
        return values.distribution
    if isinstance(values, Mapping):
        if probabilities is not None:
            raise ValueError('do not pass probabilities when values is a mapping')
        value_sequence = tuple(values.keys())
        probability_sequence = tuple(values.values())
    else:
        try:
            value_sequence = tuple(values)
        except TypeError as exc:
            raise TypeError('values must be a distribution mapping or iterable') from exc
        if probabilities is None:
            probability_sequence = (1 / len(value_sequence),) * len(value_sequence) if value_sequence else ()
        else:
            probability_sequence = tuple(probabilities)
    if not value_sequence:
        raise ValueError('distribution cannot be empty')
    if len(value_sequence) != len(probability_sequence):
        raise ValueError('values and probabilities must have the same length')
    checked = tuple(_probability(value, f'probability {index}')
                    for index, value in enumerate(probability_sequence))
    if not math.isclose(sum(checked), 1.0, abs_tol=tolerance, rel_tol=0):
        raise ValueError('probabilities must sum to one')
    distribution = OrderedDict()
    for value, probability in zip(value_sequence, checked):
        try:
            distribution[value] = distribution.get(value, 0.0) + probability
        except TypeError as exc:
            raise TypeError('distribution values must be hashable') from exc
    return dict(distribution)


def expected_value(values, probabilities=None, *, transform=None, tolerance=1e-12):
    """Return the expectation of a finite discrete distribution."""
    distribution = _distribution(values, probabilities, tolerance)
    function = (lambda value: value) if transform is None else transform
    if not callable(function):
        raise TypeError('transform must be callable or None')
    try:
        return float(sum(function(value) * probability
                         for value, probability in distribution.items()))
    except (TypeError, ValueError) as exc:
        raise TypeError('expected value requires numeric transformed values') from exc


expectation = expected_value


def probability_variance(values, probabilities=None, *, tolerance=1e-12):
    """Return the population variance of a finite discrete distribution."""
    distribution = _distribution(values, probabilities, tolerance)
    center = expected_value(distribution)
    result = expected_value(
        distribution, transform=lambda value: (value - center) ** 2
    )
    return 0.0 if math.isclose(result, 0.0, abs_tol=tolerance, rel_tol=0) else result


__all__ = [
    'complement_probability', 'addition_rule', 'independent_intersection',
    'independent_union', 'conditional_probability', 'joint_probability',
    'total_probability', 'law_of_total_probability', 'bayes_theorem', 'bayes',
    'are_independent', 'are_mutually_exclusive', 'odds', 'probability_from_odds',
    'permutations', 'combinations', 'multinomial', 'SampleSpace', 'Event',
    'DiscreteRandomVariable', 'RandomVariable', 'expected_value', 'expectation',
    'probability_variance',
]
