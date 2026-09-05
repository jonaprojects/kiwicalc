# Probability theory

KiwiCalc provides small functions for probability laws and richer objects when a
problem has a finite sample space. The API uses ordinary Python values, so it is
appropriate for lessons, simulations, and engineering calculations.

## Probability laws

```python
import kiwicalc as kw

kw.complement_probability(0.3)
kw.addition_rule(0.6, 0.5, intersection=0.2)
kw.independent_intersection(0.5, 0.4)
kw.independent_union(0.5, 0.4)

joint = kw.joint_probability(conditional=0.4, given=0.5)
kw.conditional_probability(joint, given=0.5)
```

The names make assumptions visible: `independent_union()` and
`independent_intersection()` should only be used when the events are independent.
`addition_rule()` accepts the actual overlap and validates that it is possible.

The law of total probability and Bayes' theorem are similarly direct:

```python
evidence = kw.total_probability(
    conditionals=[0.95, 0.05],
    priors=[0.01, 0.99],
)

posterior = kw.bayes(
    prior=0.01,
    likelihood=0.95,
    evidence=evidence,
)

# Or let KiwiCalc calculate the evidence from alternative hypotheses.
posterior = kw.bayes(
    prior=0.01,
    likelihood=0.95,
    alternatives=[(0.99, 0.05)],
)
```

Priors in a total-probability partition must sum to one. Zero-probability
conditioning events and inconsistent intersections produce clear errors.

## Counting

```python
kw.permutations(8, 3)
kw.permutations(8, 3, repetition=True)
kw.combinations(8, 3)
kw.combinations(8, 3, repetition=True)
kw.multinomial(2, 3, 1)
```

Impossible selections without repetition return zero. Inputs must be non-negative
integers.

## Finite sample spaces and events

```python
die = kw.SampleSpace(range(1, 7))

even = die.event(lambda roll: roll % 2 == 0)
high = die.event([4, 5, 6])

even.probability
(even & high).probability
(even | high).probability
(~even).probability
even.conditional_probability(high)
even.is_independent(high)
```

Events are immutable subsets tied to their sample space. They support union (`|`),
intersection (`&`), difference (`-`), symmetric difference (`^`), and complement
(`~`). Combining events from different sample spaces is rejected.

Spaces are uniform by default. Weighted spaces accept either a mapping or aligned
probabilities:

```python
coin = kw.SampleSpace({"heads": 0.6, "tails": 0.4})
coin.probability("heads")

# Normalize relative weights explicitly when they do not already sum to one.
machine = kw.SampleSpace(["A", "B"], [2, 3], normalize=True)
```

## Discrete random variables

Assign a value to every outcome with a callable, mapping, or aligned sequence:

```python
roll = die.random_variable(lambda outcome: outcome, name="roll")
parity = kw.RandomVariable(die, lambda outcome: outcome % 2)

roll.distribution
roll.expectation
roll.variance
roll.standard_deviation
roll.pmf(3)
roll.cdf(3)
roll.probability(lambda value: value >= 5)

square = roll.transform(lambda value: value**2, name="square")
roll.covariance(square)
roll.correlation(square)
```

Equal values are aggregated automatically in the probability mass function.
Random-variable covariance uses the shared underlying outcome probabilities, so
both variables must belong to the same sample-space object. Correlation with a
constant random variable is undefined and raises `ValueError`.

For a distribution without an explicit sample space, use:

```python
kw.expected_value({0: 0.25, 2: 0.75})
kw.probability_variance({0: 0.25, 2: 0.75})
```

Distribution probabilities are validated and must sum to one.
