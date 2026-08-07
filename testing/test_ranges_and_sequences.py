import pytest

import kiwicalc as kw
from kiwicalc.core.ranges import values_in_range


@pytest.mark.parametrize(
    ('source', 'assignments', 'expected'),
    [
        ('0<=x<5', {'x': 0}, True),
        ('0<=x<5', {'x': 5}, False),
        ('x>3', {'x': 4}, True),
        ('x>3', {'x': 3}, False),
        ('x<=3', {'x': 3}, True),
    ],
)
def test_range_parsing_and_evaluation(source, assignments, expected):
    assert kw.Range(source).evaluate_when(**assignments) is expected


def test_range_collections_apply_boolean_logic():
    x = kw.Var('x')
    positive = kw.Range(x.when(x=2), (0, None), (kw.LESS_THAN, None))
    below_one = kw.Range(x.when(x=2), (None, 1), (None, kw.LESS_THAN))

    assert kw.RangeOR((positive, below_one)).try_evaluate() is True
    assert kw.RangeAND((positive, below_one)).try_evaluate() is False


def test_values_in_range_filters_none_and_normalizes_booleans():
    values, results = values_in_range(
        lambda value: None if value < 0 else value > 0,
        -1,
        2,
        1,
    )
    assert values == [0, 1, 2]
    assert results == [0.0, 1.0, 1.0]


def test_arithmetic_progression_contract():
    sequence = kw.ArithmeticProg([3], 2)
    assert sequence.first == 3
    assert sequence.difference == 2
    assert sequence.in_index(4) == 9
    assert sequence.index_of(9) == 4
    assert sequence.sum_first_n(4) == 24
    assert sequence[1:4] == [3, 5, 7, 9]


def test_geometric_sequence_contract():
    sequence = kw.GeometricSeq([2], 3)
    assert sequence.first == 2
    assert sequence.ratio == 3
    assert sequence.in_index(4) == 54
    assert sequence.index_of(54) == 4
    assert sequence.sum_first_n(4) == 80


def test_constant_geometric_sequence_sum():
    assert kw.GeometricSeq([5], 1).sum_first_n(4) == 20


@pytest.mark.parametrize('sequence_type', [kw.ArithmeticProg, kw.GeometricSeq])
def test_sequences_reject_empty_initial_values(sequence_type):
    with pytest.raises(ValueError):
        sequence_type([])


def test_recursive_fibonacci_sequence_and_cache():
    sequence = kw.RecursiveSeq('a_n = a_{n-1} + a_{n-2}', (1, 1, 2))
    assert sequence.in_index(1) == 1
    assert sequence.in_index(6) == 8
    assert sequence.place_already_found(5)
    with pytest.raises(ValueError):
        sequence.in_index(0)
