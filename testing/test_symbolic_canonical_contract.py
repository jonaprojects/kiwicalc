from dataclasses import FrozenInstanceError
from itertools import permutations
import random

import pytest

import kiwicalc as kw
from kiwicalc.serialization import object_from_dict, object_to_dict


def assert_canonical_shape(expression):
    assert kw.is_canonical_symbolic(expression)
    if isinstance(expression, kw.Add):
        assert len(expression.terms) >= 2
        assert not any(isinstance(term, kw.Add) for term in expression.terms)
        numeric = [index for index, term in enumerate(expression.terms) if isinstance(term, kw.ExactNumber)]
        assert numeric in ([], [0])
        assert not numeric or expression.terms[0] != kw.ExactNumber(0)
        assert list(map(str, expression.terms[1 if numeric else 0:])) == sorted(
            map(str, expression.terms[1 if numeric else 0:])
        )
        for term in expression.terms:
            assert_canonical_shape(term)
    elif isinstance(expression, kw.Multiply):
        assert len(expression.factors) >= 2
        assert not any(isinstance(factor, kw.Multiply) for factor in expression.factors)
        numeric = [index for index, factor in enumerate(expression.factors) if isinstance(factor, kw.ExactNumber)]
        assert not numeric or numeric == [0]
        assert expression.factors[0] != kw.ExactNumber(1)
        assert list(map(str, expression.factors[1 if numeric else 0:])) == sorted(
            map(str, expression.factors[1 if numeric else 0:])
        )
        for factor in expression.factors:
            assert_canonical_shape(factor)
    elif isinstance(expression, kw.Power):
        assert expression.exponent != kw.ExactNumber(1)
        assert_canonical_shape(expression.base)
        assert_canonical_shape(expression.exponent)
    elif isinstance(expression, kw.SymbolicFunction):
        for argument in expression.arguments:
            assert_canonical_shape(argument)


@pytest.mark.parametrize(
    ("source", "rendered"),
    (
        ("2+3", "5"),
        ("x+x+3*x", "5*x"),
        ("1*x", "x"),
        ("x^1", "x"),
        ("(2/3)+(1/6)", "5/6"),
        ("sqrt(16/9)", "4/3"),
        ("abs(-3)", "3"),
        ("sin(pi)", "0"),
        ("y+2+x+y+1", "3 + 2*y + x"),
    ),
)
def test_basic_canonical_examples(source, rendered):
    result = kw.simplify_symbolic(source)
    assert str(result) == rendered
    assert kw.is_canonical_symbolic(result)
    assert kw.simplify_symbolic(result) == result


def test_addition_and_multiplication_are_permutation_deterministic():
    terms = (kw.Symbol("z"), kw.ExactNumber(2), kw.Symbol("x"), kw.Symbol("z"))
    sums = {kw.simplify_symbolic(kw.Add(order)) for order in permutations(terms)}
    products = {kw.simplify_symbolic(kw.Multiply(order)) for order in permutations(terms)}
    assert len(sums) == 1
    assert len(products) == 1
    assert str(next(iter(sums))) == "2 + 2*z + x"
    assert str(next(iter(products))) == "2*x*z*z"


@pytest.mark.parametrize(
    "source",
    (
        "x + (y + 2) + x",
        "2*(x*3)*y",
        "(x+1)^2",
        "sqrt(x^2)+abs(y)",
        "ln(x)+ln(y)-ln(x)",
        "0*ln(x)",
        "sin(pi/4)+cos(pi/4)",
        "1/(x-1)+1/(x-1)",
    ),
)
def test_representative_forms_are_fixed_points(source):
    once = kw.simplify_symbolic(source)
    twice = kw.simplify_symbolic(once)
    assert once == twice
    assert hash(once) == hash(twice)
    assert_canonical_shape(once)


def test_canonicalization_is_exact_and_does_not_introduce_float_roundoff():
    result = kw.simplify_symbolic("0.1+0.2")
    assert isinstance(result, kw.ExactNumber)
    assert (result.numerator, result.denominator) == (3, 10)
    assert result.evaluate() == pytest.approx(0.3)


def test_canonicalization_preserves_values_on_the_original_domain():
    x = kw.Symbol("x")
    y = kw.Symbol("y")
    logarithm = kw.SymbolicFunction("ln", (x,))
    raw_expressions = (
        kw.Add((x, kw.Add((kw.ExactNumber(2), y)), x)),
        kw.Multiply((kw.ExactNumber(2), kw.Multiply((x, kw.ExactNumber(3))), y)),
        kw.Add((logarithm, kw.Multiply((kw.ExactNumber(-1), logarithm)))),
        kw.Power(kw.Power(x, kw.ExactNumber(-1)), kw.ExactNumber(0)),
    )
    values = {"x": 2, "y": -3}
    for raw in raw_expressions:
        canonical = kw.simplify_symbolic(raw)
        assert canonical.evaluate(values) == pytest.approx(raw.evaluate(values))


def test_domain_sensitive_cancellation_has_a_canonical_guarded_zero():
    logarithm = kw.simplify_symbolic("ln(x)-ln(x)")
    reciprocal = kw.simplify_symbolic("1/x-1/x")
    for expression in (logarithm, reciprocal):
        assert isinstance(expression, kw.Multiply)
        assert expression.factors[0] == kw.ExactNumber(0)
        assert not any(isinstance(factor, kw.Multiply) for factor in expression.factors)
        assert kw.simplify_symbolic(expression) == expression

    assert kw.solve_equation("ln(x)-ln(x)=0").conditions == ("x > 0",)
    assert kw.solve_equation("1/x-1/x=0").conditions == ("x != 0",)


def test_zero_power_preserves_the_domain_of_a_partial_base():
    guarded = kw.simplify_symbolic("(1/x)^0")
    ordinary = kw.simplify_symbolic("x^0")
    assert isinstance(guarded, kw.Power)
    assert kw.is_canonical_symbolic(guarded)
    assert ordinary == kw.ExactNumber(1)
    solution = kw.solve_equation("(1/x)^0=1")
    assert isinstance(solution.solution_set, kw.UniversalSolutionSet)
    assert solution.conditions == ("x != 0",)


def test_basic_simplification_explicitly_does_not_expand_or_factor():
    compact = kw.simplify_symbolic("(x+1)^2")
    expanded = kw.simplify_symbolic("x^2+2*x+1")
    factored = kw.simplify_symbolic("(x-1)*(x+1)")
    difference = kw.simplify_symbolic("x^2-1")
    assert compact != expanded
    assert factored != difference
    assert kw.is_canonical_symbolic(compact)
    assert kw.is_canonical_symbolic(expanded)


def test_canonicalization_does_not_mutate_raw_input_and_results_are_immutable():
    raw = kw.Add((kw.Symbol("y"), kw.ExactNumber(1), kw.Symbol("x")))
    before = repr(raw)
    canonical = kw.simplify_symbolic(raw)
    assert repr(raw) == before
    assert canonical is not raw
    with pytest.raises(FrozenInstanceError):
        canonical.terms = ()


def test_canonical_serialization_round_trip_is_stable():
    canonical = kw.simplify_symbolic("ln(x)-ln(x)+sqrt(y^2)")
    payload = object_to_dict(canonical)
    restored = object_from_dict(payload)
    assert restored == canonical
    assert object_to_dict(restored) == payload
    assert kw.is_canonical_symbolic(restored)


def test_structural_equality_uses_basic_canonical_form_only():
    assert kw.structurally_equal("x+x", "2*x")
    assert kw.structurally_equal("x+y", "y+x")
    assert not kw.structurally_equal("(x+1)^2", "x^2+2*x+1")


def test_random_raw_expression_trees_are_idempotent_and_shape_canonical():
    rng = random.Random(20260908)
    leaves = [kw.ExactNumber(value) for value in (-2, -1, 0, 1, 2, 3)] + [kw.Symbol("x"), kw.Symbol("y")]

    def expression(depth):
        if depth == 0 or rng.random() < 0.28:
            return rng.choice(leaves)
        kind = rng.randrange(5)
        if kind == 0:
            return kw.Add(tuple(expression(depth - 1) for _ in range(rng.randint(1, 4))))
        if kind == 1:
            return kw.Multiply(tuple(expression(depth - 1) for _ in range(rng.randint(1, 4))))
        if kind == 2:
            return kw.Power(expression(depth - 1), rng.choice(tuple(leaves[:5])))
        return kw.SymbolicFunction(
            rng.choice(("sqrt", "abs", "exp", "ln", "sin", "cos", "tan")),
            (expression(depth - 1),),
        )

    checked = 0
    for _ in range(3000):
        raw = expression(4)
        try:
            canonical = kw.simplify_symbolic(raw)
        except (ValueError, ZeroDivisionError):
            continue
        assert kw.simplify_symbolic(canonical) == canonical
        assert_canonical_shape(canonical)
        checked += 1
    assert checked >= 2500
