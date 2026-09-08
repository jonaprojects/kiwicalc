import inspect
import json

import pytest

import kiwicalc as kw
from kiwicalc.serialization import object_from_dict, object_to_dict


def test_normalization_public_signatures_are_stable():
    assert tuple(inspect.signature(kw.normalize_polynomial_symbolic).parameters) == (
        "value", "variable", "max_degree",
    )
    assert tuple(inspect.signature(kw.normalize_rational_symbolic).parameters) == (
        "value", "variable", "assumptions", "domain", "max_degree",
    )


@pytest.mark.parametrize(
    "source, expected",
    (
        ("(x+1)^3", "1+3*x+3*x^2+x^3"),
        ("(x-1)*(x+1)", "-1+x^2"),
        ("x/2+1/3", "1/3+x/2"),
        ("0*(x+1)^4", "0"),
        ("7", "7"),
    ),
)
def test_polynomial_normalization_is_exact_and_expanded(source, expected):
    assert kw.normalize_polynomial_symbolic(source) == kw.parse_symbolic(expected)


def test_polynomial_normalization_is_idempotent_and_deterministic():
    sources = ("(x+2)*(x-3)", "(x-3)*(2+x)", "x*(x-1)-6")
    normalized = [kw.normalize_polynomial_symbolic(source) for source in sources]
    assert normalized[0] == normalized[1] == normalized[2]
    assert kw.normalize_polynomial_symbolic(normalized[0]) == normalized[0]


@pytest.mark.parametrize(
    "source, expected, required",
    (
        ("(x^2-1)/(x-1)", "x+1", "x - 1 != 0"),
        ("(x^2-1)/(x^2-x)", "(x+1)/x", "x - 1 != 0"),
        ("(2*x+2)/(4*x+4)", "1/2", "x + 1 != 0"),
        ("(x+1)/(-2*x-2)", "-1/2", "x + 1 != 0"),
        ("0/(x-1)", "0", "x - 1 != 0"),
    ),
)
def test_guarded_cancellation_retains_the_removed_factor(source, expected, required):
    result = kw.normalize_rational_symbolic(source)
    assert result.expression == kw.parse_symbolic(expected)
    assert required in result.conditions
    assert result.applications[0].rule == "normalize-rational"
    assert required in result.applications[0].required.rendered


def test_fraction_sum_uses_a_primitive_common_denominator():
    result = kw.normalize_rational_symbolic("1/x+1/(x+1)")
    assert result.expression == kw.parse_symbolic("(2*x+1)/(x^2+x)")
    assert set(result.conditions) >= {"x != 0", "x + 1 != 0", "x^2 + x != 0"}
    assert result.assumptions.is_defined(result.expression)


def test_partial_cancellation_retains_original_and_remaining_denominator_domains():
    result = kw.normalize_rational_symbolic("(x^2-1)/(x^2-x)")
    assert result.expression == kw.parse_symbolic("(x+1)/x")
    assert set(result.conditions) >= {"x^2 - x != 0", "x - 1 != 0", "x != 0"}
    assert result.assumptions.is_defined(result.expression)


def test_no_cancellation_still_normalizes_content_and_denominator_sign():
    positive = kw.normalize_rational_symbolic("(2*x+1)/(-4*x+2)")
    assert positive.expression == kw.parse_symbolic("(-2*x-1)/(4*x-2)")
    assert positive.conditions


def test_rational_normalization_is_idempotent_without_losing_assumptions():
    first = kw.normalize_rational_symbolic("(x^2-1)/(x^2-x)")
    second = kw.normalize_rational_symbolic(first.expression, assumptions=first.assumptions)
    assert second.expression == first.expression
    assert set(second.conditions) == set(first.conditions)
    assert second.applications == ()


def test_result_and_audit_trail_serialize_structurally():
    result = kw.normalize_rational_symbolic("(x^2-1)/(x-1)")
    payload = object_to_dict(result)
    assert object_from_dict(payload) == result
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_rational_normalization_rewrite_is_explicit_and_non_default():
    source = "1/x+1/(x+1)"
    assert kw.rewrite_symbolic(source).expression == kw.parse_symbolic(source)
    explicit = kw.rewrite_symbolic(source, rules="normalize-rational")
    assert explicit.expression == kw.parse_symbolic("(2*x+1)/(x^2+x)")
    assert set(explicit.conditions) == {"x != 0", "x + 1 != 0"}


def test_polynomial_nonzero_assumption_entails_each_exact_polynomial_factor():
    assumptions = kw.AssumptionSet(("x^2-x != 0",))
    assert assumptions.entails(kw.parse_condition("x != 0")) is True
    assert assumptions.entails(kw.parse_condition("x-1 != 0")) is True
    assert kw.AssumptionSet(("x != 0",)).entails(
        kw.parse_condition("x^2-x != 0")
    ) is False
    assert assumptions.entails(kw.parse_condition("x+1 != 0")) is False


@pytest.mark.parametrize("common", ("x-1", "x+2", "x^2+1"))
@pytest.mark.parametrize("numerator", ("x+1", "2*x-3", "x^2+x+1"))
@pytest.mark.parametrize("denominator", ("x+3", "x^2+2"))
def test_generated_cancellations_preserve_values_away_from_holes(
    common, numerator, denominator,
):
    source = kw.parse_symbolic(f"(({common})*({numerator}))/(({common})*({denominator}))")
    result = kw.normalize_rational_symbolic(source)
    for point in (-5, -2, -1, 0, 1, 2, 4):
        try:
            original_value = source.evaluate({"x": point})
        except ZeroDivisionError:
            continue
        normalized_value = result.expression.evaluate({"x": point})
        assert complex(normalized_value) == pytest.approx(complex(original_value))


@pytest.mark.parametrize(
    "call, error",
    (
        (lambda: kw.normalize_polynomial_symbolic("x+y"), kw.AmbiguousVariableError),
        (lambda: kw.normalize_polynomial_symbolic("x+y", variable="x"),
         kw.UnsupportedExpressionError),
        (lambda: kw.normalize_polynomial_symbolic("sin(x)"),
         kw.UnsupportedExpressionError),
        (lambda: kw.normalize_polynomial_symbolic("1/x"),
         kw.UnsupportedExpressionError),
        (lambda: kw.normalize_rational_symbolic("x+y"), kw.AmbiguousVariableError),
        (lambda: kw.normalize_rational_symbolic("x+y", variable="x"),
         kw.UnsupportedExpressionError),
        (lambda: kw.normalize_rational_symbolic("sin(x)"),
         kw.UnsupportedExpressionError),
        (lambda: kw.normalize_rational_symbolic("x", domain="integer"), ValueError),
        (lambda: kw.normalize_polynomial_symbolic("x", variable="not valid"), ValueError),
        (lambda: kw.normalize_polynomial_symbolic("x", max_degree=True), ValueError),
        (lambda: kw.normalize_rational_symbolic("x", max_degree=-1), ValueError),
        (lambda: kw.normalize_polynomial_symbolic("x^4", max_degree=3),
         kw.UnsupportedExpressionError),
    ),
)
def test_normalization_rejects_ambiguous_unsupported_or_unbounded_input(call, error):
    with pytest.raises(error):
        call()


def test_symbol_objects_and_explicit_single_variable_names_are_supported():
    assert kw.normalize_polynomial_symbolic("(theta+1)^2", kw.Symbol("theta")) == (
        kw.parse_symbolic("theta^2+2*theta+1")
    )
    assert kw.normalize_rational_symbolic("(theta^2-1)/(theta-1)", "theta").expression == (
        kw.parse_symbolic("theta+1")
    )


def test_existing_assumptions_and_complex_domain_are_preserved():
    result = kw.normalize_rational_symbolic(
        "(x^2-1)/(x-1)", assumptions="x > 2", domain="complex",
    )
    assert "x > 2" in result.conditions
    assert "x - 1 != 0" in result.conditions
