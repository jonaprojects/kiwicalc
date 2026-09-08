import random

import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("0", "zero"),
        ("-3", "negative"),
        ("2/3", "positive"),
        ("pi", "positive"),
        ("e", "positive"),
        ("i", "undefined"),
        ("x", "unknown"),
        ("x^2", "nonnegative"),
        ("x^-2", "positive"),
        ("abs(x)", "nonnegative"),
        ("exp(x)", "positive"),
    ),
)
def test_structural_sign_lattice_without_assumptions(expression, expected):
    assert kw.AssumptionSet().infer_sign(expression) == expected


@pytest.mark.parametrize(
    ("conditions", "expected"),
    (
        (("x > 0",), "positive"),
        (("x >= 0",), "nonnegative"),
        (("x < 0",), "negative"),
        (("x <= 0",), "nonpositive"),
        (("x != 0",), "nonzero"),
        (("x = 0",), "zero"),
        (("x >= 0", "x != 0"), "positive"),
        (("x <= 0", "x != 0"), "negative"),
        (("x > 4",), "positive"),
        (("x < -2",), "negative"),
    ),
)
def test_relational_assumptions_narrow_symbol_signs(conditions, expected):
    assert kw.AssumptionSet(conditions).infer_sign("x") == expected


def test_between_conditions_and_substitutions_participate_in_sign_inference():
    interval = kw.AssumptionSet((kw.parse_condition("0 <= x <= 2"),))
    positive_interval = kw.AssumptionSet((kw.parse_condition("0 < x <= 2"),))
    substituted = kw.AssumptionSet(("x = a", "a = -3"))
    assert interval.infer_sign("x") == "nonnegative"
    assert positive_interval.infer_sign("x") == "positive"
    assert substituted.infer_sign("x") == "negative"


@pytest.mark.parametrize(
    ("conditions", "expression", "expected"),
    (
        (("x > 0", "y < 0"), "x*y", "negative"),
        (("x < 0", "y < 0"), "x*y", "positive"),
        (("x != 0", "y != 0"), "x*y", "nonzero"),
        (("x > 0",), "-x", "negative"),
        (("x > 0", "y >= 0"), "x+y", "positive"),
        (("x >= 0", "y >= 0"), "x+y", "nonnegative"),
        (("x < 0", "y <= 0"), "x+y", "negative"),
        (("x < 0",), "1/x", "negative"),
        (("x < 0",), "x^3", "negative"),
        (("x != 0",), "x^4", "positive"),
        (("x > 0",), "sqrt(x)", "positive"),
        (("x != 0",), "abs(x)", "positive"),
    ),
)
def test_signs_propagate_through_expression_structure(conditions, expression, expected):
    assert kw.AssumptionSet(conditions).infer_sign(expression) == expected


@pytest.mark.parametrize(
    ("conditions", "expected"),
    (
        (("x > 1",), "positive"),
        (("x = 1",), "zero"),
        (("x > 0", "x < 1"), "negative"),
        (("ln(x) != 0",), "nonzero"),
    ),
)
def test_logarithm_sign_uses_argument_ranges_and_direct_evidence(conditions, expected):
    assert kw.AssumptionSet(conditions).infer_sign("ln(x)") == expected


def test_sign_inference_strengthens_entailment_and_refutation():
    assumptions = kw.AssumptionSet(("x > 0", "y < 0"))
    assert assumptions.entails(kw.parse_condition("x*y < 0"))
    assert assumptions.entails(kw.parse_condition("x^2 > 0"))
    assert assumptions.refutes(kw.parse_condition("x*y >= 0"))
    assert not assumptions.entails(kw.parse_condition("x+y > 0"))


def test_sign_is_on_the_expression_domain_but_entailment_also_requires_definedness():
    unconstrained = kw.AssumptionSet()
    nonzero = kw.AssumptionSet(("x != 0",))
    predicate = kw.parse_condition("x^-2 > 0")
    assert unconstrained.infer_sign("x^-2") == "positive"
    assert not unconstrained.entails(predicate)
    assert nonzero.entails(predicate)


def test_sign_inference_can_discharge_a_composite_rewrite_guard():
    rule = kw.RewriteRule(
        "positive-product",
        lambda expression, context: kw.Symbol("z") if expression == kw.Symbol("a") else None,
        lambda before, after, context: kw.parse_condition("x*y > 0"),
    )
    result = kw.rewrite_symbolic("a", rule, assumptions=("x > 0", "y > 0"))
    assert result.expression == kw.Symbol("z")
    assert result.applications[0].introduced.rendered == ()


@pytest.mark.parametrize(
    ("conditions", "expression", "defined", "domain"),
    (
        ((), "2", True, "real"),
        ((), "x", True, "real"),
        ((), "i", False, "complex"),
        ((), "sqrt(-1)", False, "complex"),
        ((), "ln(-1)", False, "complex"),
        ((), "ln(0)", False, "undefined"),
        (("x > 0",), "sqrt(x)", True, "real"),
        (("x < 0",), "sqrt(x)", False, "complex"),
        (("x > 0",), "ln(x)", True, "real"),
        (("x < 0",), "ln(x)", False, "complex"),
        (("x != 0",), "1/x", True, "real"),
        (("x = 0",), "1/x", False, "undefined"),
    ),
)
def test_definedness_and_domain_inference(conditions, expression, defined, domain):
    assumptions = kw.AssumptionSet(conditions)
    assert assumptions.is_defined(expression, "real") is defined
    assert assumptions.infer_domain(expression) == domain


def test_domain_inference_remains_unknown_when_a_guard_is_undecided():
    assumptions = kw.AssumptionSet()
    assert assumptions.is_defined("sqrt(x)", "real") is None
    assert assumptions.is_defined("ln(x)", "real") is None
    assert assumptions.is_defined("1/x", "real") is None
    assert assumptions.infer_domain("sqrt(x)") == "unknown"


def test_explicit_definedness_and_value_substitution_are_honored():
    explicitly_real = kw.AssumptionSet((kw.DefinedCondition(kw.parse_symbolic("sqrt(x)"), "real"),))
    substituted = kw.AssumptionSet(("x = -1",))
    assert explicitly_real.is_defined("sqrt(x)", "real") is True
    assert explicitly_real.infer_domain("sqrt(x)") == "real"
    assert substituted.is_defined("sqrt(x)", "real") is False
    assert substituted.infer_domain("sqrt(x)") == "complex"


def test_contradictory_assumptions_have_no_admissible_sign_or_domain():
    assumptions = kw.AssumptionSet(("x > 0", "x <= 0"))
    assert assumptions.infer_sign("x") == "undefined"
    assert assumptions.is_defined("x") is False
    assert assumptions.infer_domain("x") == "undefined"


def test_domain_argument_is_validated():
    with pytest.raises(ValueError):
        kw.AssumptionSet().is_defined("x", "integer")


def test_randomized_sign_inference_never_excludes_observed_valid_values():
    rng = random.Random(20260908)
    allowed = {
        "negative": {-1}, "zero": {0}, "positive": {1},
        "nonpositive": {-1, 0}, "nonnegative": {0, 1},
        "nonzero": {-1, 1}, "unknown": {-1, 0, 1},
    }
    expressions = (
        "x", "y", "x*y", "x+y", "-x*y", "x^2", "x^3",
        "x^2+y^2", "1/x", "abs(x)", "exp(x)", "sqrt(abs(x))",
    )
    for _ in range(500):
        x = rng.choice((-4, -2, -1, 1, 2, 5))
        y = rng.choice((-3, -1, 0, 1, 4))
        assumptions = kw.AssumptionSet((
            f"x {'>' if x > 0 else '<'} 0",
            f"y {'>' if y > 0 else '<' if y < 0 else '='} 0",
        ))
        expression = kw.parse_symbolic(rng.choice(expressions))
        inferred = assumptions.infer_sign(expression)
        observed = expression.evaluate({"x": x, "y": y})
        observed = observed.real if isinstance(observed, complex) else observed
        sign = 0 if observed == 0 else 1 if observed > 0 else -1
        assert inferred != "undefined"
        assert sign in allowed[inferred]
