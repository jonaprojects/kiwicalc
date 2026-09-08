from dataclasses import FrozenInstanceError
import inspect
import json
import math

import pytest

import kiwicalc as kw
from kiwicalc.serialization import object_from_dict, object_to_dict


def test_rewrite_api_contract_and_builtin_registry_are_stable():
    assert tuple(inspect.signature(kw.rewrite_symbolic).parameters) == (
        "value", "rules", "assumptions", "domain", "strategy",
        "introduce_conditions", "max_steps", "max_nodes",
    )
    assert kw.available_rewrite_rules() == (
        "cancel-reciprocal", "exp-log-inverse", "sqrt-square",
        "normalize-rational",
    )


def test_cancel_reciprocal_preserves_the_original_nonzero_domain():
    result = kw.rewrite_symbolic("x/x")
    assert result.expression == kw.ExactNumber(1)
    assert result.conditions == ("x != 0",)
    assert result.status == "fixed_point" and result.converged
    assert len(result.applications) == 1
    application = result.applications[0]
    assert application.rule == "cancel-reciprocal"
    assert application.conditions == ("x != 0",)
    assert not application.introduced


def test_nested_rewrite_records_the_local_tree_path():
    result = kw.rewrite_symbolic("1+x/x", rules="cancel-reciprocal")
    assert result.expression == kw.ExactNumber(2)
    assert result.applications[0].path
    assert result.applications[0].before == kw.simplify_symbolic("x/x")
    assert result.applications[0].after == kw.ExactNumber(1)


def test_cancellation_retains_every_domain_condition_of_a_partial_factor():
    result = kw.rewrite_symbolic("ln(x)/ln(x)")
    assert result.expression == kw.ExactNumber(1)
    assert set(result.conditions) == {"x > 0", "ln(x) != 0"}
    assert result.assumptions.evaluate({"x": math.e ** 2}) is True
    assert result.assumptions.evaluate({"x": 1}) is False


@pytest.mark.parametrize("domain, condition", (("real", "x > 0"), ("complex", "x != 0")))
def test_exp_log_inverse_uses_domain_specific_guards(domain, condition):
    result = kw.rewrite_symbolic("exp(ln(x))", domain=domain)
    assert result.expression == kw.Symbol("x")
    assert result.conditions == (condition,)
    assert result.applications[0].rule == "exp-log-inverse"


def test_real_sqrt_square_rewrites_to_abs_without_a_spurious_condition():
    real = kw.rewrite_symbolic("sqrt(x^2)")
    complex_result = kw.rewrite_symbolic("sqrt(x^2)", domain="complex")
    assert real.expression == kw.parse_symbolic("abs(x)")
    assert real.conditions == ()
    assert complex_result.expression == kw.parse_symbolic("sqrt(x^2)")
    assert complex_result.applications == ()


def test_square_root_rule_does_not_treat_the_imaginary_unit_as_real():
    result = kw.rewrite_symbolic("sqrt(i^2)", domain="real")
    assert result.expression == kw.parse_symbolic("sqrt(i^2)")
    assert result.applications == ()


def _positive_sqrt_rule():
    def transform(expression, context):
        if (
            isinstance(expression, kw.SymbolicFunction)
            and expression.name == "sqrt"
            and isinstance(expression.arguments[0], kw.Power)
            and expression.arguments[0].exponent == kw.ExactNumber(2)
        ):
            return expression.arguments[0].base
        return None

    def guard(expression, replacement, context):
        return kw.RelationCondition(replacement, ">=", kw.ExactNumber(0))

    return kw.RewriteRule(
        "positive-square-root", transform, guard,
        "Use sqrt(x^2)=x only on the nonnegative branch.",
    )


def test_unknown_guard_is_skipped_by_default():
    result = kw.rewrite_symbolic("sqrt(x^2)", _positive_sqrt_rule())
    assert result.expression == kw.parse_symbolic("sqrt(x^2)")
    assert result.applications == ()
    assert result.conditions == ()


def test_unknown_guard_can_be_explicitly_introduced_and_audited():
    result = kw.rewrite_symbolic(
        "sqrt(x^2)", _positive_sqrt_rule(), introduce_conditions=True,
    )
    assert result.expression == kw.Symbol("x")
    assert result.conditions == ("x >= 0",)
    assert result.applications[0].required.rendered == ("x >= 0",)
    assert result.applications[0].introduced.rendered == ("x >= 0",)


def test_entailed_guard_applies_without_being_marked_as_new():
    result = kw.rewrite_symbolic(
        "sqrt(x^2)", _positive_sqrt_rule(), assumptions="x >= 0",
    )
    assert result.expression == kw.Symbol("x")
    assert result.conditions == ("x >= 0",)
    assert not result.applications[0].introduced


def test_refuted_guard_is_never_applied_even_when_introduction_is_enabled():
    result = kw.rewrite_symbolic(
        "sqrt(x^2)", _positive_sqrt_rule(), assumptions="x < 0",
        introduce_conditions=True,
    )
    assert result.expression == kw.parse_symbolic("sqrt(x^2)")
    assert result.conditions == ("x < 0",)
    assert result.applications == ()


def test_candidate_domain_restrictions_are_automatically_guarded():
    reciprocal = kw.RewriteRule(
        "introduce-reciprocal",
        lambda expression, context: kw.parse_symbolic("1/x") if expression == kw.Symbol("a") else None,
    )
    skipped = kw.rewrite_symbolic("a", reciprocal)
    conditional = kw.rewrite_symbolic("a", reciprocal, introduce_conditions=True)
    assert skipped.expression == kw.Symbol("a")
    assert skipped.applications == ()
    assert conditional.expression == kw.parse_symbolic("1/x")
    assert conditional.conditions == ("x != 0",)
    assert conditional.applications[0].introduced.rendered == ("x != 0",)


def test_rule_priority_and_traversal_are_deterministic():
    first = kw.RewriteRule(
        "x-to-y", lambda expression, context: kw.Symbol("y") if expression == kw.Symbol("x") else None,
    )
    second = kw.RewriteRule(
        "x-to-z", lambda expression, context: kw.Symbol("z") if expression == kw.Symbol("x") else None,
    )
    assert kw.rewrite_symbolic("x", (first, second)).expression == kw.Symbol("y")
    assert kw.rewrite_symbolic("x", (second, first)).expression == kw.Symbol("z")
    for strategy in ("bottom_up", "top_down"):
        repeated = [kw.rewrite_symbolic("1+x/x", strategy=strategy) for _ in range(3)]
        assert repeated[0] == repeated[1] == repeated[2]


def test_rewriting_is_a_fixed_point_and_does_not_mutate_input():
    source = kw.parse_symbolic("exp(ln(x))+x/x")
    before = repr(source)
    first = kw.rewrite_symbolic(source)
    second = kw.rewrite_symbolic(first.expression, assumptions=first.assumptions)
    assert repr(source) == before
    assert first.expression == second.expression
    assert first.assumptions == second.assumptions
    assert second.applications == ()
    with pytest.raises(FrozenInstanceError):
        first.expression = kw.ExactNumber(0)


def test_rewrite_result_and_application_serialize_structurally():
    result = kw.rewrite_symbolic("exp(ln(x))+x/x")
    payload = result.to_dict()
    assert payload["type"] == "rewrite_result"
    assert payload["applications"][0]["type"] == "rewrite_application"
    assert kw.RewriteResult.from_dict(payload) == result
    assert kw.RewriteApplication.from_dict(result.applications[0].to_dict()) == result.applications[0]
    assert object_from_dict(object_to_dict(result)) == result
    assert object_from_dict(object_to_dict(result.applications[0])) == result.applications[0]
    json.dumps(payload, allow_nan=False, sort_keys=True)

    with pytest.raises(ValueError):
        kw.RewriteResult.from_dict({"type": "wrong"})
    with pytest.raises(ValueError):
        kw.RewriteApplication.from_dict({"type": "wrong"})


def test_cycle_detection_stops_before_revisiting_a_state():
    def toggle(expression, context):
        if expression == kw.Symbol("x"):
            return kw.Symbol("y")
        if expression == kw.Symbol("y"):
            return kw.Symbol("x")
        return None

    result = kw.rewrite_symbolic("x", kw.RewriteRule("toggle-symbol", toggle))
    assert result.status == "cycle" and not result.converged
    assert result.expression == kw.Symbol("y")
    assert len(result.applications) == 1
    assert "revisit" in result.message


def test_step_and_node_resource_limits_are_reported_or_rejected():
    one_step = kw.RewriteRule(
        "x-to-y", lambda expression, context: kw.Symbol("y") if expression == kw.Symbol("x") else None,
    )
    limited = kw.rewrite_symbolic("x", one_step, max_steps=1)
    assert limited.status == "step_limit"

    expanding = kw.RewriteRule(
        "expand-node", lambda expression, context: kw.Add((expression, kw.Symbol("y"))) if expression == kw.Symbol("x") else None,
    )
    node_limited = kw.rewrite_symbolic("x", expanding, max_nodes=2)
    assert node_limited.status == "node_limit"
    assert node_limited.expression == kw.Symbol("x")
    with pytest.raises(kw.UnsupportedExpressionError):
        kw.rewrite_symbolic("x+y", rules=(), max_nodes=2)


@pytest.mark.parametrize(
    "call, error",
    (
        (lambda: kw.RewriteRule("Bad_Name", lambda expression, context: None), ValueError),
        (lambda: kw.RewriteRule("valid-name", object()), TypeError),
        (lambda: kw.RewriteRule("valid-name", lambda expression, context: None,
                                domain_preserving=1), TypeError),
        (lambda: kw.rewrite_symbolic("x", "missing-rule"), ValueError),
        (lambda: kw.rewrite_symbolic("x", (object(),)), TypeError),
        (lambda: kw.rewrite_symbolic("x", strategy="inside_out"), ValueError),
        (lambda: kw.rewrite_symbolic("x", domain="integer"), ValueError),
        (lambda: kw.rewrite_symbolic("x", introduce_conditions=1), TypeError),
        (lambda: kw.rewrite_symbolic("x", max_steps=0), ValueError),
        (lambda: kw.rewrite_symbolic("x", max_nodes=True), ValueError),
    ),
)
def test_rewrite_api_rejects_invalid_configuration(call, error):
    with pytest.raises(error):
        call()


def test_guard_must_return_an_explicit_supported_decision():
    rule = kw.RewriteRule(
        "invalid-guard",
        lambda expression, context: kw.Symbol("y") if expression == kw.Symbol("x") else None,
        lambda before, after, context: None,
    )
    with pytest.raises(TypeError):
        kw.rewrite_symbolic("x", rule)


@pytest.mark.parametrize(
    "guard_result",
    (
        kw.AssumptionSet(("x > 0",)),
        kw.parse_condition("x > 0"),
        "x > 0",
        (kw.parse_condition("x > 0"),),
    ),
)
def test_guard_condition_container_forms_are_supported(guard_result):
    rule = kw.RewriteRule(
        "guard-container",
        lambda expression, context: kw.Symbol("y") if expression == kw.Symbol("x") else None,
        lambda before, after, context: guard_result,
    )
    result = kw.rewrite_symbolic("x", rule, assumptions="x > 0")
    assert result.expression == kw.Symbol("y")
    assert result.applications[0].required.rendered == ("x > 0",)


def test_explicit_false_guard_and_duplicate_rule_ids_are_rejected_safely():
    blocked = kw.RewriteRule(
        "blocked-rule",
        lambda expression, context: kw.Symbol("y") if expression == kw.Symbol("x") else None,
        lambda before, after, context: False,
    )
    assert kw.rewrite_symbolic("x", blocked).applications == ()
    with pytest.raises(ValueError):
        kw.rewrite_symbolic("x", (blocked, blocked))


def test_rewrite_record_types_validate_manual_construction():
    x, y = kw.Symbol("x"), kw.Symbol("y")
    assumptions = kw.AssumptionSet()
    application = kw.RewriteApplication("valid", (), x, y)
    invalid_applications = (
        lambda: kw.RewriteApplication("", (), x, y),
        lambda: kw.RewriteApplication("valid", (True,), x, y),
        lambda: kw.RewriteApplication("valid", (), 1, y),
        lambda: kw.RewriteApplication("valid", (), x, y, required=()),
        lambda: kw.RewriteApplication("valid", (), x, y, explanation=1),
    )
    for factory in invalid_applications:
        with pytest.raises((TypeError, ValueError)):
            factory()

    invalid_results = (
        lambda: kw.RewriteResult(1, assumptions),
        lambda: kw.RewriteResult(x, ()),
        lambda: kw.RewriteResult(x, assumptions, (object(),)),
        lambda: kw.RewriteResult(x, assumptions, (application,), "unknown"),
    )
    for factory in invalid_results:
        with pytest.raises((TypeError, ValueError)):
            factory()

    with pytest.raises(TypeError):
        kw.RewriteContext(())
    with pytest.raises(ValueError):
        kw.RewriteContext(domain="integer")
    with pytest.raises(TypeError):
        kw.RewriteRule("valid-rule", lambda expression, context: None, guard=object())
    with pytest.raises(TypeError):
        kw.RewriteRule("valid-rule", lambda expression, context: None, explanation=1)


def test_builtin_rewrites_preserve_values_where_their_conditions_hold():
    examples = (
        ("x/x", {"x": 3}),
        ("exp(ln(x))", {"x": 2}),
        ("sqrt(x^2)", {"x": -4}),
    )
    for source, values in examples:
        original = kw.parse_symbolic(source)
        result = kw.rewrite_symbolic(original)
        assert result.assumptions.evaluate(values) is True
        assert result.expression.evaluate(values) == pytest.approx(original.evaluate(values))
