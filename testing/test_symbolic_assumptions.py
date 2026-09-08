import inspect
import json

import pytest

import kiwicalc as kw
from kiwicalc.serialization import object_from_dict, object_to_dict


def test_existing_solver_signature_remains_frozen_and_assuming_api_is_additive():
    assert tuple(inspect.signature(kw.solve_equation).parameters) == (
        "equation", "variable", "domain", "interval", "method",
        "numeric_fallback", "tolerance", "max_iterations", "steps",
    )
    assert tuple(inspect.signature(kw.solve_equation_assuming).parameters) == (
        "equation", "assumptions", "variable", "domain", "interval", "method",
        "numeric_fallback", "tolerance", "max_iterations", "steps",
    )


@pytest.mark.parametrize(
    ("text", "condition_type", "values", "expected"),
    (
        ("x > 0", kw.RelationCondition, {"x": 2}, True),
        ("x != 1", kw.RelationCondition, {"x": 1}, False),
        ("0 <= x <= 2", kw.BetweenCondition, {"x": 2}, True),
        ("sqrt(x) is defined in the real domain", kw.DefinedCondition, {"x": -1}, False),
    ),
)
def test_atomic_condition_parser_and_three_valued_evaluation(text, condition_type, values, expected):
    condition = kw.parse_condition(text)
    assert isinstance(condition, condition_type)
    assert condition.evaluate(values) is expected
    assert condition.evaluate({}) is None


def test_condition_parser_rejects_boolean_text_instead_of_guessing_precedence():
    with pytest.raises(kw.EquationParseError):
        kw.parse_condition("x > 0 and y > 0")


def test_assumption_set_detects_direct_and_bound_contradictions():
    direct = kw.AssumptionSet((kw.parse_condition("x = 1"), kw.parse_condition("x != 1")))
    bounded = kw.AssumptionSet((kw.parse_condition("x > 2"), kw.parse_condition("x <= 2")))
    separated = kw.AssumptionSet((kw.parse_condition("x >= 3"), kw.parse_condition("x < 2")))
    assert direct.contradictory
    assert bounded.contradictory
    assert separated.contradictory
    assert direct.evaluate({"x": 1}) is False


def test_assumption_entailment_understands_strict_and_weak_bounds():
    assumptions = kw.AssumptionSet((kw.parse_condition("x >= 0"), kw.parse_condition("x != 0")))
    assert assumptions.entails(kw.parse_condition("x > 0"))
    assert assumptions.entails(kw.parse_condition("x >= -1"))
    assert assumptions.refutes(kw.parse_condition("x <= 0"))


def test_assumption_substitution_reduces_known_predicates():
    assumptions = kw.AssumptionSet((kw.parse_condition("x > 0"), kw.parse_condition("a != 1")))
    substituted = assumptions.substitute({"x": 3})
    assert substituted.rendered == ("a != 1",)
    assert substituted.evaluate({"a": 2}) is True
    assert substituted.evaluate({}) is None


def test_solver_exposes_structural_domain_assumptions_with_legacy_strings():
    result = kw.solve_equation("ln(x)=2")
    assert result.conditions == ("x > 0",)
    assert isinstance(result.assumptions, kw.AssumptionSet)
    assert isinstance(result.assumptions.conditions[0], kw.RelationCondition)
    assert result.assumptions.evaluate({"x": 2}) is True
    assert result.assumptions.evaluate({"x": -1}) is False


def test_unresolved_results_retain_original_domain_assumptions():
    result = kw.solve_equation("ln(x)+x=0", method="symbolic")
    assert result.status == "unresolved"
    assert result.conditions == ("x > 0",)
    assert isinstance(result.assumptions.conditions[0], kw.RelationCondition)


def test_assuming_api_filters_finite_roots_against_conditions():
    result = kw.solve_equation_assuming("x^2=1", "x > 0")
    assert tuple(float(value) for value in result.solution_set.values) == (1.0,)
    assert result.conditions == ("x > 0",)


def test_assuming_api_resolves_parameter_dependent_linear_branches():
    nonzero = kw.solve_equation_assuming("a*x+b=0", "a != 0", variable="x")
    identity = kw.solve_equation_assuming("a*x+b=0", ("a=0", "b=0"), variable="x")
    contradiction = kw.solve_equation_assuming("a*x+b=0", ("a=0", "b!=0"), variable="x")
    assert isinstance(nonzero.solution_set, kw.FiniteSolutionSet)
    assert isinstance(identity.solution_set, kw.UniversalSolutionSet)
    assert isinstance(contradiction.solution_set, kw.EmptySolutionSet)


def test_assuming_api_accepts_exact_value_mapping():
    result = kw.solve_equation_assuming("a*x=4", {"a": 2}, variable="x")
    assert tuple(float(value) for value in result.solution_set.values) == (2.0,)
    assert result.assumptions.evaluate({"a": 2}) is True
    exponential = kw.solve_equation_assuming("a^x=8", {"a": 2}, variable="x")
    assert tuple(float(value) for value in exponential.solution_set.values) == (3.0,)


def test_numeric_fallback_uses_parameter_values_and_filters_by_assumptions():
    parameterized = kw.solve_equation_assuming(
        "a*x=2", {"a": 2}, variable="x", method="numeric", interval=(-2, 2)
    )
    assert tuple(parameterized.solution_set.values) == pytest.approx((1.0,))

    positive = kw.solve_equation_assuming(
        "x^2=1", "x > 0", method="numeric", interval=(-2, 2)
    )
    assert tuple(positive.solution_set.values) == pytest.approx((1.0,))


def test_contradictory_caller_assumptions_return_explicit_empty_result():
    result = kw.solve_equation_assuming("x=x", ("x > 0", "x <= 0"))
    assert isinstance(result.solution_set, kw.EmptySolutionSet)
    assert result.complete and result.exact
    assert result.assumptions.contradictory
    assert "contradictory" in result.message


def test_caller_assumptions_reject_malformed_predicates_instead_of_ignoring_them():
    with pytest.raises(kw.EquationParseError):
        kw.solve_equation_assuming("x=1", "x is probably positive")


def test_impossible_between_condition_simplifies_to_a_contradiction():
    assumptions = kw.AssumptionSet((kw.parse_condition("2 < x < 2"),))
    assert assumptions.contradictory


def test_conditional_solution_and_steps_retain_structural_assumptions():
    result = kw.solve_equation("1/(x-1)=0", steps=True)
    assert isinstance(result.steps[1].assumptions.conditions[0], kw.RelationCondition)
    conditional = kw.ConditionalSolutionSet(kw.FiniteSolutionSet((1,)), (kw.parse_condition("a > 0"),))
    assert conditional.conditions == ("a > 0",)
    assert conditional.assumptions.evaluate({"a": 1}) is True


def test_condition_assumption_and_solution_serialization_is_structural_and_strict_json():
    condition = kw.parse_condition("0 <= x <= 2")
    assumptions = kw.AssumptionSet((condition, kw.parse_condition("a != 0")))
    restored_condition = kw.condition_from_dict(condition.to_dict())
    restored_assumptions = kw.AssumptionSet.from_dict(assumptions.to_dict())
    assert restored_condition == condition
    assert restored_assumptions == assumptions
    assert object_from_dict(object_to_dict(condition)) == condition
    assert object_from_dict(object_to_dict(assumptions)) == assumptions

    result = kw.solve_equation_assuming("x^2=1", assumptions=("x > 0",))
    payload = result.to_dict()
    assert payload["assumptions"]["conditions"][0]["type"] == "relation"
    assert kw.EquationSolution.from_dict(payload) == result
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_legacy_condition_only_payloads_remain_readable_and_become_structural():
    result = kw.solve_equation("ln(x)=2")
    payload = result.to_dict()
    payload.pop("assumptions")
    restored = kw.EquationSolution.from_dict(payload)
    assert restored.conditions == result.conditions
    assert isinstance(restored.assumptions.conditions[0], kw.RelationCondition)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: kw.RelationCondition(kw.Symbol("x"), "approximately", kw.ExactNumber(0)),
        lambda: kw.DefinedCondition(kw.Symbol("x"), "integers"),
        lambda: kw.CompoundCondition("not", (kw.parse_condition("x>0"), kw.parse_condition("x<0"))),
        lambda: kw.AssumptionSet((object(),)),
    ),
)
def test_condition_model_rejects_invalid_construction(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("tolerance", (-1, float("nan"), True))
def test_condition_evaluation_validates_tolerance(tolerance):
    with pytest.raises(ValueError):
        kw.parse_condition("x > 0").evaluate({"x": 1}, tolerance=tolerance)


def test_truth_and_opaque_conditions_cover_legacy_three_valued_behavior():
    assert str(kw.TruthCondition(True)) == "True"
    assert str(kw.TruthCondition(False)) == "False"
    assert kw.TruthCondition(True).substitute({}) == kw.TruthCondition(True)
    assert kw.OpaqueCondition("True").evaluate() is True
    assert kw.OpaqueCondition("False").evaluate() is False
    assert kw.OpaqueCondition("legacy predicate").evaluate() is None
    assert kw.OpaqueCondition("legacy predicate").substitute({}) == kw.OpaqueCondition("legacy predicate")
    with pytest.raises(TypeError):
        kw.TruthCondition(False).evaluate([])
    with pytest.raises(TypeError):
        kw.TruthCondition(False).substitute([])
    with pytest.raises(TypeError):
        kw.TruthCondition(1)
    with pytest.raises(ValueError):
        kw.OpaqueCondition("")


def test_relation_conditions_handle_unknown_exception_nonfinite_and_complex_values():
    missing = kw.parse_condition("x > y")
    assert missing.variables == {"x", "y"}
    assert missing.evaluate({"x": 2}) is None
    assert kw.parse_condition("1/x > 0").evaluate({"x": 0}) is False
    assert kw.parse_condition("x = 0").evaluate({"x": float("inf")}) is False
    assert kw.parse_condition("x > 0").evaluate({"x": 1j}) is False
    assert kw.parse_condition("x < 2").evaluate({"x": 1}) is True
    assert kw.parse_condition("x <= 2").evaluate({"x": 2}) is True
    assert kw.parse_condition("x >= 2").evaluate({"x": 2}) is True
    assert kw.parse_condition("x = 2").substitute({"x": 2}) == kw.TruthCondition(True)
    assert kw.parse_condition("x = 2").substitute({"x": 3}) == kw.TruthCondition(False)
    assert str(kw.RelationCondition(kw.Symbol("x"), ">", kw.ExactNumber(0))) == "x > 0"
    with pytest.raises(TypeError):
        kw.RelationCondition(1, ">", kw.ExactNumber(0))
    with pytest.raises(ValueError):
        kw.RelationCondition(kw.Symbol("x"), ">", kw.ExactNumber(0), "")


def test_defined_conditions_cover_real_complex_and_undefined_cases():
    real = kw.DefinedCondition(kw.parse_symbolic("sqrt(x)"), "real")
    complex_domain = kw.DefinedCondition(kw.parse_symbolic("sqrt(x)"), "complex")
    assert real.variables == {"x"}
    assert real.evaluate({"x": 4}) is True
    assert real.evaluate({"x": -1}) is False
    assert complex_domain.evaluate({"x": -1}) is True
    assert kw.DefinedCondition(kw.parse_symbolic("1/x"), "complex").evaluate({"x": 0}) is False
    assert real.substitute({"x": 4}) == kw.TruthCondition(True)
    assert str(real) == "sqrt(x) is defined in the real domain"
    with pytest.raises(TypeError):
        kw.DefinedCondition("x")


def test_between_conditions_support_open_bounds_substitution_and_validation():
    condition = kw.BetweenCondition(
        kw.Symbol("x"), kw.ExactNumber(0), kw.ExactNumber(2), False, False
    )
    assert condition.variables == {"x"}
    assert str(condition) == "0 < x < 2"
    assert condition.evaluate({"x": 0}) is False
    assert condition.evaluate({"x": 1}) is True
    assert condition.evaluate({}) is None
    assert condition.substitute({"x": 1}) == kw.TruthCondition(True)
    with pytest.raises(TypeError):
        kw.BetweenCondition(kw.Symbol("x"), 0, kw.ExactNumber(1))
    with pytest.raises(TypeError):
        kw.BetweenCondition(kw.Symbol("x"), kw.ExactNumber(0), kw.ExactNumber(1), 1, True)
    with pytest.raises(ValueError):
        kw.BetweenCondition(kw.Symbol("x"), kw.ExactNumber(0), kw.ExactNumber(1), display="")


def test_compound_conditions_evaluate_simplify_substitute_and_render():
    positive = kw.parse_condition("x > 0")
    small = kw.parse_condition("x < 2")
    conjunction = kw.CompoundCondition("and", (positive, small))
    disjunction = kw.CompoundCondition("or", (positive, small))
    negation = kw.CompoundCondition("not", (positive,))
    assert conjunction.variables == {"x"}
    assert conjunction.evaluate({"x": 1}) is True
    assert conjunction.evaluate({"x": 3}) is False
    assert conjunction.evaluate({}) is None
    assert disjunction.evaluate({"x": 3}) is True
    assert disjunction.evaluate({}) is None
    assert kw.CompoundCondition("or", (kw.TruthCondition(False), kw.TruthCondition(False))).evaluate() is False
    assert negation.evaluate({"x": 1}) is False
    assert negation.evaluate({}) is None
    assert negation.substitute({"x": 1}) == kw.TruthCondition(False)
    assert "and" in str(conjunction) and str(negation).startswith("not")

    nested = kw.CompoundCondition("and", (conjunction, kw.TruthCondition(True)))
    assert kw.simplify_condition(nested) == conjunction
    assert kw.simplify_condition(kw.CompoundCondition("and", (positive, kw.TruthCondition(False)))) == kw.TruthCondition(False)
    assert kw.simplify_condition(kw.CompoundCondition("or", (positive, kw.TruthCondition(True)))) == kw.TruthCondition(True)
    assert kw.simplify_condition(kw.CompoundCondition("or", (kw.TruthCondition(False), positive))) == positive
    assert kw.simplify_condition(kw.CompoundCondition("and", (kw.TruthCondition(True),))) == kw.TruthCondition(True)
    assert kw.simplify_condition(kw.CompoundCondition("not", (kw.TruthCondition(True),))) == kw.TruthCondition(False)

    with pytest.raises(ValueError):
        kw.CompoundCondition("xor", (positive,))
    with pytest.raises(TypeError):
        kw.CompoundCondition("and", ())
    with pytest.raises(TypeError):
        kw.CompoundCondition("and", ("x > 0",))


def test_assumption_collection_protocol_substitutions_and_compound_entailment():
    right_equality = kw.RelationCondition(kw.ExactNumber(3), "=", kw.Symbol("b"))
    assumptions = kw.AssumptionSet(("x > 0", right_equality, kw.DefinedCondition(kw.Symbol("z"), "real")))
    assert len(assumptions) == 3 and bool(assumptions)
    assert tuple(assumptions) == assumptions.conditions
    assert "x > 0" in assumptions
    assert kw.parse_condition("x > 0") in assumptions
    assert assumptions.variables == {"x", "b", "z"}
    assert assumptions.substitutions["b"] == kw.ExactNumber(3)
    assert "x > 0" in str(assumptions)
    assert str(kw.AssumptionSet()) == "True"
    assert not kw.AssumptionSet()

    chained = kw.AssumptionSet(("a=b", "b=2"))
    assert chained.substitutions["a"] == kw.ExactNumber(2)

    both = kw.CompoundCondition("and", (kw.parse_condition("x >= -1"), kw.parse_condition("x != -2")))
    either = kw.CompoundCondition("or", (kw.parse_condition("x > 10"), kw.parse_condition("x >= -1")))
    assert assumptions.entails(both)
    assert assumptions.entails(either)
    assert kw.AssumptionSet((kw.TruthCondition(False),)).entails(kw.parse_condition("y=2"))
    assert assumptions.refutes(kw.TruthCondition(False))


def test_assumption_contradiction_covers_equalities_strict_edges_and_negation():
    assert kw.AssumptionSet(("x=1", "x=2")).contradictory
    assert kw.AssumptionSet(("x=1", "x>1")).contradictory
    assert kw.AssumptionSet(("x=1", "x<1")).contradictory
    assert kw.AssumptionSet(("x>=2", "x<1")).contradictory
    predicate = kw.parse_condition("x>0")
    assert kw.AssumptionSet((predicate, kw.CompoundCondition("not", (predicate,)))).contradictory
    assert not kw.AssumptionSet(("x>=0", "x<=0")).contradictory


def test_relation_implication_covers_all_bound_directions_and_equality():
    cases = (
        ("x=2", "x>=1"), ("x!=2", "x!=2"),
        ("x>2", "x>=2"), ("x>2", "x!=1"),
        ("x>=2", "x>1"), ("x>=2", "x!=1"),
        ("x<2", "x<=2"), ("x<2", "x!=3"),
        ("x<=2", "x<3"), ("x<=2", "x!=3"),
    )
    for source, target in cases:
        assert kw.AssumptionSet((source,)).entails(kw.parse_condition(target))
    assert not kw.AssumptionSet(("x>0",)).entails(kw.parse_condition("y>0"))


def test_condition_serialization_covers_every_predicate_variant_and_errors():
    conditions = (
        kw.TruthCondition(True),
        kw.RelationCondition(kw.Symbol("x"), ">", kw.ExactNumber(0)),
        kw.DefinedCondition(kw.Symbol("x"), "complex"),
        kw.BetweenCondition(kw.Symbol("x"), kw.ExactNumber(0), kw.ExactNumber(1)),
        kw.OpaqueCondition("legacy"),
        kw.CompoundCondition("not", (kw.parse_condition("x=0"),)),
    )
    for condition in conditions:
        assert kw.condition_from_dict(condition.to_dict()) == condition
    with pytest.raises(ValueError):
        kw.condition_from_dict({"type": "missing"})
    with pytest.raises(ValueError):
        kw.AssumptionSet.from_dict({"type": "wrong"})

    class UnknownCondition(kw.Condition):
        pass

    with pytest.raises(TypeError):
        UnknownCondition().to_dict()


def test_assuming_input_forms_and_validation_are_explicit():
    parsed = kw.AssumptionSet(("x>0",))
    assert kw.solve_equation_assuming("x=1", parsed).assumptions == parsed
    assert isinstance(kw.solve_equation_assuming("x=1", kw.parse_condition("x>0")).solution_set, kw.FiniteSolutionSet)
    assert isinstance(kw.solve_equation_assuming("x=1", {kw.Symbol("p"): "real"}).assumptions.conditions[0], kw.DefinedCondition)
    with pytest.raises(TypeError):
        kw.solve_equation_assuming("x=1", 42)
    with pytest.raises(TypeError):
        kw.solve_equation_assuming("x=1", [object()])
    with pytest.raises((TypeError, ValueError)):
        kw.solve_equation_assuming("x=1", {"p": "integers"})
