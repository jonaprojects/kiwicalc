import math

import pytest

import kiwicalc as kw
from kiwicalc.equations import symbolic
from kiwicalc.serialization import object_from_dict, object_to_dict


def _step(result, identifier):
    return next(step for step in result.steps if step.rule == identifier)


def test_solver_transformations_are_an_ordered_guarded_rule_registry():
    assert tuple(rule.identifier for rule in symbolic._SOLVER_RULES) == (
        "clear_denominators",
        "trigonometric_reduction",
        "algebraic_substitution",
        "solve_polynomial",
        "solve_zero_product",
        "solve_symbolic_quadratic",
        "solve_symbolic_linear",
        "combine_logarithms",
        "invert_exponential",
        "split_absolute",
        "isolate_radical",
        "isolate_additive_radical",
        "invert_logarithm",
        "invert_trigonometric",
    )
    assert len({rule.identifier for rule in symbolic._SOLVER_RULES}) == len(symbolic._SOLVER_RULES)
    assert all(callable(rule.transform) for rule in symbolic._SOLVER_RULES)
    assert all(rule.domains for rule in symbolic._SOLVER_RULES)


@pytest.mark.parametrize(
    "equation, identifier, admissible_points",
    (
        ("(x+1)/(x-2)=0", "clear_denominators", (-1, 0, 3)),
        ("ln(x)+ln(x-1)=ln(2)", "combine_logarithms", (2, 3, 5)),
        ("exp(2*x+1)=7", "invert_exponential", (0, (math.log(7)-1)/2, 2)),
        ("sqrt(x+1)=x-1", "isolate_radical", (1, 3, 8)),
        ("ln(2*x+1)=3", "invert_logarithm", (0, (math.exp(3)-1)/2, 10)),
        ("2^(x+1)=2^4", "invert_exponential", (0, 3, 6)),
    ),
)
def test_equation_to_equation_rules_emit_replayable_structural_steps(
    equation, identifier, admissible_points,
):
    result = kw.solve_equation(equation, steps=True)
    transformation = _step(result, identifier)
    assert isinstance(transformation.before, kw.EquationState)
    assert isinstance(transformation.after, kw.EquationState)
    for point in admissible_points:
        assert transformation.equivalent_at({"x": point})


@pytest.mark.parametrize(
    "equation, identifier, result_type",
    (
        ("x^2=1", "solve_polynomial", kw.FiniteSolutionSet),
        ("abs(2*x-1)=3", "split_absolute", kw.FiniteSolutionSet),
        ("sin(x)=0", "invert_trigonometric", kw.ParametricSolutionSet),
    ),
)
def test_terminal_rules_emit_solution_sets(equation, identifier, result_type):
    result = kw.solve_equation(equation, steps=True)
    terminal = _step(result, identifier)
    assert isinstance(terminal.before, kw.EquationState)
    assert isinstance(terminal.after, result_type)
    assert terminal.after == result.solution_set


def test_rule_guards_flow_into_steps_and_final_assumptions():
    rational = kw.solve_equation("(x^2-1)/(x-1)=0", steps=True)
    cancellation = _step(rational, "clear_denominators")
    assert cancellation.conditions == ("x - 1 != 0",)
    assert cancellation.assumptions.conditions
    assert rational.conditions == ("x - 1 != 0",)

    radical = kw.solve_equation("sqrt(x+1)=x-1", steps=True)
    squaring = _step(radical, "isolate_radical")
    assert "-1 + x >= 0" in squaring.conditions
    assert "-1 + x >= 0" in radical.conditions


def test_rule_sequence_is_deterministic_and_serialization_safe():
    first = kw.solve_equation("ln(x)+ln(x-1)=ln(2)", steps=True)
    second = kw.solve_equation("ln(x)+ln(x-1)=ln(2)", steps=True)
    assert first == second
    assert [step.rule for step in first.steps] == [
        "normalize", "combine_logarithms", "solve_polynomial",
    ]
    assert object_from_dict(object_to_dict(first)) == first


def test_real_only_rules_do_not_claim_complete_complex_branches():
    for equation in ("2^x=-1", "exp(x)=1", "sin(x)=0", "sqrt(x)=1"):
        result = kw.solve_equation(
            equation, variable="x", domain="complex", method="symbolic",
            steps=True,
        )
        assert result.status == "unresolved"
        assert [step.rule for step in result.steps] == ["normalize"]


def test_guarded_rules_preserve_legacy_impossible_real_results():
    for equation in ("exp(x)=-1", "sqrt(x)=-1", "abs(x)=-1", "sin(x)=2"):
        result = kw.solve_equation(equation, method="symbolic", steps=True)
        assert result.status == "solved"
        assert isinstance(result.solution_set, kw.EmptySolutionSet)


def test_rule_resource_boundary_returns_unresolved_instead_of_looping():
    left, right = symbolic._equation_input("ln(x)=2")
    result = symbolic._symbolic_dispatch(
        left, right, "x", "real", None, lambda *args: None,
        max_transformations=1,
    )
    assert result[0] is None
    assert result[2] is False


@pytest.mark.parametrize(
    "call, error",
    (
        (lambda: symbolic._SolverRule("Bad-Rule", lambda state, context: None), ValueError),
        (lambda: symbolic._SolverRule("valid_rule", object()), TypeError),
        (lambda: symbolic._SolverRule("valid_rule", lambda state, context: None,
                                     guard=object()), TypeError),
        (lambda: symbolic._SolverRule("valid_rule", lambda state, context: None,
                                     domains=("integer",)), ValueError),
        (lambda: symbolic._SolverRuleOutcome(object()), TypeError),
        (lambda: symbolic._SolverRuleOutcome(kw.EmptySolutionSet(), complete=1), TypeError),
    ),
)
def test_internal_solver_rule_contract_rejects_malformed_definitions(call, error):
    with pytest.raises(error):
        call()
