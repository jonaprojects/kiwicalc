import pytest

import kiwicalc as kw


def numeric(value):
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value)


def collect_conditions(solution_set):
    if isinstance(solution_set, kw.ConditionalSolutionSet):
        return set(solution_set.conditions) | collect_conditions(solution_set.solution_set)
    if isinstance(solution_set, kw.UnionSolutionSet):
        return set().union(*(collect_conditions(part) for part in solution_set.sets))
    return set()


def test_general_symbolic_quadratic_returns_complete_conditional_formula():
    result = kw.solve_equation(
        "a*x^2+b*x+c=0", variable="x", steps=True,
    )

    assert result.status == "solved"
    assert result.exact and result.complete
    assert isinstance(result.solution_set, kw.UnionSolutionSet)
    conditions = collect_conditions(result.solution_set)
    assert "a != 0" in conditions
    assert "a = 0" in conditions
    assert any("b^2" in condition and "> 0" in condition for condition in conditions)
    assert result.steps[-1].rule == "solve_symbolic_quadratic"


@pytest.mark.parametrize(
    "parameters, expected",
    (
        ({"a": 1, "b": -5, "c": 6}, (2.0, 3.0)),
        ({"a": 0, "b": 2, "c": -4}, (2.0,)),
    ),
)
def test_parameter_values_reduce_to_existing_exact_solver(parameters, expected):
    result = kw.solve_equation_assuming(
        "a*x^2+b*x+c=0", parameters, variable="x",
    )

    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    values = tuple(sorted(numeric(value).real for value in result.solutions))
    assert values == pytest.approx(expected)


def test_symbolic_quadratic_degenerate_identity_and_contradiction():
    identity = kw.solve_equation_assuming(
        "a*x^2+b*x+c=0", {"a": 0, "b": 0, "c": 0}, variable="x",
    )
    contradiction = kw.solve_equation_assuming(
        "a*x^2+b*x+c=0", {"a": 0, "b": 0, "c": 1}, variable="x",
    )

    assert isinstance(identity.solution_set, kw.UniversalSolutionSet)
    assert isinstance(contradiction.solution_set, kw.EmptySolutionSet)
    assert identity.complete and contradiction.complete


def test_discriminant_conditions_select_distinct_repeated_and_empty_real_branches():
    distinct = kw.solve_equation_assuming(
        "x^2+p*x+1=0", "p^2-4>0", variable="x",
    )
    repeated = kw.solve_equation_assuming(
        "x^2+p*x+1=0", "p^2-4=0", variable="x",
    )
    empty = kw.solve_equation_assuming(
        "x^2+p*x+1=0", "p^2-4<0", variable="x",
    )

    assert isinstance(distinct.solution_set, kw.FiniteSolutionSet)
    assert len(distinct.solutions) == 2
    assert isinstance(repeated.solution_set, kw.FiniteSolutionSet)
    assert repeated.solution_set.multiplicities == (2,)
    assert isinstance(empty.solution_set, kw.EmptySolutionSet)


def test_complex_symbolic_quadratic_uses_zero_and_nonzero_discriminant_branches():
    result = kw.solve_equation(
        "a*x^2+b*x+c=0", variable="x", domain="complex",
    )

    conditions = collect_conditions(result.solution_set)
    assert "a != 0" in conditions
    assert any("!= 0" in condition and "b^2" in condition for condition in conditions)
    assert any("= 0" in condition and "b^2" in condition for condition in conditions)
    assert result.exact and result.complete


def test_nonzero_leading_assumption_removes_degenerate_branch():
    result = kw.solve_equation_assuming(
        "a*x^2+b*x+c=0", "a!=0", variable="x",
    )

    assert "a = 0" not in collect_conditions(result.solution_set)
    assert result.complete


def test_zero_product_solves_factored_symbolic_cubic_without_expanding_formula():
    result = kw.solve_equation(
        "(a*x+b)*(x-1)*(x+2)=0", variable="x", steps=True,
    )

    assert result.exact and result.complete
    assert result.steps[-1].rule == "solve_zero_product"
    rendered = str(result.solution_set)
    assert "1" in rendered and "-2" in rendered
    assert "a != 0" in collect_conditions(result.solution_set)


def test_parameter_only_zero_factor_retains_universal_branch():
    result = kw.solve_equation("a*(x-1)=0", variable="x")

    assert result.exact and result.complete
    assert isinstance(result.solution_set, kw.UnionSolutionSet)
    assert any(
        isinstance(part, kw.ConditionalSolutionSet)
        and isinstance(part.solution_set, kw.UniversalSolutionSet)
        and "a = 0" in part.conditions
        for part in result.solution_set.sets
    )


def test_symbolic_quadratic_result_is_deterministic_and_serializable():
    first = kw.solve_equation("a*x^2+b*x+c=0", variable="x", steps=True)
    second = kw.solve_equation("a*x^2+b*x+c=0", variable="x", steps=True)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert kw.EquationSolution.from_dict(first.to_dict()) == first
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(first)) == first


def test_parameter_aware_biquadratic_returns_complete_conditional_branches():
    result = kw.solve_equation(
        "a*x^4+b*x^2+c=0", variable="x", steps=True,
    )

    assert result.status == "solved"
    assert result.exact and result.complete
    assert isinstance(result.solution_set, kw.UnionSolutionSet)
    assert result.steps[-1].rule == "algebraic_substitution"
    conditions = collect_conditions(result.solution_set)
    assert "a != 0" in conditions
    assert "a = 0" in conditions


def test_biquadratic_parameter_values_produce_all_four_exact_roots():
    result = kw.solve_equation_assuming(
        "a*x^4+b*x^2+c=0",
        {"a": 1, "b": -5, "c": 4}, variable="x",
    )

    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    assert tuple(sorted(numeric(value).real for value in result.solutions)) == pytest.approx(
        (-2, -1, 1, 2)
    )


def test_biquadratic_complex_domain_is_complete_and_serializable():
    result = kw.solve_equation(
        "a*x^4+b*x^2+c=0", variable="x", domain="complex", steps=True,
    )

    assert result.exact and result.complete
    restored = kw.EquationSolution.from_dict(result.to_dict())
    assert restored == result


def test_symbolic_polynomial_degree_above_supported_formula_remains_unresolved():
    result = kw.solve_equation("a*x^3+b*x+c=0", variable="x")

    assert result.status == "unresolved"
    assert not result.complete
