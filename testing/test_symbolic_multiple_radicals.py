import math

import pytest

import kiwicalc as kw


def numeric(value):
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value)


def numeric_values(result):
    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    return sorted(numeric(value).real for value in result.solutions)


def test_two_additive_square_roots_are_isolated_repeatedly():
    result = kw.solve_equation(
        "sqrt(x+1)+sqrt(x-1)=3", method="symbolic", steps=True,
    )

    assert result.exact and result.complete
    assert numeric_values(result) == pytest.approx([85 / 36])
    assert [step.rule for step in result.steps].count("isolate_radical") == 2
    assert [step.rule for step in result.steps].count("isolate_additive_radical") == 2


def test_radicals_on_both_sides_preserve_principal_sign_conditions():
    result = kw.solve_equation(
        "sqrt(x+2)=sqrt(2*x-1)+1", method="symbolic", steps=True,
    )

    assert numeric_values(result) == pytest.approx([6 - 2 * math.sqrt(7)])
    assert result.exact and result.complete
    assert any(">= 0" in condition for condition in result.conditions)


def test_nested_radical_reduces_to_an_exact_polynomial():
    result = kw.solve_equation(
        "sqrt(x+sqrt(x))=2", method="symbolic", steps=True,
    )

    assert numeric_values(result) == pytest.approx([(9 - math.sqrt(17)) / 2])
    assert result.steps[-1].rule == "solve_polynomial"


def test_repeated_identical_radicals_combine_before_isolation():
    result = kw.solve_equation("sqrt(x)+sqrt(x)=2", method="symbolic")

    assert numeric_values(result) == pytest.approx([1])


def test_multiple_squaring_extraneous_roots_are_rejected_in_original_equation():
    result = kw.solve_equation(
        "sqrt(x+2)=sqrt(2*x-1)+1", method="symbolic",
    )

    assert len(result.solutions) == 1
    value = numeric(result.solutions[0]).real
    assert math.sqrt(value + 2) == pytest.approx(math.sqrt(2 * value - 1) + 1)
    assert value != pytest.approx(6 + 2 * math.sqrt(7))


def test_inconsistent_multiple_radical_equation_returns_exact_empty_set():
    result = kw.solve_equation(
        "sqrt(x+1)+sqrt(x-1)=-1", method="symbolic",
    )

    assert isinstance(result.solution_set, kw.EmptySolutionSet)
    assert result.exact and result.complete


def test_nonadditive_radical_products_decline_without_partial_roots():
    result = kw.solve_equation(
        "sqrt(x)*sqrt(x+1)=2", method="symbolic",
    )

    assert result.status == "unresolved"
    assert not result.complete


def test_multiple_radical_result_is_deterministic_and_serializable():
    equation = "sqrt(x+1)+sqrt(x-1)=3"
    first = kw.solve_equation(equation, method="symbolic", steps=True)
    second = kw.solve_equation(equation, method="symbolic", steps=True)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert kw.EquationSolution.from_dict(first.to_dict()) == first
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(first)) == first


def test_existing_single_radical_contract_remains_unchanged():
    result = kw.solve_equation("sqrt(x+1)=x-1", method="symbolic")

    assert numeric_values(result) == pytest.approx([3])
    assert result.conditions == ("x + 1 >= 0", "-1 + x >= 0")
