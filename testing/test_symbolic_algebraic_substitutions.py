import math

import pytest

import kiwicalc as kw
from kiwicalc.equations import symbolic
from kiwicalc.serialization import object_from_dict, object_to_dict


def _numeric_values(result):
    assert isinstance(result.solution_set, kw.FiniteSolutionSet)
    return tuple(float(complex(value.evaluate({})).real) for value in result.solution_set.values)


def test_sparse_even_polynomial_uses_exact_power_substitution():
    result = kw.solve_equation("x^4-5*x^2+4=0", steps=True)
    assert result.solution_set.values == (
        kw.ExactNumber(-2), kw.ExactNumber(-1), kw.ExactNumber(1), kw.ExactNumber(2),
    )
    assert [step.rule for step in result.steps] == ["normalize", "algebraic_substitution"]
    assert isinstance(result.steps[-1].after, kw.FiniteSolutionSet)
    assert "_kiwi_u = x^2" in result.steps[-1].explanation


def test_power_substitution_preserves_composed_multiplicity():
    result = kw.solve_equation("x^4-2*x^2+1=0")
    assert result.solution_set.values == (kw.ExactNumber(-1), kw.ExactNumber(1))
    assert result.solution_set.multiplicities == (2, 2)


def test_sparse_odd_power_reduces_the_degree_of_algebraic_roots():
    result = kw.solve_equation("x^6-5*x^3+6=0")
    assert len(result.solution_set.values) == 2
    assert all(isinstance(value, kw.RootOf) for value in result.solution_set.values)
    assert _numeric_values(result) == pytest.approx(
        (2 ** (1 / 3), 3 ** (1 / 3)), abs=1e-10,
    )


def test_repeated_nonlinear_algebraic_expression_is_substituted():
    result = kw.solve_equation("(x^2+1)^2-5*(x^2+1)+6=0", steps=True)
    assert _numeric_values(result) == pytest.approx(
        (-math.sqrt(2), -1, 1, math.sqrt(2)), abs=1e-10,
    )
    assert result.steps[-1].rule == "algebraic_substitution"


def test_repeated_rational_expression_retains_its_original_hole():
    result = kw.solve_equation("(x+1/x)^2-5*(x+1/x)+6=0")
    assert _numeric_values(result) == pytest.approx(
        ((3-math.sqrt(5))/2, 1, (3+math.sqrt(5))/2), abs=1e-10,
    )
    assert "x != 0" in result.conditions


@pytest.mark.parametrize(
    "equation, expected",
    (
        ("exp(x)^2-5*exp(x)+6=0", (math.log(2), math.log(3))),
        ("exp(2*x)-5*exp(x)+6=0", (math.log(2), math.log(3))),
        ("(2^x)^2-5*2^x+6=0", (1, math.log(3, 2))),
        ("2^(2*x)-5*2^x+6=0", (1, math.log(3, 2))),
        ("4^x-5*2^x+6=0", (1, math.log(3, 2))),
    ),
)
def test_repeated_and_compatible_exponential_families(equation, expected):
    result = kw.solve_equation(equation, steps=True)
    assert _numeric_values(result) == pytest.approx(expected, abs=1e-10)
    assert result.steps[-1].rule == "algebraic_substitution"


def test_trigonometric_substitution_returns_complete_periodic_branches():
    unbounded = kw.solve_equation("sin(x)^2-sin(x)=0", steps=True)
    assert isinstance(unbounded.solution_set, kw.UnionSolutionSet)
    assert all(isinstance(part, kw.ParametricSolutionSet) for part in unbounded.solution_set.sets)
    assert unbounded.complete and unbounded.exact

    bounded = kw.solve_equation(
        "sin(x)^2-sin(x)=0", interval=(0, 2 * math.pi),
    )
    assert _numeric_values(bounded) == pytest.approx(
        (0, math.pi / 2, math.pi, 2 * math.pi), abs=1e-10,
    )


def test_absolute_radical_and_logarithmic_substitutions_preserve_domains():
    absolute = kw.solve_equation("abs(x)^2-5*abs(x)+6=0")
    assert absolute.solution_set.values == tuple(map(kw.ExactNumber, (-3, -2, 2, 3)))

    radical = kw.solve_equation("sqrt(x)^2-3*sqrt(x)+2=0")
    assert radical.solution_set.values == (kw.ExactNumber(1), kw.ExactNumber(4))
    assert radical.conditions == ("x >= 0",)

    logarithmic = kw.solve_equation("ln(x)^2-3*ln(x)+2=0")
    assert _numeric_values(logarithmic) == pytest.approx((math.e, math.e ** 2))
    assert logarithmic.conditions == ("x > 0",)


def test_rational_function_of_a_kernel_is_substituted_with_hole_preservation():
    result = kw.solve_equation("1/sin(x)=2", interval=(0, 2 * math.pi))
    assert _numeric_values(result) == pytest.approx(
        (math.pi / 6, 5 * math.pi / 6), abs=1e-10,
    )
    assert "sin(x) != 0" in result.conditions


@pytest.mark.parametrize(
    "equation",
    (
        "exp(x)^2+3*exp(x)+2=0",
        "sin(x)^2-4=0",
        "sqrt(x)^2+3*sqrt(x)+2=0",
    ),
)
def test_outer_roots_outside_the_inner_function_range_are_rejected(equation):
    result = kw.solve_equation(equation, method="symbolic")
    assert isinstance(result.solution_set, kw.EmptySolutionSet)
    assert result.complete


def test_unsupported_inner_equation_is_not_returned_as_a_partial_solution():
    result = kw.solve_equation("sin(x^2)^2=1", method="symbolic", steps=True)
    assert result.status == "unresolved"
    assert [step.rule for step in result.steps] == ["normalize"]


def test_substitution_degree_limit_declines_excessive_expansion_safely():
    result = kw.solve_equation("sin(x)^33=1", method="symbolic")
    assert result.status == "unresolved"


def test_complex_domain_uses_algebraic_but_not_transcendental_substitution():
    algebraic = kw.solve_equation("x^4-5*x^2+4=0", domain="complex", steps=True)
    assert algebraic.status == "solved"
    assert algebraic.steps[-1].rule == "algebraic_substitution"

    transcendental = kw.solve_equation(
        "exp(x)^2-5*exp(x)+6=0", domain="complex", method="symbolic",
    )
    assert transcendental.status == "unresolved"


def test_substitution_results_are_deterministic_and_serializable():
    first = kw.solve_equation("4^x-5*2^x+6=0", steps=True)
    second = kw.solve_equation("4^x-5*2^x+6=0", steps=True)
    assert first == second
    assert object_from_dict(object_to_dict(first)) == first


def test_auxiliary_symbol_is_fresh_when_user_equation_contains_default_name():
    auxiliary = symbolic._fresh_substitution_symbol(
        kw.parse_symbolic("_kiwi_u+sin(x)^2"),
    )
    assert auxiliary.name == "__kiwi_u"
