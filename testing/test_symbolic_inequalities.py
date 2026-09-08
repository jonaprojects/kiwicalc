import inspect
import math

import pytest

import kiwicalc as kw


def numeric(value):
    if value is None:
        return None
    return complex(value.evaluate({}) if isinstance(value, kw.SymbolicExpression) else value).real


def parts(solution_set):
    return solution_set.sets if isinstance(solution_set, kw.UnionSolutionSet) else (solution_set,)


def interval_signature(solution_set):
    result = []
    for part in parts(solution_set):
        if isinstance(part, kw.FiniteSolutionSet):
            result.append(("points", tuple(numeric(value) for value in part.values)))
        else:
            result.append((
                numeric(part.lower), numeric(part.upper),
                part.lower_closed, part.upper_closed,
            ))
    return result


def test_solve_inequality_is_public_with_a_stable_keyword_contract():
    assert kw.solve_inequality is not None
    assert tuple(inspect.signature(kw.solve_inequality).parameters) == (
        "inequality", "variable", "domain", "assumptions", "steps",
    )


def test_quadratic_inequality_returns_two_exact_unbounded_intervals():
    result = kw.solve_inequality("x^2-5*x+6>=0", steps=True)

    assert result.exact and result.complete
    assert interval_signature(result.solution_set) == [
        (None, 2.0, False, True),
        (3.0, None, True, False),
    ]
    assert [step.rule for step in result.steps] == [
        "normalize_inequality", "rational_sign_chart",
    ]


def test_strict_quadratic_inequality_uses_open_exact_roots():
    result = kw.solve_inequality("x^2-2<0")

    assert interval_signature(result.solution_set) == [
        (-math.sqrt(2), math.sqrt(2), False, False),
    ]


def test_even_multiplicity_root_does_not_flip_sign():
    nonnegative = kw.solve_inequality("(x-1)^2>=0")
    positive = kw.solve_inequality("(x-1)^2>0")
    nonpositive = kw.solve_inequality("(x-1)^2<=0")

    assert isinstance(nonnegative.solution_set, kw.IntervalSolutionSet)
    assert nonnegative.solution_set.lower is None and nonnegative.solution_set.upper is None
    assert interval_signature(positive.solution_set) == [
        (None, 1.0, False, False), (1.0, None, False, False),
    ]
    assert interval_signature(nonpositive.solution_set) == [("points", (1.0,))]


def test_rational_inequality_excludes_poles_and_propagates_signs():
    result = kw.solve_inequality("1/(x+2)<0")

    assert interval_signature(result.solution_set) == [
        (None, -2.0, False, False),
    ]
    assert any("!= 0" in condition for condition in result.conditions)


def test_cancelled_rational_factor_remains_a_hole():
    result = kw.solve_inequality("(x^2-1)/(x-1)>0")

    assert interval_signature(result.solution_set) == [
        (-1.0, 1.0, False, False),
        (1.0, None, False, False),
    ]


def test_partially_cancelled_zero_at_a_hole_stays_excluded():
    result = kw.solve_inequality("(x-1)^2/(x-1)>=0")

    assert interval_signature(result.solution_set) == [
        (1.0, None, False, False),
    ]


def test_not_equal_returns_the_complement_of_zeros_and_holes():
    result = kw.solve_inequality("(x-1)/(x+2)!=0")

    assert interval_signature(result.solution_set) == [
        (None, -2.0, False, False),
        (-2.0, 1.0, False, False),
        (1.0, None, False, False),
    ]


def test_constant_inequalities_with_explicit_variable():
    true_result = kw.solve_inequality("2>1", variable="x")
    false_result = kw.solve_inequality("2<1", variable="x")
    unequal = kw.solve_inequality("0!=0", variable="x")

    assert isinstance(true_result.solution_set, kw.IntervalSolutionSet)
    assert true_result.solution_set.lower is None and true_result.solution_set.upper is None
    assert isinstance(false_result.solution_set, kw.EmptySolutionSet)
    assert isinstance(unequal.solution_set, kw.EmptySolutionSet)


def test_parameter_substitution_enables_an_exact_inequality():
    result = kw.solve_inequality(
        "a*x^2-1>=0", variable="x", assumptions={"a": 1},
    )

    assert interval_signature(result.solution_set) == [
        (None, -1.0, False, True),
        (1.0, None, True, False),
    ]


def test_symbolic_coefficients_and_multivariable_forms_fail_closed():
    symbolic = kw.solve_inequality("a*x^2-1>=0", variable="x")
    multivariable = kw.solve_inequality("x^2+y^2<=1", variable="x")

    assert symbolic.status == "unresolved" and not symbolic.complete
    assert multivariable.status == "unresolved" and not multivariable.complete
    assert isinstance(symbolic.solution_set, kw.EmptySolutionSet)


@pytest.mark.parametrize("text", ("x=1", "x<y<2", "<1", "x>"))
def test_inequality_parser_rejects_invalid_relations(text):
    with pytest.raises((kw.EquationParseError, ValueError)):
        kw.solve_inequality(text, variable="x")


def test_inequality_rejects_complex_domain_and_invalid_steps():
    with pytest.raises(ValueError, match="real domain"):
        kw.solve_inequality("x>0", domain="complex")
    with pytest.raises(TypeError, match="steps"):
        kw.solve_inequality("x>0", steps="yes")


def test_inequality_validates_input_and_accepts_named_variable_objects():
    class NamedVariable:
        name = "x"

    result = kw.solve_inequality("x>=0", variable=NamedVariable())
    assert interval_signature(result.solution_set) == [
        (0.0, None, True, False),
    ]

    with pytest.raises(TypeError, match="string"):
        kw.solve_inequality(("x", "0"), variable="x")
    with pytest.raises(kw.AmbiguousVariableError, match="Specify variable"):
        kw.solve_inequality("2>1")
    with pytest.raises(TypeError, match="non-empty"):
        kw.solve_inequality("x>0", variable="")


def test_inequality_result_and_unbounded_intervals_serialize_deterministically():
    result = kw.solve_inequality("(x^2-1)/(x-1)>0", steps=True)
    restored = kw.EquationSolution.from_dict(result.to_dict())

    assert restored == result
    assert restored.to_dict() == result.to_dict()
    from kiwicalc.serialization import object_from_dict, object_to_dict
    assert object_from_dict(object_to_dict(result)) == result
    assert str(kw.IntervalSolutionSet(None, 2, False, True)) == "(-inf, 2]"


def test_contradictory_assumptions_return_an_exact_empty_set():
    result = kw.solve_inequality(
        "x^2>=0", assumptions=("x>0", "x<0"),
    )

    assert isinstance(result.solution_set, kw.EmptySolutionSet)
    assert result.exact and result.complete
