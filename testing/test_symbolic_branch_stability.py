from fractions import Fraction
import math

import numpy as np
import pytest

import kiwicalc as kw


def test_symbolic_primitives_validate_and_convert():
    half = kw.ExactNumber(1, 2)
    assert float(half) == 0.5 and complex(half) == 0.5
    for name in ("", "2x", None):
        with pytest.raises(ValueError):
            kw.Symbol(name)
    with pytest.raises(ValueError):
        kw.SymbolicConstant("tau")
    with pytest.raises(TypeError):
        kw.Add(())
    with pytest.raises(TypeError):
        kw.Multiply((kw.Symbol("x"), 2))
    with pytest.raises(TypeError):
        kw.Power(2, kw.ExactNumber(2))
    with pytest.raises(ValueError):
        kw.SymbolicFunction("gamma", (kw.Symbol("x"),))
    with pytest.raises(TypeError):
        kw.SymbolicFunction("sin", ())
    with pytest.raises(ValueError):
        kw.SymbolicFunction("sin", (kw.Symbol("x"), kw.Symbol("y")))


@pytest.mark.parametrize(
    "arguments",
    (
        ((kw.ExactNumber(1),), 0, None),
        ((kw.ExactNumber(0), kw.ExactNumber(1)), 0, None),
        ((kw.ExactNumber(1), kw.ExactNumber(0)), True, None),
        ((kw.ExactNumber(1), kw.ExactNumber(0)), 1, None),
        ((kw.ExactNumber(1), kw.ExactNumber(0)), 0, (2, 1)),
    ),
)
def test_rootof_rejects_invalid_construction(arguments):
    with pytest.raises((TypeError, ValueError)):
        kw.RootOf(*arguments)


def test_expression_evaluation_substitution_and_supported_derivatives():
    expression = kw.parse_symbolic("exp(x)+ln(x)+sin(x)+cos(x)+tan(x)+sqrt(x)+abs(x)")
    assert expression.variables == {"x"}
    assert math.isfinite(complex(expression.evaluate({"x": 2})).real)
    assert not expression.substitute({"x": 2}).variables
    for source in ("3", "pi", "y", "x^3", "x*y", "sin(x)", "cos(x)", "tan(x)", "exp(x)", "ln(x)", "sqrt(x)"):
        derivative = kw.differentiate_symbolic(source, "x")
        assert isinstance(derivative, kw.SymbolicExpression)
    with pytest.raises(kw.UnsupportedExpressionError):
        kw.differentiate_symbolic("asin(x)", "x")
    with pytest.raises(kw.UnsupportedExpressionError):
        kw.differentiate_symbolic("x^x", "x")


def test_symbolic_adapters_cover_numbers_and_errors():
    assert kw.to_symbolic(Fraction(2, 3)) == kw.ExactNumber(2, 3)
    assert kw.to_symbolic(np.float64(0.25)) == kw.ExactNumber(1, 4)
    with pytest.raises(TypeError):
        kw.to_symbolic(True)
    with pytest.raises(TypeError):
        kw.to_symbolic(object())
    with pytest.raises(TypeError):
        kw.to_symbolic(float("nan"))
    with pytest.raises(kw.UnsupportedExpressionError, match="multi-character"):
        kw.to_legacy_expression("alpha+1")


@pytest.mark.parametrize(
    "source",
    (None, "", "|x", "log(x,2,3)", "sin(x,2)", "foo(x)", ")", "x+"),
)
def test_parser_rejects_all_malformed_categories(source):
    with pytest.raises((TypeError, kw.EquationParseError, kw.UnsupportedExpressionError)):
        kw.parse_symbolic(source)


def test_solution_set_and_result_invariants():
    with pytest.raises(ValueError):
        kw.UniversalSolutionSet("integer")
    with pytest.raises(TypeError):
        kw.FiniteSolutionSet(([1],))
    with pytest.raises(TypeError):
        kw.FiniteSolutionSet((float("nan"),))
    for multiplicities in ((0,), (True,), (1, 2)):
        with pytest.raises(ValueError):
            kw.FiniteSolutionSet((1,), multiplicities)
    with pytest.raises(TypeError):
        kw.IntervalSolutionSet(0, 1, lower_closed="yes")
    with pytest.raises(ValueError):
        kw.IntervalSolutionSet(2, 1)
    symbol = kw.Symbol("k")
    for arguments in (("", symbol), ("x", symbol, "x"), ("x", 1), ("x", symbol, "k", "real")):
        with pytest.raises((TypeError, ValueError)):
            kw.ParametricSolutionSet(*arguments)
    with pytest.raises(TypeError):
        kw.UnionSolutionSet(())
    with pytest.raises(TypeError):
        kw.ConditionalSolutionSet(1, ("x>0",))
    with pytest.raises(ValueError):
        kw.ConditionalSolutionSet(kw.FiniteSolutionSet((1,)), ())
    with pytest.raises(TypeError):
        kw.SolutionStep(1, "a", "b", "c")

    base = dict(variable="x", solution_set=kw.FiniteSolutionSet((1,)), status="solved", method="symbolic", exact=True, complete=True)
    for update in (
        {"variable": ""}, {"solution_set": 1}, {"status": "bad"}, {"method": "bad"},
        {"exact": 1}, {"evaluations": -1}, {"conditions": ("",)},
        {"residuals": (float("nan"),)}, {"steps": ("bad",)}, {"message": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            kw.EquationSolution(**dict(base, **update))


def test_every_solution_set_serializes_deterministically():
    sets = (
        kw.EmptySolutionSet(), kw.UniversalSolutionSet("complex"),
        kw.FiniteSolutionSet((kw.ExactNumber(1), 2j), (2, 1)),
        kw.IntervalSolutionSet(0, 1, False, True),
        kw.ParametricSolutionSet("x", kw.Symbol("n"), "n"),
        kw.UnionSolutionSet((kw.FiniteSolutionSet((1,)), kw.FiniteSolutionSet((2,)))),
        kw.ConditionalSolutionSet(kw.FiniteSolutionSet((1,)), ("a != 0",)),
    )
    for solution_set in sets:
        result = kw.EquationSolution("x", solution_set, "solved", "symbolic", True, True)
        assert kw.EquationSolution.from_dict(result.to_dict()) == result
    unresolved = kw.EquationSolution("x", kw.EmptySolutionSet(), "unresolved", "symbolic", False, False, message="unsupported")
    assert not unresolved.converged and "unresolved" in str(unresolved)
    step = kw.SolutionStep("rule", "a", "b", "explain", ("c",))
    result = kw.EquationSolution("x", kw.FiniteSolutionSet((1,)), "solved", "symbolic", True, True, steps=(step,))
    assert result.formatted_steps() == ("1. explain  a -> b",)
    assert str(result) == "x in {1}" and result.converged
    with pytest.raises(ValueError, match="Unknown symbolic"):
        kw.symbolic_from_dict({"type": "mystery"})
    malformed = result.to_dict()
    malformed["solution_set"] = {"type": "mystery"}
    with pytest.raises(ValueError, match="Unknown solution"):
        kw.EquationSolution.from_dict(malformed)

    class UnknownExpression(kw.SymbolicExpression):
        pass

    class UnknownSolutionSet(kw.SolutionSet):
        pass

    with pytest.raises(TypeError, match="Unsupported symbolic"):
        UnknownExpression().to_dict()
    with pytest.raises(TypeError, match="Unsupported solution"):
        UnknownSolutionSet().to_dict()

    algebraic = kw.solve_equation("x^5-x+1=0", domain="complex")
    assert kw.EquationSolution.from_dict(algebraic.to_dict()) == algebraic


def test_simplification_and_exact_factory_branches():
    values = (
        kw.Add((kw.ExactNumber(1), kw.Symbol("x"))),
        kw.Multiply((kw.ExactNumber(2), kw.Symbol("x"))),
        kw.Power(kw.Symbol("x"), kw.ExactNumber(2)),
        kw.SymbolicFunction("sin", (kw.Symbol("x"),)),
    )
    assert all(isinstance(kw.simplify_symbolic(value), kw.SymbolicExpression) for value in values)
    assert kw.parse_symbolic("x^0") == kw.ExactNumber(1)
    assert kw.parse_symbolic("sqrt(4)") == kw.ExactNumber(2)
    assert kw.parse_symbolic("+x") == kw.Symbol("x")
    assert kw.parse_symbolic("tan(pi/4)") == kw.ExactNumber(1)
    assert kw.parse_symbolic("tan(pi/3)").evaluate({}) == pytest.approx(math.sqrt(3))
    assert kw.parse_symbolic("tan(pi/2)").__class__ is kw.SymbolicFunction
    assert kw.parse_symbolic("cos(pi/3)") == kw.ExactNumber(1, 2)


def test_domain_guards_recurse_through_all_container_types():
    cases = (
        ("0*exp(1/x)=0", "x != 0"),
        ("0*(1+1/x)=0", "x != 0"),
        ("0*(x/x)=0", "x != 0"),
        ("tan(x)=tan(x)", "cos(x) != 0"),
        ("asin(x)=asin(x)", "x >= -1"),
        ("x^(1/2)=x^(1/2)", "x >= 0"),
        ("0*(x^x)=0", "x^x is defined in the real domain"),
    )
    for equation, condition in cases:
        result = kw.solve_equation(equation, variable="x")
        assert result.complete and condition in result.conditions


@pytest.mark.parametrize(
    ("equation", "expected"),
    (("1^x=1", "universal"), ("1^x=2", "empty"), ("2^x=-1", "empty"), ("0^x=0", "unresolved")),
)
def test_exponential_edge_cases_are_honest(equation, expected):
    result = kw.solve_equation(equation, variable="x")
    if expected == "universal":
        assert isinstance(result.solution_set, kw.UniversalSolutionSet)
    elif expected == "empty":
        assert isinstance(result.solution_set, kw.EmptySolutionSet) and result.complete
    else:
        assert result.status == "unresolved" and not result.complete


def test_symbolic_function_edge_branches_remain_complete_or_honestly_unresolved():
    assert isinstance(kw.solve_equation("exp(x)=0").solution_set, kw.EmptySolutionSet)
    assert isinstance(kw.solve_equation("sqrt(x)=-1").solution_set, kw.EmptySolutionSet)
    assert kw.solve_equation("abs(x)=0").solutions == (kw.ExactNumber(0),)
    assert kw.solve_equation("2^x=2").solutions == (kw.ExactNumber(1),)
    assert kw.solve_equation("2^x=3").complete
    for equation in ("exp(x^2)=2", "abs(x^2)=1"):
        result = kw.solve_equation(equation, variable="x", method="symbolic")
        assert result.status == "unresolved" and not result.complete
    nested = kw.solve_equation("sqrt(sin(x))=1", variable="x", method="symbolic")
    assert nested.status == "solved" and nested.complete
    assert isinstance(nested.solution_set, kw.ParametricSolutionSet)


@pytest.mark.parametrize(
    "equation",
    ("sin(x)=1/3", "cos(x)=1/3", "tan(x)=1/3", "sin(2*x+1)=-sqrt(3)/2"),
)
def test_generic_affine_trigonometric_families(equation):
    result = kw.solve_equation(equation)
    assert result.complete and isinstance(result.solution_set, (kw.ParametricSolutionSet, kw.UnionSolutionSet))


def test_constant_and_input_contract_branches():
    assert isinstance(kw.solve_equation("1=1", variable="z", interval=(-1, 1)).solution_set, kw.IntervalSolutionSet)
    assert isinstance(kw.solve_equation("sqrt(-1)=sqrt(-1)", variable="x").solution_set, kw.EmptySolutionSet)
    with pytest.raises(kw.AmbiguousVariableError):
        kw.solve_equation("x+y=1")
    with pytest.raises(TypeError):
        kw.solve_equation((object(), 1), variable="x")
    with pytest.raises(TypeError):
        kw.solve_equation([1, 2, 3], variable="x")
    with pytest.raises(kw.EquationParseError):
        kw.solve_equation("=1", variable="x")
    with pytest.raises(kw.EquationParseError):
        kw.solve_equation("x", variable="x")
    with pytest.raises(kw.UnsupportedExpressionError, match="degree"):
        kw.solve_equation("x^101=1")
    assert kw.solve_equation("x^3=0").solutions == (kw.ExactNumber(0),)


@pytest.mark.parametrize(
    "interval",
    ("0,1", (0,), (0, 1, 2), (0, float("inf")), (1, 1)),
)
def test_interval_validation_categories(interval):
    with pytest.raises((TypeError, ValueError)):
        kw.solve_equation("x=1", interval=interval)


def test_numerical_fallback_endpoints_no_roots_and_steps():
    endpoint = kw.solve_equation("cos(x)=x", method="numeric", interval=(0, 1), steps=True)
    assert endpoint.steps[-1].rule == "numeric_isolation"
    none = kw.solve_equation("exp(x)=0", method="numeric", interval=(-1, 1))
    assert isinstance(none.solution_set, kw.EmptySolutionSet) and not none.complete
    disabled = kw.solve_equation("cos(x)=x", numeric_fallback=False)
    assert disabled.status == "unresolved"


def test_system_validation_serialization_and_failure_modes():
    with pytest.raises(ValueError, match="At least one"):
        kw.solve_equation_system(())
    with pytest.raises(ValueError, match="domain"):
        kw.solve_equation_system(("x=1",), domain="integer")
    with pytest.raises(ValueError, match="real domain"):
        kw.solve_equation_system(("x=1",), domain="complex", numeric_fallback=True)
    for variables in (("x", "x"), ("",), "x"):
        with pytest.raises(ValueError):
            kw.solve_equation_system(("x=1",), variables=variables)
    with pytest.raises(ValueError, match="initial values"):
        kw.solve_equation_system(("sin(x)+y=1", "2*x+y=1"), numeric_fallback=True)
    with pytest.raises(ValueError, match="initial keys"):
        kw.solve_equation_system(("sin(x)+y=1", "2*x+y=1"), numeric_fallback=True, initial={"x": 1})

    result = kw.solve_equation_system(("x+y=2",), steps=True)
    restored = kw.EquationSystemSolution.from_dict(result.to_dict())
    assert restored == result and restored.to_dict() == result.to_dict()


def test_system_result_constructor_invariants():
    base = dict(variables=("x",), solutions=({"x": 1},), status="solved", method="symbolic", exact=True, complete=True)
    updates = (
        {"variables": ()}, {"variables": ("x", "x")}, {"solutions": ({"y": 1},)},
        {"status": "bad"}, {"method": "bad"}, {"exact": 1},
        {"parameters": ("t", "t")}, {"residuals": (-1,)},
        {"steps": ("bad",)}, {"message": 1},
    )
    for update in updates:
        with pytest.raises((TypeError, ValueError)):
            kw.EquationSystemSolution(**dict(base, **update))


def test_numeric_system_sequence_initial_and_steps():
    result = kw.solve_equation_system(
        ("sin(x)+y=1", "2*x+y=1"), variables=("x", "y"),
        numeric_fallback=True, initial=(0.2, 0.6), steps=True,
    )
    assert result.status == "solved" and result.steps[-1].rule == "local_newton"
