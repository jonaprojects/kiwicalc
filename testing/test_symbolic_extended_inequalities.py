import pytest

import kiwicalc as kw


def numeric(value):
    if value is None:
        return None
    if isinstance(value, kw.SymbolicExpression):
        value = value.evaluate({})
    return complex(value).real


def parts(solution_set):
    return solution_set.sets if isinstance(solution_set, kw.UnionSolutionSet) else (solution_set,)


def signature(solution_set):
    if isinstance(solution_set, kw.EmptySolutionSet):
        return []
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


@pytest.mark.parametrize(("inequality", "expected"), (
    ("abs(2*x-1)<=3", [(-1.0, 2.0, True, True)]),
    ("abs(x)>2", [(None, -2.0, False, False), (2.0, None, False, False)]),
    ("abs(x)!=2", [
        (None, -2.0, False, False), (-2.0, 2.0, False, False),
        (2.0, None, False, False),
    ]),
    ("2*abs(x-1)+1<=5", [(-1.0, 3.0, True, True)]),
    ("abs(x)<=x+1", [(-0.5, None, True, False)]),
    ("abs(x)<abs(x-2)", [(None, 1.0, False, False)]),
    ("abs(x)<-1", []),
    ("abs(x)<=0", [("points", (0.0,))]),
    ("abs(x)>=-1", [(None, None, False, False)]),
    ("abs(x)!=-1", [(None, None, False, False)]),
))
def test_absolute_value_inequalities(inequality, expected):
    result = kw.solve_inequality(inequality, variable="x", steps=True)

    assert result.exact and result.complete and result.status == "solved"
    assert signature(result.solution_set) == expected
    assert [step.rule for step in result.steps] == [
        "normalize_inequality", "absolute_value_reduction", "rational_sign_chart",
    ]


@pytest.mark.parametrize(("inequality", "expected"), (
    ("sqrt(x+1)>x-1", [(-1.0, 3.0, True, False)]),
    ("sqrt(x+1)<=2", [(-1.0, 3.0, True, True)]),
    ("sqrt(x)<-1", []),
    ("sqrt(x)>=-1", [(0.0, None, True, False)]),
    ("sqrt(x)!=0", [(0.0, None, False, False)]),
    ("sqrt(x)<=0", [("points", (0.0,))]),
    ("5-2*sqrt(x)>=1", [(0.0, 4.0, True, True)]),
    ("sqrt(x+1)<sqrt(3-x)", [(-1.0, 1.0, True, False)]),
    ("sqrt((x-1)/(x+1))<1", [(1.0, None, True, False)]),
))
def test_radical_inequalities_retain_domains(inequality, expected):
    result = kw.solve_inequality(inequality, variable="x", steps=True)

    assert result.exact and result.complete and result.status == "solved"
    assert signature(result.solution_set) == expected
    assert [step.rule for step in result.steps] == [
        "normalize_inequality", "radical_reduction", "rational_sign_chart",
    ]


@pytest.mark.parametrize(("inequality", "expected"), (
    ("x^(1/3)>2", [(8.0, None, False, False)]),
    ("x^(2/3)<=4", [(-8.0, 8.0, True, True)]),
    ("x^(2/3)>=0", [(None, None, False, False)]),
    ("(x-1)^(3/2)<=8", [(1.0, 5.0, True, True)]),
    ("(x-1)^(1/2)>0", [(1.0, None, False, False)]),
    ("2*x^(1/3)+1>5", [(8.0, None, False, False)]),
    ("x^(1/3)>x", [
        (None, -1.0, False, False), (0.0, 1.0, False, False),
    ]),
    ("x^(2/3)<(x-1)^(2/3)", [(None, 0.5, False, False)]),
    ("(x+1)^(1/3)<(2*x)^(1/3)", [(1.0, None, False, False)]),
    ("(x+1)^(1/2)<=(3-x)^(1/2)", [(-1.0, 1.0, True, True)]),
    ("x^(-1/2)>=1/2", [(0.0, 4.0, False, True)]),
    ("x^(-1/3)<-1", [(-1.0, 0.0, False, False)]),
    ("x^(-2/3)>1", [
        (-1.0, 0.0, False, False), (0.0, 1.0, False, False),
    ]),
    ("x^(-1/3)>0", [(0.0, None, False, False)]),
    ("x^(-1/2)!=1/2", [
        (0.0, 4.0, False, False), (4.0, None, False, False),
    ]),
    ("x^(-1/2)>=-1", [(0.0, None, False, False)]),
))
def test_rational_power_inequalities_use_real_parity_rules(inequality, expected):
    result = kw.solve_inequality(inequality, variable="x", steps=True)

    assert result.exact and result.complete and result.status == "solved"
    assert signature(result.solution_set) == expected
    assert [step.rule for step in result.steps] == [
        "normalize_inequality", "rational_power_reduction", "rational_sign_chart",
    ]


def test_odd_denominator_powers_evaluate_to_their_real_values():
    cube_root = kw.parse_symbolic("x^(1/3)")
    inverse_cube_root = kw.parse_symbolic("x^(-1/3)")

    assert cube_root.evaluate({"x": -8}) == pytest.approx(-2)
    assert inverse_cube_root.evaluate({"x": -8}) == pytest.approx(-0.5)


def test_unsupported_multi_radical_and_variable_negative_power_fail_closed():
    multi_radical = kw.solve_inequality(
        "sqrt(x)+sqrt(x+1)>2", variable="x",
    )
    variable_reciprocal = kw.solve_inequality("x^(-1/3)<x", variable="x")
    unlike_powers = kw.solve_inequality("x^(1/3)<x^(1/5)", variable="x")

    assert multi_radical.status == "unresolved" and not multi_radical.complete
    assert variable_reciprocal.status == "unresolved" and not variable_reciprocal.complete
    assert unlike_powers.status == "unresolved" and not unlike_powers.complete


def test_extended_inequality_results_serialize_deterministically():
    result = kw.solve_inequality("sqrt(x+1)>x-1", variable="x", steps=True)

    assert kw.EquationSolution.from_dict(result.to_dict()) == result
    assert result.to_dict() == kw.EquationSolution.from_dict(result.to_dict()).to_dict()
