import pytest

import kiwicalc as kw


def numeric(value, values=None):
    if value is None:
        return None
    if isinstance(value, kw.SymbolicExpression):
        value = value.evaluate(values or {})
    return complex(value).real


def parts(solution_set):
    return solution_set.sets if isinstance(solution_set, kw.UnionSolutionSet) else (solution_set,)


def signature(solution_set, values=None):
    if isinstance(solution_set, kw.EmptySolutionSet):
        return []
    if isinstance(solution_set, kw.UniversalSolutionSet):
        return [(None, None, False, False)]
    result = []
    for part in parts(solution_set):
        if isinstance(part, kw.FiniteSolutionSet):
            result.append(("points", tuple(numeric(value, values) for value in part.values)))
        else:
            result.append((
                numeric(part.lower, values), numeric(part.upper, values),
                part.lower_closed, part.upper_closed,
            ))
    return result


def applicable_parts(solution_set, values):
    if isinstance(solution_set, kw.ConditionalSolutionSet):
        applies = all(
            kw.parse_condition(condition).evaluate(values) is True
            for condition in solution_set.conditions
        )
        return applicable_parts(solution_set.solution_set, values) if applies else []
    if isinstance(solution_set, kw.UnionSolutionSet):
        return [
            part
            for subset in solution_set.sets
            for part in applicable_parts(subset, values)
        ]
    return [] if isinstance(solution_set, kw.EmptySolutionSet) else [solution_set]


def conditional_signature(solution_set, values):
    selected = applicable_parts(solution_set, values)
    if not selected:
        return []
    specialized = selected[0] if len(selected) == 1 else kw.UnionSolutionSet(tuple(selected))
    return signature(specialized, values)


def collect_conditions(solution_set):
    if isinstance(solution_set, kw.ConditionalSolutionSet):
        return set(solution_set.conditions) | collect_conditions(solution_set.solution_set)
    if isinstance(solution_set, kw.UnionSolutionSet):
        return set().union(*(collect_conditions(part) for part in solution_set.sets))
    return set()


def test_general_parameterized_linear_inequality_is_complete_and_conditional():
    result = kw.solve_inequality("a*x+b>0", variable="x", steps=True)

    assert result.status == "solved" and result.exact and result.complete
    assert {"a > 0", "a < 0", "a = 0", "b > 0"} <= collect_conditions(
        result.solution_set
    )
    assert [step.rule for step in result.steps] == [
        "normalize_inequality",
        "parameterized_inequality_reduction",
        "rational_sign_chart",
    ]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({"a": 2, "b": -4}, [(2.0, None, False, False)]),
        ({"a": -2, "b": -4}, [(None, -2.0, False, False)]),
        ({"a": 0, "b": 1}, [(None, None, False, False)]),
        ({"a": 0, "b": -1}, []),
    ),
)
def test_parameter_values_reduce_linear_inequality(parameters, expected):
    general = kw.solve_inequality("a*x+b>0", variable="x")
    result = kw.solve_inequality(
        "a*x+b>0", variable="x", assumptions=parameters,
    )

    assert result.status == "solved" and result.complete
    assert signature(result.solution_set) == expected
    assert conditional_signature(general.solution_set, parameters) == expected


def test_sign_assumption_prunes_linear_branches_without_approximating_endpoint():
    result = kw.solve_inequality(
        "a*x-2>0", variable="x", assumptions="a>0",
    )

    assert isinstance(result.solution_set, kw.IntervalSolutionSet)
    assert str(result.solution_set.lower) == "2*a^-1"
    assert result.solution_set.upper is None
    assert result.conditions == ("a>0",)


@pytest.mark.parametrize("operator", ("<", "<=", ">", ">=", "!="))
def test_every_relation_matches_direct_linear_substitution(operator):
    parameterized = kw.solve_inequality(
        f"a*x+b{operator}0", variable="x", assumptions={"a": -2, "b": 1},
    )
    direct = kw.solve_inequality(f"-2*x+1{operator}0", variable="x")

    assert parameterized.solution_set == direct.solution_set


def test_general_parameterized_quadratic_partitions_degree_and_discriminant():
    result = kw.solve_inequality(
        "a*x^2+b*x+c>=0", variable="x", steps=True,
    )

    conditions = collect_conditions(result.solution_set)
    assert result.status == "solved" and result.exact and result.complete
    assert {"a > 0", "a < 0", "a = 0", "b > 0", "b < 0", "b = 0"} <= conditions
    assert any("b^2" in condition and "> 0" in condition for condition in conditions)
    assert any("b^2" in condition and "= 0" in condition for condition in conditions)
    assert any("b^2" in condition and "< 0" in condition for condition in conditions)


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({"a": 1, "b": -5, "c": 6}, [
            (None, 2.0, False, True), (3.0, None, True, False),
        ]),
        ({"a": -1, "b": 0, "c": 1}, [(-1.0, 1.0, True, True)]),
        ({"a": 0, "b": 2, "c": -4}, [(2.0, None, True, False)]),
    ),
)
def test_parameter_values_reduce_quadratic_and_degenerate_cases(parameters, expected):
    general = kw.solve_inequality("a*x^2+b*x+c>=0", variable="x")
    result = kw.solve_inequality(
        "a*x^2+b*x+c>=0", variable="x", assumptions=parameters,
    )

    assert result.status == "solved" and result.complete
    assert signature(result.solution_set) == expected
    assert conditional_signature(general.solution_set, parameters) == expected


def test_quadratic_repeated_root_and_no_real_root_cases_are_exact():
    repeated = kw.solve_inequality(
        "a*x^2+b*x+c>0", variable="x",
        assumptions={"a": 1, "b": -2, "c": 1},
    )
    always = kw.solve_inequality(
        "a*x^2+b*x+c>0", variable="x",
        assumptions={"a": 1, "b": 0, "c": 1},
    )
    never = kw.solve_inequality(
        "a*x^2+b*x+c<0", variable="x",
        assumptions={"a": 1, "b": 0, "c": 1},
    )

    assert signature(repeated.solution_set) == [
        (None, 1.0, False, False), (1.0, None, False, False),
    ]
    assert signature(always.solution_set) == [(None, None, False, False)]
    assert isinstance(never.solution_set, kw.EmptySolutionSet)


def test_discriminant_assumption_selects_a_quadratic_branch():
    result = kw.solve_inequality(
        "a*x^2+b*x+c>0", variable="x",
        assumptions=("a>0", "b^2-4*a*c<0"),
    )

    assert isinstance(result.solution_set, kw.UniversalSolutionSet)
    assert result.complete


@pytest.mark.parametrize("operator", ("<", "<=", ">", ">=", "!="))
def test_every_relation_matches_direct_quadratic_substitution(operator):
    parameterized = kw.solve_inequality(
        f"a*x^2+b*x+c{operator}0", variable="x",
        assumptions={"a": -1, "b": 2, "c": 3},
    )
    direct = kw.solve_inequality(
        f"-x^2+2*x+3{operator}0", variable="x",
    )

    assert parameterized.solution_set == direct.solution_set


def test_parameter_scaled_rational_inequality_preserves_sign_and_domain():
    general = kw.solve_inequality(
        "a*(x-1)/(x+2)>0", variable="x", steps=True,
    )
    positive = kw.solve_inequality(
        "a*(x-1)/(x+2)>0", variable="x", assumptions={"a": 1},
    )
    negative = kw.solve_inequality(
        "a*(x-1)/(x+2)>0", variable="x", assumptions={"a": -1},
    )

    assert {"a > 0", "a < 0"} <= collect_conditions(general.solution_set)
    assert signature(positive.solution_set) == [
        (None, -2.0, False, False), (1.0, None, False, False),
    ]
    assert signature(negative.solution_set) == [(-2.0, 1.0, False, False)]
    assert "x + 2 != 0" in general.conditions


def test_zero_parameter_keeps_rational_domain_hole_for_nonstrict_relation():
    result = kw.solve_inequality(
        "a*(x-1)/(x+2)>=0", variable="x", assumptions={"a": 0},
    )

    assert signature(result.solution_set) == [
        (None, -2.0, False, False), (-2.0, None, False, False),
    ]
    assert "x + 2 != 0" in result.conditions


def test_parameter_only_inequality_is_a_conditional_universal_set():
    result = kw.solve_inequality("a>0", variable="x")

    assert isinstance(result.solution_set, kw.ConditionalSolutionSet)
    assert isinstance(result.solution_set.solution_set, kw.UniversalSolutionSet)
    assert result.solution_set.conditions == ("a > 0",)


def test_unsupported_parameterized_forms_fail_closed():
    cubic = kw.solve_inequality("a*x^3+x+1>0", variable="x")
    moving_pole = kw.solve_inequality("(a*x+1)/(x-2)>0", variable="x")
    parameterized_absolute = kw.solve_inequality("abs(x)<=a", variable="x")

    for result in (cubic, moving_pole, parameterized_absolute):
        assert result.status == "unresolved" and not result.complete


def test_parameterized_inequality_is_deterministic_and_serializable():
    first = kw.solve_inequality("a*x^2+b*x+c<=0", variable="x", steps=True)
    second = kw.solve_inequality("a*x^2+b*x+c<=0", variable="x", steps=True)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert kw.EquationSolution.from_dict(first.to_dict()) == first
