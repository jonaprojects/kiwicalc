import math

import pytest

import kiwicalc as kw


def _roots(equation, interval, **kwargs):
    result = kw.solve_equation(
        equation, method="numeric", interval=interval, tolerance=1e-10,
        **kwargs,
    )
    assert result.status == "solved"
    assert result.method == "numeric"
    assert result.exact is False and result.complete is False
    return result


def test_adaptive_isolation_uses_less_work_for_a_smooth_crossing():
    result = _roots("cos(x)=x", (0, 1), steps=True)
    assert result.solutions == pytest.approx((0.7390851332151607,))
    assert result.evaluations < 257  # the former fixed grid's minimum
    assert result.steps[-1].rule == "numeric_isolation"
    assert "Adaptively" in result.steps[-1].explanation
    assert "adaptive isolation" in result.message


@pytest.mark.parametrize(
    "equation, interval, expected",
    (
        ("(x-0.1234567)^2=0", (-1, 1), (0.1234567,)),
        ("1e12*(x-0.3333333)^2=0", (0, 1), (0.3333333,)),
        ("((x-0.2)*(x-0.7))^2=0", (0, 1), (0.2, 0.7)),
    ),
)
def test_adaptive_minimum_refinement_finds_even_and_tangent_roots(
    equation, interval, expected,
):
    result = _roots(equation, interval)
    assert result.solutions == pytest.approx(expected, abs=2e-8)
    assert max(result.residuals, default=0) <= 1e-8


def test_disjoint_evidence_brackets_preserve_nearby_distinct_roots():
    result = _roots("(x-0.1)*(x-0.102)=0", (0.09, 0.11))
    assert result.solutions == pytest.approx((0.1, 0.102), abs=1e-9)


@pytest.mark.parametrize(
    "equation, interval",
    (
        ("1/(x-0.1)=0", (-1, 1)),
        ("exp(x)=0", (-30, -20)),
        ("1/(x-0.1)^2=0", (-1, 1)),
    ),
)
def test_poles_and_small_asymptotic_tails_are_not_reported_as_roots(
    equation, interval,
):
    result = _roots(equation, interval)
    assert isinstance(result.solution_set, kw.EmptySolutionSet)
    assert result.residuals == ()


def test_adaptive_isolation_separates_tangent_poles_from_crossing_roots():
    result = _roots("tan(x)=0", (-4, 4))
    assert result.solutions == pytest.approx((-math.pi, 0, math.pi), abs=1e-9)
    assert all(abs(abs(root) - math.pi / 2) > 0.1 for root in result.solutions)


def test_adaptive_curvature_subdivision_resolves_oscillation_and_endpoints():
    result = _roots("sin(20*x)=0", (-math.pi, math.pi))
    expected = tuple(index * math.pi / 20 for index in range(-20, 21))
    assert result.solutions == pytest.approx(expected, abs=2e-9)
    assert result.solutions[0] == -math.pi
    assert result.solutions[-1] == math.pi


def test_undefined_regions_do_not_exhaust_the_search_for_a_valid_root():
    result = _roots("sqrt(x)=0.5", (-1, 1))
    assert result.solutions == pytest.approx((0.25,))
    assert result.evaluations < 300


def test_adaptive_isolation_is_deterministic_and_evaluation_bounded():
    first = _roots("cos(x)=x", (0, 1))
    second = _roots("cos(x)=x", (0, 1))
    assert first.solution_set == second.solution_set
    assert first.residuals == second.residuals
    assert first.evaluations == second.evaluations

    limited = _roots("cos(x)=x", (0, 1), max_iterations=1)
    assert limited.evaluations <= 257


def test_interval_endpoint_roots_are_retained():
    result = _roots("(x+1)*(x-1)=0", (-1, 1))
    assert result.solutions == (-1.0, 1.0)


def test_adaptive_numeric_candidates_still_obey_structural_assumptions():
    result = kw.solve_equation_assuming(
        "x^2=1", "x > 0", method="numeric", interval=(-2, 2),
    )
    assert result.solutions == pytest.approx((1.0,))
