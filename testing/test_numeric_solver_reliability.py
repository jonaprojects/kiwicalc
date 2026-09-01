import warnings

import pytest

from kiwicalc.numeric import roots


def test_iterative_solvers_accept_an_exact_initial_root():
    function = lambda x: x**2 - 4

    assert roots.newton_raphson(function, lambda x: 2 * x, 2) == 2
    assert roots.halleys_method(function, lambda x: 2 * x, lambda _x: 2, 2) == 2
    assert roots.secant_method(function, 2, 3) == 2
    assert roots.secant_method(function, 3, -2) == -2
    assert roots.steffensen_method(function, 2) == 2


@pytest.mark.parametrize(
    ("solver", "args"),
    [
        (roots.halleys_method, (lambda x: x + 1, lambda _x: 1, lambda _x: 0, 10)),
        (roots.secant_method, (lambda x: x**2 + 1, 0, 1)),
        (roots.steffensen_method, (lambda x: x + 1, 10)),
    ],
)
def test_iterative_solvers_return_last_estimate_and_warn_on_exhaustion(solver, args):
    with pytest.warns(UserWarning, match="not converged"):
        result = solver(*args, nmax=0)

    assert result is not None


def test_secant_honors_iteration_limit():
    calls = 0

    def function(x):
        nonlocal calls
        calls += 1
        return x**2 + 1

    with pytest.warns(UserWarning, match="not converged"):
        roots.secant_method(function, 0, 1, epsilon=0, nmax=1)

    assert calls == 3


def test_steffensen_honors_iteration_limit():
    calls = 0

    def function(x):
        nonlocal calls
        calls += 1
        return x + 1

    with pytest.warns(UserWarning, match="not converged"):
        roots.steffensen_method(function, 10, epsilon=0, nmax=1)

    assert calls == 2


def test_secant_equal_function_values_fail_cleanly():
    with pytest.warns(UserWarning, match="function values are equal"):
        result = roots.secant_method(lambda x: x**2 + 1, -1, 1)

    assert result == -1


@pytest.mark.parametrize(("bounds", "expected"), [((2, 5), 2), ((0, 2), 2), ((5, 2), 2)])
def test_bisection_accepts_roots_at_either_endpoint(bounds, expected):
    assert roots.bisection_method(lambda x: x - 2, *bounds) == expected


def test_bisection_reuses_cached_endpoint_evaluation():
    calls = []

    def function(x):
        calls.append(x)
        return x - 0.25

    assert roots.bisection_method(function, 0, 1) == pytest.approx(0.25)
    assert calls.count(0) == 1


def test_multi_root_solver_does_not_mutate_coefficients():
    coefficients = [2, 0, -8]

    solutions = roots.durand_kerner(lambda x: 2 * x**2 - 8, coefficients)

    assert coefficients == [2, 0, -8]
    assert solutions == {complex(-2), complex(2)}
