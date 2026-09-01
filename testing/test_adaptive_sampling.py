import math

import numpy as np
import pytest

import kiwicalc as kw


def test_fixed_sampling_remains_the_default():
    curve = kw.ParametricCurve2D("t", "t^2", samples=37)
    assert curve.sampling == "fixed"
    assert len(curve.sample()[0]) == 37


def test_adaptive_sampling_uses_few_points_for_a_line():
    curve = kw.ParametricCurve2D("t", "2*t+1", t_range=(-10, 10), sampling="adaptive")
    x, y = curve.sample()
    assert len(x) == len(y) == 17
    assert y == pytest.approx(2 * x + 1)


def test_adaptive_sampling_refines_curved_sections():
    loose = kw.ParametricCurve2D("t", "sin(t)", t_range=(0, 2 * math.pi), sampling="adaptive", tolerance=0.1)
    precise = loose.adaptive(tolerance=1e-5, max_depth=12)
    assert len(precise.sample()[0]) > len(loose.sample()[0])


def test_explicit_sample_count_overrides_adaptive_mode():
    curve = kw.ParametricCurve2D("cos(t)", "sin(t)", sampling="adaptive")
    assert len(curve.sample(samples=23)[0]) == 23


def test_adaptive_polar_and_3d_curves():
    polar = kw.PolarCurve2D("1+0.2*cos(5*theta)", sampling="adaptive", tolerance=1e-3)
    assert len(polar.sample()[0]) > 17

    curve3d = kw.ParametricCurve3D("cos(t)", "sin(t)", "t", sampling="adaptive", tolerance=1e-3)
    axes = curve3d.sample()
    assert len(axes[0]) == len(axes[1]) == len(axes[2])
    assert len(axes[0]) > 17
    assert curve3d.adaptive(tolerance=0.1).sampling == "adaptive"


def test_adaptive_sampling_preserves_domain_gaps():
    curve = kw.ParametricCurve2D("t", lambda t: math.sqrt(t), t_range=(-1, 1), sampling="adaptive", max_depth=5)
    x, y = curve.sample()
    assert np.isnan(y).any()
    assert np.isfinite(y).any()
    assert len(x) < 1000


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kw.ParametricCurve2D("t", "t", sampling="unknown"),
        lambda: kw.ParametricCurve2D("t", "t", sampling="adaptive", tolerance=0),
        lambda: kw.ParametricCurve2D("t", "t", sampling="adaptive", max_depth=0),
        lambda: kw.ParametricCurve3D("t", "t", "t", sampling="adaptive", tolerance=float("inf")),
    ],
)
def test_invalid_adaptive_options_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
