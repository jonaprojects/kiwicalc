import math

import numpy as np
import pytest

import kiwicalc as kw


def test_2d_transformations_are_chainable_and_non_mutating():
    original = kw.ParametricCurve2D("t", "0", t_range=(0, 1), samples=3)
    transformed = original.translate(2, 3).rotate(math.pi / 2).scale(2)

    assert original.sample()[0] == pytest.approx([0, 0.5, 1])
    x, y = transformed.sample()
    assert x == pytest.approx([-6, -6, -6])
    assert y == pytest.approx([4, 5, 6])
    assert isinstance(transformed, kw.TransformedCurve2D)
    assert transformed.source is original


def test_2d_rotation_scaling_centers_and_reflections():
    point_curve = kw.ParametricCurve2D("1", "2", samples=2)
    x, y = point_curve.rotate(math.pi, center=(1, 1)).sample()
    assert (x[0], y[0]) == pytest.approx((1, 0))

    x, y = point_curve.scale(2, 3, center=(1, 1)).sample()
    assert (x[0], y[0]) == pytest.approx((1, 4))
    assert point_curve.reflect("x").sample()[1][0] == pytest.approx(-2)
    assert point_curve.reflect("y").sample()[0][0] == pytest.approx(-1)
    assert point_curve.reflect("origin").sample()[0][0] == pytest.approx(-1)
    assert point_curve.reflect("y=x").sample()[0][0] == pytest.approx(2)


def test_3d_translation_scaling_rotation_and_reflection():
    curve = kw.ParametricCurve3D("1", "0", "0", samples=2)
    moved = curve.translate(1, 2, 3)
    assert tuple(axis[0] for axis in moved.sample()) == pytest.approx((2, 2, 3))

    scaled = curve.scale(2, 3, 4, center=(1, 1, 1))
    assert tuple(axis[0] for axis in scaled.sample()) == pytest.approx((1, -2, -3))

    rotated = curve.rotate_z(math.pi / 2)
    assert tuple(axis[0] for axis in rotated.sample()) == pytest.approx((0, 1, 0), abs=1e-10)
    assert tuple(axis[0] for axis in curve.rotate_x(math.pi / 2).sample()) == pytest.approx((1, 0, 0))
    assert tuple(axis[0] for axis in curve.rotate_y(math.pi / 2).sample()) == pytest.approx((0, 0, -1), abs=1e-10)
    assert curve.reflect("yz").sample()[0][0] == pytest.approx(-1)
    assert curve.reflect("xy").sample()[2][0] == pytest.approx(0)
    assert curve.reflect("xz").sample()[1][0] == pytest.approx(0)
    assert curve.reflect("origin").sample()[0][0] == pytest.approx(-1)


def test_arbitrary_axis_rotation_preserves_distance_from_axis():
    curve = kw.ParametricCurve3D("1", "0", "0", samples=2)
    rotated = curve.rotate(2 * math.pi / 3, axis=(1, 1, 1))
    assert tuple(axis[0] for axis in rotated.sample()) == pytest.approx((0, 1, 0), abs=1e-10)


@pytest.mark.parametrize(
    "action",
    [
        lambda: kw.Ellipse().reflect("diagonal"),
        lambda: kw.Helix().reflect("bad"),
        lambda: kw.Helix().rotate(1, axis=(0, 0, 0)),
        lambda: kw.Helix().rotate(1, axis=(1, 0)),
        lambda: kw.Helix().scale(2, center=(0, 0)),
        lambda: kw.Ellipse().transform(np.eye(2)),
        lambda: kw.Helix().transform(np.eye(3)),
    ],
)
def test_invalid_transformations_raise_friendly_errors(action):
    with pytest.raises((TypeError, ValueError)):
        action()

