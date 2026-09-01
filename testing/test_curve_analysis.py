import math

import pytest

import kiwicalc as kw


def test_point_tangent_normal_and_bounds_in_2d():
    line = kw.ParametricCurve2D("2*t", "t", t_range=(0, 1), samples=101)
    point = line.point_at(0.5)
    assert (point.x, point.y) == pytest.approx((1, 0.5))
    tangent = line.tangent_at(0.5)
    assert tangent == pytest.approx((2 / math.sqrt(5), 1 / math.sqrt(5)))
    assert line.normal_at(0.5) == pytest.approx((-tangent[1], tangent[0]))
    assert line.bounds[0] == pytest.approx((0, 2))
    assert line.bounds[1] == pytest.approx((0, 1))
    assert line.arc_length() == pytest.approx(math.sqrt(5), rel=1e-4)


def test_circle_curvature_and_length_are_numerically_accurate():
    circle = kw.Ellipse(2, 2, samples=2001)
    assert circle.curvature_at(0.25) == pytest.approx(0.5, rel=2e-3)
    assert circle.arc_length() == pytest.approx(4 * math.pi, rel=2e-4)


def test_2d_curve_intersections_return_points_and_deduplicate_vertices():
    first = kw.ParametricCurve2D("t", "t", t_range=(-1, 1), samples=101)
    second = kw.ParametricCurve2D("t", "-t", t_range=(-1, 1), samples=101)
    intersections = first.intersections(second)
    assert len(intersections) == 1
    assert (intersections[0].x, intersections[0].y) == pytest.approx((0, 0), abs=1e-9)

    parallel = first.translate(0, 1)
    assert first.intersections(parallel) == []


def test_analysis_works_after_transformations():
    line = kw.ParametricCurve2D("t", "0", t_range=(0, 1), samples=51).rotate(math.pi / 2).translate(2, 3)
    assert tuple(line.point_at(0.5).coordinates) == pytest.approx((2, 3.5))
    assert line.tangent_at(0.5) == pytest.approx((0, 1), abs=1e-10)


def test_3d_analysis_on_line_and_helix():
    line = kw.Line3D(point=(1, 2, 3), direction=(2, 0, 0), t_range=(0, 1), samples=101)
    assert tuple(line.point_at(0.5).coordinates) == pytest.approx((2, 2, 3))
    assert line.tangent_at(0.5) == pytest.approx((1, 0, 0))
    assert line.arc_length() == pytest.approx(2)
    assert line.curvature_at(0.5) == pytest.approx(0, abs=1e-10)
    assert line.bounds[0] == pytest.approx((1, 3))
    assert line.bounds[1] == pytest.approx((2, 2))
    assert line.bounds[2] == pytest.approx((3, 3))
    with pytest.raises(ValueError, match="normal"):
        line.normal_at(0.5)

    helix = kw.Helix(radius=2, pitch=1, turns=2, samples=2001)
    tangent = helix.tangent_at(0.25)
    normal = helix.normal_at(0.25)
    assert sum(a * b for a, b in zip(tangent, normal)) == pytest.approx(0, abs=1e-5)
    assert helix.curvature_at(0.25) > 0


@pytest.mark.parametrize("position", [-0.1, 1.1])
def test_analysis_rejects_positions_outside_normalized_range(position):
    with pytest.raises(ValueError, match="between 0 and 1"):
        kw.Ellipse().point_at(position)


def test_analysis_handles_gaps_and_invalid_intersection_inputs():
    hyperbola = kw.Hyperbola(samples=101)
    with pytest.raises(ValueError, match="gap"):
        hyperbola.point_at(0.5)
    with pytest.raises(TypeError):
        kw.Ellipse().intersections(object())
    with pytest.raises(ValueError, match="positive"):
        kw.Ellipse().intersections(kw.Ellipse(), tolerance=0)
