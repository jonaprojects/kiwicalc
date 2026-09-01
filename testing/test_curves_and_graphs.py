import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_parametric_polar_and_implicit_curves_accept_friendly_inputs():
    circle = kw.ParametricCurve2D("cos(t)", "sin(t)", samples=9)
    x, y = circle.sample()
    assert x[0] == pytest.approx(1)
    assert y[0] == pytest.approx(0)

    polar = kw.PolarCurve2D("theta", theta_range=(0, math.pi), samples=5)
    assert polar.theta_range == (0.0, math.pi)
    assert len(polar.sample()[0]) == 5

    implicit = kw.ImplicitCurve2D("x^2 + y^2 = 1", x_range=(-2, 2), y_range=(-2, 2), resolution=9)
    X, Y, Z = implicit.sample()
    assert X.shape == Y.shape == Z.shape == (9, 9)
    assert Z[4, 4] == pytest.approx(-1)
    contour = implicit.plot(show=False, label="circle", colors="red")
    assert contour is not None
    assert any(line.get_label() == "circle" for line in plt.gca().lines)
    function_notation = kw.ImplicitCurve2D("f(x,y)=x^2+y^2-1", resolution=5)
    assert function_notation.sample()[2][2, 2] == pytest.approx(-1)


def test_curve_evaluation_turns_domain_failures_into_gaps():
    curve = kw.ParametricCurve2D(lambda t: t, lambda t: math.sqrt(t), t_range=(-1, 1), samples=5)
    x, y = curve.sample()
    assert np.isnan(y[:2]).all()
    assert np.isfinite(x).all()

    implicit = kw.ImplicitCurve2D(lambda x, y: 1 / x, x_range=(-1, 1), resolution=5)
    assert np.isnan(implicit.sample()[2][:, 2]).all()


def test_bezier_and_catmull_rom_2d_sampling_contracts():
    bezier = kw.BezierCurve2D([(0, 0), (1, 2), (2, 0)], samples=11)
    x, y = bezier.sample()
    assert (x[0], y[0]) == pytest.approx((0, 0))
    assert (x[-1], y[-1]) == pytest.approx((2, 0))
    assert bezier.control_points.shape == (3, 2)

    spline = kw.CatmullRomSpline2D([(0, 0), (1, 2), (2, 0)], samples=12)
    sx, sy = spline.sample(samples=9)
    assert (sx[0], sy[0]) == pytest.approx((0, 0))
    assert (sx[-1], sy[-1]) == pytest.approx((2, 0))
    closed = kw.CatmullRomSpline2D([(0, 0), (1, 0), (0, 1)], samples=12, closed=True)
    cx, cy = closed.sample()
    assert (cx[0], cy[0]) == pytest.approx((cx[-1], cy[-1]))


@pytest.mark.parametrize(
    "curve",
    [
        kw.Ellipse(3, 2, center=(1, -1), rotation=0.2, samples=20),
        kw.Arc(2, start_angle=0, end_angle=math.pi / 2, samples=20),
        kw.Parabola(0.5, samples=20),
        kw.Hyperbola(2, 1, samples=20),
        kw.ArchimedeanSpiral(samples=20),
        kw.LogarithmicSpiral(samples=20),
        kw.LissajousCurve2D(samples=20),
    ],
)
def test_named_2d_curves_sample_and_plot(curve):
    x, y = curve.sample()
    assert len(x) == len(y)
    assert np.isfinite(x).any() and np.isfinite(y).any()
    assert curve.plot(show=False, color="navy") in plt.gca().lines
    assert curve.scatter(show=False).get_offsets().shape[1] == 2


def test_hyperbola_has_a_gap_between_its_two_branches():
    x, y = kw.Hyperbola(samples=10).sample()
    assert np.isnan(x).sum() == np.isnan(y).sum() == 1


def test_3d_generic_curves_and_named_curves():
    curves = [
        kw.ParametricCurve3D("cos(t)", "sin(t)", "t", samples=12),
        kw.BezierCurve3D([(0, 0, 0), (1, 2, 3)], samples=12),
        kw.CatmullRomSpline3D([(0, 0, 0), (1, 2, 1), (2, 0, 2)], samples=12),
        kw.CatmullRomSpline3D([(0, 0, 0), (1, 0, 1), (0, 1, 2)], samples=12, closed=True),
        kw.Line3D(direction=(1, 2, 3), samples=12),
        kw.Helix(turns=2, samples=12),
        kw.LissajousCurve3D(samples=12),
        kw.TorusKnot(2, 3, samples=12),
    ]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for curve in curves:
        values = curve.sample()
        assert all(len(axis) >= 12 for axis in values)
        assert curve.plot(show=False, fig=fig, ax=ax) in ax.lines
    assert len(curves[1].control_points) == 2
    assert len(curves[2].control_points) == 3


def test_3d_curve_scatter_helper():
    curve = kw.Helix(samples=10)
    artist = kw.scatter_curve_3d(curve, show=False, color="green")
    assert artist in plt.gca().collections


@pytest.mark.parametrize(
    "surface",
    [
        kw.ExplicitSurface3D("z=x+y", resolution=6),
        kw.ParametricSurface3D("u", "v", "u+v", resolution=(6, 7)),
        kw.Sphere(2, resolution=6),
        kw.Ellipsoid((2, 1, 3), resolution=6),
        kw.Cylinder(1, 3, resolution=6),
        kw.Cone(1, 3, resolution=6),
        kw.Torus(2, 0.4, resolution=6),
    ],
)
def test_generic_and_named_surfaces_sample_and_plot(surface):
    X, Y, Z = surface.sample()
    assert X.shape == Y.shape == Z.shape
    artist = surface.plot(show=False, alpha=0.5)
    assert artist in plt.gca().collections


def test_surface_wireframe_and_domain_failures():
    surface = kw.ExplicitSurface3D(lambda x, y: 1 / x, x_range=(-1, 1), resolution=5)
    assert np.isnan(surface.sample()[2][:, 2]).all()
    artist = surface.plot(show=False, wireframe=True, color="black")
    assert artist in plt.gca().collections


def test_plane_surface_handles_vertical_planes():
    plane = kw.Surface((1, 0, 0, -2), resolution=5)
    X, Y, Z = plane.sample()
    assert np.allclose(X, 2)
    assert plane.plot(show=False) in plt.gca().collections
    assert plane.plot3d(show=False, wireframe=True) in plt.gca().collections
    regular = kw.Surface((1, 1, 1, 0))
    mesh = np.meshgrid([0, 1], [0, 1])
    assert regular.plot(show=False, meshgrid=mesh) in plt.gca().collections


def test_graph2d_composes_curves_legacy_objects_and_styles():
    graph = kw.Graph2D([kw.Function("f(x)=x")])
    ellipse = kw.Ellipse(samples=20)
    graph.add(ellipse, label="ellipse", color="red", linewidth=2)
    graph.add(kw.Circle(1), label="circle", color="blue")
    hidden = kw.Arc()
    graph.add(hidden, visible=False)
    artists = graph.plot(show=False, legend=True, grid=True, equal_aspect=True, return_artists=True)
    assert len(artists) == 3
    assert graph.fig is graph.ax.figure
    assert graph.artists == artists
    assert graph.remove(ellipse) is ellipse
    with pytest.raises(ValueError):
        graph.remove(ellipse)
    assert graph.clear() is graph
    assert graph.is_empty()


def test_graph_keeps_legacy_mutable_items_behavior():
    graph = kw.Graph2D()
    graph.items.append(kw.Ellipse(samples=10))
    assert len(graph.plot(show=False, return_artists=True)) == 1


def test_graph2d_accepts_lines_points_vectors_callables_and_rejects_3d():
    graph = kw.Graph2D([
        kw.Line2D((0, 0), (1, 1)),
        kw.Line2D((2, 0), (2, 1)),
        kw.Point2D(1, 2),
        kw.Vector2D(1, 1),
        lambda x: x * x,
    ])
    assert len(graph.plot(values=[-1, 0, 1], show=False, return_artists=True)) == 5
    with pytest.raises(TypeError):
        kw.Graph2D([kw.Helix()]).plot(show=False)


def test_graph3d_uses_stored_items_and_legacy_function_argument():
    graph = kw.Graph3D()
    graph.add(kw.Helix(samples=20), label="helix", color="purple")
    graph.add(kw.Sphere(resolution=6), alpha=0.2)
    graph.add(kw.Surface((0, 0, 1, 0), resolution=6), wireframe=True, color="black")
    artists = graph.plot(show=False, legend=True, grid=True, xlim=(-3, 3), return_artists=True)
    assert len(artists) == 3
    assert graph.ax.get_xlim() == pytest.approx((-3, 3))

    legacy = kw.Graph3D()
    assert len(legacy.plot([lambda x, y: x + y], start=0, stop=1, step=0.5, show=False, return_artists=True)) == 1


def test_graph3d_scatter_and_validation():
    graph = kw.Graph3D([kw.Helix(samples=10)])
    assert len(graph.scatter(show=False, return_artists=True)) == 1
    with pytest.raises(TypeError):
        kw.Graph3D([kw.Sphere(resolution=4)]).scatter(show=False)
    with pytest.raises(TypeError):
        kw.Graph3D([object()]).plot(show=False)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kw.ParametricCurve2D("t", "t", t_range=(1, 1)),
        lambda: kw.ParametricCurve2D("t", "t", samples=1),
        lambda: kw.BezierCurve2D([(0, 0)]),
        lambda: kw.CatmullRomSpline2D([(0, 0), (1, 1)]),
        lambda: kw.Ellipse(0, 1),
        lambda: kw.Arc(-1),
        lambda: kw.Parabola(0),
        lambda: kw.Hyperbola(-1, 1),
        lambda: kw.LogarithmicSpiral(0),
        lambda: kw.Line3D(direction=(0, 0, 0)),
        lambda: kw.Helix(radius=0),
        lambda: kw.TorusKnot(0, 1),
        lambda: kw.Sphere(0),
        lambda: kw.Ellipsoid((1, -1, 1)),
        lambda: kw.Cylinder(height=0),
        lambda: kw.Cone(radius=0),
        lambda: kw.Torus(minor_radius=0),
    ],
)
def test_invalid_curve_and_surface_parameters_are_friendly(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_invalid_expression_and_resolution_inputs():
    with pytest.raises(TypeError):
        kw.ParametricCurve2D(object(), "t")
    with pytest.raises(ValueError):
        kw.ExplicitSurface3D("x+y", resolution=1)
    with pytest.raises(ValueError):
        kw.ParametricSurface3D("u", "v", "u", u_range=(2, 1))
