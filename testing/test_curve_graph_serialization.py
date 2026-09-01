import json

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "curve",
    [
        kw.ParametricCurve2D("cos(t)", "sin(t)", sampling="adaptive"),
        kw.PolarCurve2D("1+theta", samples=20),
        kw.ImplicitCurve2D("x^2+y^2=1", resolution=20),
        kw.BezierCurve2D([(0, 0), (1, 2), (2, 0)], samples=20),
        kw.CatmullRomSpline2D([(0, 0), (1, 2), (2, 0)], samples=20, closed=True),
        kw.Ellipse(3, 2, center=(1, 2), rotation=0.3, samples=20),
        kw.Arc(2, center=(1, 1), samples=20),
        kw.Parabola(0.5, vertex=(1, 2), samples=20),
        kw.Hyperbola(2, 1, center=(1, 0), samples=20),
        kw.ArchimedeanSpiral(1, 0.3, samples=20),
        kw.LogarithmicSpiral(0.2, 0.1, samples=20),
        kw.LissajousCurve2D(samples=20),
        kw.ParametricCurve3D("cos(t)", "sin(t)", "t", sampling="adaptive"),
        kw.BezierCurve3D([(0, 0, 0), (1, 2, 3)], samples=20),
        kw.CatmullRomSpline3D([(0, 0, 0), (1, 2, 1), (2, 0, 2)], samples=20),
        kw.Line3D(samples=20),
        kw.Helix(samples=20),
        kw.LissajousCurve3D(samples=20),
        kw.TorusKnot(samples=20),
    ],
)
def test_curve_round_trip_preserves_samples(curve):
    restored = kw.Curve2D.from_dict(curve.to_dict())
    assert type(restored) is type(curve)
    for actual, expected in zip(restored.sample(), curve.sample()):
        assert actual == pytest.approx(expected, nan_ok=True)
    assert json.loads(curve.to_json())["type"] == type(curve).__name__


def test_transformed_curve_round_trip():
    curve = kw.Ellipse(samples=20).rotate(0.5).translate(2, 3)
    restored = kw.Curve2D.from_dict(curve.to_dict())
    assert isinstance(restored, kw.TransformedCurve2D)
    assert restored.matrix == pytest.approx(curve.matrix)
    assert restored.sample()[0] == pytest.approx(curve.sample()[0])


@pytest.mark.parametrize(
    "surface",
    [
        kw.ExplicitSurface3D("x+y", resolution=5),
        kw.ParametricSurface3D("u", "v", "u+v", resolution=5),
        kw.Sphere(2, resolution=5), kw.Ellipsoid(resolution=5),
        kw.Cylinder(resolution=5), kw.Cone(resolution=5), kw.Torus(resolution=5),
        kw.Surface((1, 0, 0, -2), resolution=5),
    ],
)
def test_surface_round_trip(surface):
    restored = kw.Surface3D.from_dict(surface.to_dict())
    assert type(restored) is type(surface)
    assert restored.sample()[0] == pytest.approx(surface.sample()[0])


def test_graph2d_round_trip_preserves_items_styles_decorations_and_view(tmp_path):
    graph = kw.Graph2D()
    graph.add(kw.Ellipse(samples=20), label="ellipse", color="blue", linewidth=2)
    graph.add(kw.Function("f(x)=x"), label="line", visible=False, color="gray")
    graph.add(kw.Circle(1), label="circle", color="green")
    graph.mark((1, 2), label="point")
    graph.annotate("note", (1, 2)).vertical_line(1).horizontal_line(0)
    graph.fill_between("f(x)=x", 0, values=[0, 0.5, 1], color="skyblue")
    graph.plot(show=False, text="Saved graph", xlim=(-4, 4), ylim=(-3, 3), equal_aspect=True, grid=True, legend=True)

    payload = graph.to_dict()
    restored = kw.Graph2D.from_dict(payload)
    artists = restored.plot(show=False, return_artists=True)
    assert len(restored.items) == 4
    assert len(restored._decorations) == 4
    assert len(artists) == 7
    assert restored.ax.get_title() == "Saved graph"
    assert restored.ax.get_xlim() == pytest.approx((-4, 4))
    assert restored.ax.get_legend() is not None

    json_text = graph.to_json(indent=2)
    assert kw.Graph.from_json(json_text).items[0].samples == 20
    path = tmp_path / "graph.json"
    assert graph.export_json(path, indent=2) == path
    assert isinstance(kw.Graph.from_json(path), kw.Graph2D)


def test_graph3d_round_trip_preserves_curves_surfaces_and_view():
    graph = kw.Graph3D()
    graph.add(kw.Helix(samples=20), label="helix", color="purple")
    graph.add(kw.Sphere(resolution=5), alpha=0.3)
    graph.plot(show=False, title="Saved 3D", xlim=(-2, 2), zlim=(-3, 3), legend=True)
    restored = kw.Graph3D.from_dict(graph.to_dict())
    assert len(restored.plot(show=False, return_artists=True)) == 2
    assert restored.ax.get_title() == "Saved 3D"
    assert restored.ax.get_zlim() == pytest.approx((-3, 3))


def test_serialization_rejects_callables_and_unknown_types():
    with pytest.raises(TypeError, match="callables"):
        kw.ParametricCurve2D(lambda t: t, "t").to_dict()
    with pytest.raises(TypeError, match="callables"):
        kw.Graph2D([lambda x: x]).to_dict()
    with pytest.raises(ValueError, match="Unknown curve"):
        kw.Curve2D.from_dict({"type": "Mystery"})
    with pytest.raises(ValueError, match="Unknown graph"):
        kw.Graph.from_dict({"type": "Graph4D"})
