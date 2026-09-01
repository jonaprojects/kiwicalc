import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "curve",
    [
        kw.Cardioid(samples=31),
        kw.RoseCurve(samples=31),
        kw.Cycloid(samples=31),
        kw.Epicycloid(samples=31),
        kw.Hypocycloid(samples=31),
        kw.Superellipse(samples=31),
        kw.Catenary(samples=31),
        kw.Involute(samples=31),
    ],
)
def test_new_2d_curves_sample_plot_and_round_trip(curve):
    x, y = curve.sample()
    assert x.shape == y.shape == (31,)
    assert np.isfinite(x).all() and np.isfinite(y).all()

    artist = kw.Graph2D([curve]).plot(show=False, return_artists=True)
    assert len(artist) == 1

    restored = kw.Curve2D.from_dict(curve.to_dict())
    assert type(restored) is type(curve)
    restored_x, restored_y = restored.sample()
    assert restored_x == pytest.approx(x)
    assert restored_y == pytest.approx(y)


def test_named_2d_curves_have_expected_landmarks():
    assert kw.Cardioid(scale=2, center=(3, 4)).point_at(0).coordinates == pytest.approx((3, 4))
    assert kw.Cycloid(radius=2, turns=1).point_at(1).coordinates == pytest.approx((4 * np.pi, 0))
    assert kw.Catenary(scale=2, vertex=(3, 4), samples=501).point_at(0.5).coordinates == pytest.approx((3, 4))
    assert kw.Involute(radius=2, center=(3, 4)).point_at(0).coordinates == pytest.approx((5, 4))

    x, y = kw.Superellipse(3, 2, exponent=4, samples=101).sample()
    equation = (np.abs(x) / 3) ** 4 + (np.abs(y) / 2) ** 4
    assert equation == pytest.approx(np.ones_like(equation))


@pytest.mark.parametrize("curve", [kw.Epicycloid(samples=101), kw.Hypocycloid(samples=101)])
def test_default_cycloids_are_closed(curve):
    assert curve.point_at(0).coordinates == pytest.approx(curve.point_at(1).coordinates, abs=1e-8)


@pytest.mark.parametrize("curve", [kw.TrefoilKnot(samples=41), kw.FigureEightKnot(samples=41)])
def test_new_knots_sample_plot_close_and_round_trip(curve):
    x, y, z = curve.sample()
    assert x.shape == y.shape == z.shape == (41,)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()
    assert curve.point_at(0).coordinates == pytest.approx(curve.point_at(1).coordinates, abs=1e-8)
    assert len(kw.Graph3D([curve]).plot(show=False, return_artists=True)) == 1

    restored = kw.Curve3D.from_dict(curve.to_dict())
    assert type(restored) is type(curve)
    for actual, expected in zip(restored.sample(), curve.sample()):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "surface",
    [
        kw.Paraboloid(radius=2, resolution=9),
        kw.HyperbolicParaboloid(resolution=9),
        kw.Hyperboloid(resolution=9),
    ],
)
def test_new_surfaces_sample_plot_and_round_trip(surface):
    x, y, z = surface.sample()
    assert x.shape == y.shape == z.shape == (9, 9)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()
    assert len(kw.Graph3D([surface]).plot(show=False, return_artists=True)) == 1

    restored = kw.Surface3D.from_dict(surface.to_dict())
    assert type(restored) is type(surface)
    for actual, expected in zip(restored.sample(), surface.sample()):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kw.Cardioid(scale=0),
        lambda: kw.RoseCurve(petals=0),
        lambda: kw.Cycloid(turns=0),
        lambda: kw.Hypocycloid(rolling_radius=4),
        lambda: kw.Superellipse(exponent=0),
        lambda: kw.Catenary(scale=0),
        lambda: kw.Involute(radius=0),
        lambda: kw.FigureEightKnot(scale=0),
        lambda: kw.Paraboloid(radius=0),
        lambda: kw.HyperbolicParaboloid(scale_x=0),
        lambda: kw.Hyperboloid(radii=(1, 0, 1)),
    ],
)
def test_new_catalog_rejects_invalid_parameters(factory):
    with pytest.raises(ValueError):
        factory()
