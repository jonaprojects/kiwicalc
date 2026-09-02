import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, ScalarFormatter
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting.axis import (
    _degree_label, _pi_label, axis_label, configure_minor_ticks,
    configure_ticks, normalize_units,
)
from kiwicalc.plotting.themes import apply_theme


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_builtin_themes_are_public_scoped_and_chainable():
    assert kw.available_themes() == (
        "classroom", "projector", "publication", "engineering", "colorblind"
    )
    original_font_size = mpl.rcParams["font.size"]
    original_cycle = mpl.rcParams["axes.prop_cycle"]

    graph = kw.Graph2D().theme("classroom").add(lambda x: x, label="line")
    returned = graph.plot(show=False, title="Classroom", legend=True, return_artists=True)

    assert returned[0].get_linewidth() == pytest.approx(2.6)
    assert graph.ax.title.get_fontsize() == pytest.approx(17)
    assert graph.ax.get_facecolor() == pytest.approx(mpl.colors.to_rgba("white"))
    assert mpl.rcParams["font.size"] == original_font_size
    assert mpl.rcParams["axes.prop_cycle"] == original_cycle


@pytest.mark.parametrize("name", ["classroom", "projector", "publication", "engineering", "colorblind"])
def test_every_builtin_theme_renders(name):
    graph = kw.Graph2D([lambda x: x]).theme(name)
    assert graph.plot(show=False, return_artists=True)
    assert graph._theme.name == name


def test_custom_themes_are_immutable_and_validate_overrides():
    base = kw.get_theme("publication")
    custom = base.with_overrides(name="notes", line_width=4, color_cycle=["red", "blue"])
    assert base.line_width == 1.5
    assert custom.line_width == 4
    assert custom.color_cycle == ("red", "blue")
    assert kw.PlotTheme.from_dict(custom.to_dict()) == custom

    with pytest.raises(ValueError, match="Unknown theme"):
        kw.get_theme("invisible")
    with pytest.raises(ValueError, match="Unknown theme option"):
        base.with_overrides(imaginary=True)


def test_theme_resolution_accepts_objects_mappings_and_local_overrides():
    assert kw.get_theme(None) is None
    custom = kw.get_theme(None, name="custom-blue", line_width=3)
    assert custom.name == "custom-blue" and custom.line_width == 3
    assert kw.get_theme(custom) is custom
    assert kw.get_theme(custom.to_dict()) == custom
    assert kw.PlotTheme.from_dict({"name": "minimal"}).name == "minimal"
    with pytest.raises(TypeError, match="theme must be"):
        kw.get_theme(42)

    fig, ax = plt.subplots()
    apply_theme(fig, ax, kw.PlotTheme(grid=False, minor_grid=False))
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    apply_theme(fig, ax, None)


def test_axis_helper_labels_units_and_formatter_edge_cases():
    assert axis_label("Distance", None) == "Distance"
    assert axis_label("", "m") == "m"
    assert axis_label("Distance (m)", "m") == "Distance (m)"
    assert normalize_units("m", ("x", "y")) == ("m", "m")
    assert normalize_units({"x": "s"}, ("x", "y")) == ("s", None)
    assert _pi_label(0) == "0"
    assert _pi_label(-np.pi) == r"$-\pi$"
    assert _pi_label(2 * np.pi) == r"$2\pi$"
    assert _pi_label(0.123456) == "0.123456"
    assert _degree_label(np.pi / 7).endswith("°")


def test_tick_configuration_aliases_plain_mode_and_minor_tick_validation():
    fig, ax = plt.subplots()
    configure_ticks(ax.xaxis, "plain")
    assert isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
    configure_ticks(ax.xaxis, "radians")
    assert ax.xaxis.get_major_formatter()(np.pi, 0) == r"$\pi$"
    with pytest.raises(TypeError, match="tick mode"):
        configure_ticks(ax.xaxis, 3)
    with pytest.raises(ValueError, match="degree_step"):
        configure_ticks(ax.xaxis, "degrees", degree_step=0)
    with pytest.raises(ValueError, match="at least 2"):
        configure_minor_ticks(ax.xaxis, True, subdivisions=1)
    configure_minor_ticks(ax.xaxis, False)


def test_axis_configuration_supports_units_pi_ticks_origin_and_minor_grid():
    graph = (
        kw.Graph2D([lambda x: np.sin(x)])
        .theme("engineering")
        .configure_axes(
            xlabel="Angle", ylabel="Amplitude", units=("rad", None),
            x_ticks="pi", origin=True, minor_ticks=True, minor_grid=True,
        )
    )
    graph.plot(show=False, xlim=(-np.pi, np.pi), ylim=(-1.2, 1.2))

    assert graph.ax.get_xlabel() == "Angle (rad)"
    assert graph.ax.get_ylabel() == "Amplitude"
    assert graph.ax.xaxis.get_major_formatter()(np.pi / 2, 0) == r"$\frac{\pi}{2}$"
    assert graph.ax.spines["bottom"].get_position() == "zero"
    assert graph.ax.spines["top"].get_visible() is False
    assert len(graph.ax.xaxis.get_minorticklocs()) > 0


def test_degree_scientific_and_engineering_tick_modes():
    degrees_graph = kw.Graph2D([lambda x: np.sin(x)])
    degrees_graph.plot(show=False, x_ticks="degrees", xlim=(0, np.pi))
    assert degrees_graph.ax.xaxis.get_major_formatter()(np.pi / 2, 0) == "90°"

    scientific = kw.Graph2D([lambda x: x])
    scientific.plot(show=False, y_ticks="scientific")
    assert isinstance(scientific.ax.yaxis.get_major_formatter(), ScalarFormatter)

    engineering = kw.Graph2D([lambda x: x])
    engineering.plot(show=False, y_ticks="engineering", units=(None, "V"))
    formatter = engineering.ax.yaxis.get_major_formatter()
    assert isinstance(formatter, EngFormatter)
    assert "kV" in formatter(1000, 0)


def test_secondary_axes_are_chainable_labeled_and_not_duplicated():
    graph = kw.Graph2D([lambda x: np.sin(x)])
    returned = graph.secondary_xaxis(np.degrees, np.radians, label="Angle", unit="deg")
    assert returned is graph

    graph.plot(show=False, xlim=(0, 2 * np.pi))
    assert len(graph._secondary_axes) == 1

    themed_y = kw.Graph2D([lambda x: x]).theme("classroom")
    themed_y.secondary_yaxis(lambda y: y * 100, lambda y: y / 100, label="Percent", unit="%")
    themed_y.plot(show=False)
    assert themed_y._secondary_axes[0].get_ylabel() == "Percent (%)"
    assert graph._secondary_axes[0].get_xlabel() == "Angle (deg)"

    graph.plot(show=False, xlim=(0, np.pi))
    assert len(graph._secondary_axes) == 1


def test_graph_export_supports_png_svg_pdf_and_friendly_defaults(tmp_path):
    graph = kw.Graph2D([lambda x: x])
    graph.plot(show=False, title="Export")

    png = graph.save(tmp_path / "figure")
    svg = graph.export(tmp_path / "figure.svg")
    pdf = graph.save(tmp_path / "figure.pdf", transparent=True)

    assert png.name == "figure.png" and png.read_bytes().startswith(b"\x89PNG")
    assert "<svg" in svg.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="PNG, SVG, or PDF"):
        graph.save(tmp_path / "figure.jpg")
    with pytest.raises(ValueError, match="match the file extension"):
        graph.save(tmp_path / "figure.png", format="svg")
    explicit = graph.save(tmp_path / "explicit", format="pdf", tight=False)
    assert explicit.suffix == ".pdf"


def test_2d_theme_and_axis_configuration_survive_json_round_trip():
    graph = kw.Graph2D([kw.Ellipse(samples=20)]).theme("engineering")
    graph.plot(
        show=False, title="Orbit", xlabel="Horizontal", ylabel="Vertical",
        units=("km", "km"), x_ticks="engineering", origin=False,
        minor_ticks=True, minor_grid=True,
    )
    restored = kw.Graph2D.from_dict(graph.to_dict())
    restored.plot(show=False)
    assert restored._theme.name == "engineering"
    assert restored.ax.get_xlabel() == "Horizontal (km)"
    assert restored.ax.get_ylabel() == "Vertical (km)"
    assert restored.ax.spines["top"].get_visible() is True


def test_secondary_axis_serialization_rejects_unportable_callables():
    graph = kw.Graph2D([kw.Ellipse(samples=10)])
    graph.secondary_yaxis(lambda value: value * 2, lambda value: value / 2)
    graph.plot(show=False)
    with pytest.raises(TypeError, match="Secondary axes"):
        graph.to_dict()


def test_3d_themes_units_ticks_and_serialization_round_trip():
    graph = kw.Graph3D([kw.Helix(samples=20)]).theme("projector")
    graph.plot(
        show=False, title="Motion", xlabel="East", ylabel="North", zlabel="Height",
        units=("m", "m", "m"), z_ticks="engineering", grid=True,
    )
    assert graph.ax.get_xlabel() == "East (m)"
    assert graph.ax.get_ylabel() == "North (m)"
    assert graph.ax.get_zlabel() == "Height (m)"
    assert graph.artists[0].get_linewidth() == pytest.approx(3.2)

    payload = json.loads(graph.to_json())
    assert payload["view"]["theme"]["name"] == "projector"
    restored = kw.Graph3D.from_dict(payload)
    restored.plot(show=False)
    assert restored._theme.name == "projector"
    assert restored.ax.get_zlabel() == "Height (m)"


def test_3d_scatter_accepts_themes_and_configured_axis_labels():
    graph = (
        kw.Graph3D([kw.Helix(samples=12)])
        .theme("colorblind")
        .configure_axes(xlabel="X", ylabel="Y", zlabel="Z", units="m", minor_ticks=True)
    )
    assert graph.scatter(show=False, return_artists=True)
    assert graph.ax.get_xlabel() == "X (m)"
    assert graph.ax.get_zlabel() == "Z (m)"


def test_visual_axis_validation_errors_are_friendly():
    with pytest.raises(ValueError, match="Unknown axis option"):
        kw.Graph2D().configure_axes(compass="north")
    with pytest.raises(ValueError, match="one value for each axis"):
        kw.Graph2D().plot(show=False, units=("m",))
    with pytest.raises(ValueError, match="tick mode"):
        kw.Graph2D().plot(show=False, x_ticks="clock")
    with pytest.raises(ValueError, match="pi_step"):
        kw.Graph2D().plot(show=False, x_ticks="pi", pi_step=0)
    with pytest.raises(ValueError, match="linear"):
        kw.Graph2D().plot(show=False, xscale="log", origin=True, xlim=(0.1, 10))
    with pytest.raises(TypeError, match="callable"):
        kw.Graph2D().secondary_xaxis(1, 2)
