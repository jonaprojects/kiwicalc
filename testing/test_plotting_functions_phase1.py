import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_vector_field_is_public_and_returns_the_primary_artist():
    fig, ax = plt.subplots()
    artist = kw.plot_vector_field(
        lambda x, y: -y, lambda x, y: x,
        x_range=(-2, 2), y_range=(-1, 1), density=(5, 4),
        color='magnitude', colorbar=True, fig=fig, ax=ax, show=False,
    )

    assert artist.axes is ax
    assert artist.U.size == 20
    assert artist.kiwicalc_artists == (artist,)
    assert artist.kiwicalc_colorbar is not None


def test_slope_and_gradient_fields_reuse_supplied_axes():
    fig, axes = plt.subplots(1, 2)
    slope = kw.plot_slope_field(lambda x, y: x - y, density=4,
                                fig=fig, ax=axes[0], show=False)
    gradient = kw.plot_gradient_field('f(x,y)=x^2+y^2', density=5,
                                      fig=fig, ax=axes[1], show=False)

    assert slope.axes is axes[0]
    assert gradient.axes is axes[1]
    assert slope.U.size == 16
    assert gradient.U.size == 25


def test_streamlines_and_contours_expose_related_artists_and_colorbars():
    stream = kw.plot_streamlines(1, 0, samples=12, color='teal', show=False)
    contour = kw.plot_contour(
        lambda x, y: x*x + y*y, levels=[1, 4], filled=False,
        labels=True, samples=25, colorbar=True, show=False,
    )

    assert len(stream.kiwicalc_artists) == 2
    assert contour.kiwicalc_colorbar is not None
    assert len(contour.levels) == 2


@pytest.mark.parametrize(
    'call, message',
    [
        (lambda: kw.plot_vector_field(1, 1, density=1, show=False), 'at least 2'),
        (lambda: kw.plot_streamlines(1, 1, density=0, show=False), 'positive'),
        (lambda: kw.plot_contour(lambda x, y: x, x_range=(1, -1), show=False), 'smaller'),
    ],
)
def test_standalone_field_validation_matches_graph_fields(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_field_plot_aliases_and_top_level_exports():
    assert kw.plot_streamplot is kw.plot_streamlines
    assert kw.plot_contour_map is kw.plot_contour
