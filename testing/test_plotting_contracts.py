import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_plot_function_draws_on_supplied_axes_without_showing():
    fig, ax = plt.subplots()
    kw.plot_function(lambda x: x*x, values=[-1, 0, 1], show=False, show_axis=False, fig=fig, ax=ax)
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [-1, 0, 1]
    assert list(ax.lines[0].get_ydata()) == [1, 0, 1]


def test_scatter_dots_draws_points_and_validates_lengths():
    fig, ax = plt.subplots()
    kw.scatter_dots([1, 2], [3, 4], show=False, show_axis=False, fig=fig, ax=ax)
    assert len(ax.collections) == 1
    assert np.asarray(ax.collections[0].get_offsets()).tolist() == [[1.0, 3.0], [2.0, 4.0]]
    with pytest.raises(ValueError):
        kw.scatter_dots([1], [2, 3], show=False)


def test_plot_function_3d_adds_surface_to_supplied_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    values = np.array([-1.0, 0.0, 1.0])
    meshgrid = np.meshgrid(values, values)
    kw.plot_function_3d(
        lambda x, y: x + y,
        meshgrid=meshgrid,
        show=False,
        fig=fig,
        ax=ax,
    )
    assert len(ax.collections) == 1


def test_plot_vector_helpers_add_expected_artists():
    fig2d, ax2d = plt.subplots()
    kw.plot_vector_2d(0, 0, 2, 3, show=False, fig=fig2d, ax=ax2d)
    assert len(ax2d.patches) == 1

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    kw.plot_vector_3d((0, 0, 0), (1, 2, 3), show=False, fig=fig3d, ax=ax3d)
    assert len(ax3d.collections) == 1


@pytest.mark.parametrize(
    ('count', 'expected'),
    [(1, (1, 1)), (4, (2, 2)), (6, (2, 3)), (7, (3, 3))],
)
def test_subplot_shape_is_compact(count, expected):
    from kiwicalc.plotting.plots import generate_subplot_shape

    assert generate_subplot_shape(count) == expected
