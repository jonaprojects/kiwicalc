import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kiwicalc.linalg import LinearAlgebraPlot, Matrix, RowReductionExplanation


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_explain_rref_records_replayable_steps_without_mutating_source():
    matrix = Matrix([[0, 2, 4], [1, 3, 5]])
    original = matrix.to_list()

    explanation = matrix.explain_rref()

    assert isinstance(explanation, RowReductionExplanation)
    assert matrix.to_list() == original
    assert explanation.result == matrix.rref()
    assert explanation.pivot_columns == (0, 1)
    assert explanation.rank == 2
    assert explanation.steps[0].kind == "swap"
    assert explanation.steps[0].description == "R1 <-> R2"
    assert explanation.steps[-1].after == explanation.result
    assert "RREF complete" in explanation.as_text()


def test_explain_rref_handles_an_already_reduced_matrix_and_plots_steps():
    explanation = Matrix.identity(2).explain_rref()
    assert explanation.steps == ()
    assert explanation.as_text() == "The matrix is already in reduced row-echelon form."

    plot = explanation.plot(show=False, theme="classroom")
    assert isinstance(plot, LinearAlgebraPlot)
    assert len(plot.axes) == 1
    assert plot.data["explanation"] is explanation


def test_explain_rref_rejects_invalid_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        Matrix([[1]]).explain_rref(tolerance=-1)


def test_visualize_transformation_2d_exposes_transformed_data():
    matrix = Matrix([[2, 1], [0, 1]])
    plot = matrix.visualize_transformation(
        vectors=[(1, 2)], title="Shear and stretch", theme="engineering", show=False,
    )

    assert isinstance(plot, LinearAlgebraPlot)
    assert len(plot.axes) == 2
    assert plot.figure._suptitle.get_text() == "Shear and stretch"
    assert np.allclose(plot.data["transformed_vectors"], [[4, 2]])
    assert plot.axes[0].get_aspect() == 1.0


def test_visualize_transformation_supports_3d_and_checks_inputs():
    plot = Matrix.diagonal([2, 1, .5]).visualize_transformation(
        vectors=[(1, 1, 1)], grid_lines=4, show=False,
    )
    assert len(plot.axes) == 2
    assert plot.axes[0].name == "3d"
    assert np.allclose(plot.data["transformed_vectors"], [[2, 1, .5]])

    with pytest.raises(ValueError, match="2x2 or 3x3"):
        Matrix([[1, 2, 3], [4, 5, 6]]).visualize_transformation(show=False)
    with pytest.raises(ValueError, match="rows of 2"):
        Matrix.identity(2).visualize_transformation(vectors=[(1, 2, 3)], show=False)
    with pytest.raises(ValueError, match="at least 2"):
        Matrix.identity(2).visualize_transformation(grid_lines=1, show=False)


def test_visualize_eigenvectors_shows_real_eigenpairs_and_rejects_complex_ones():
    matrix = Matrix([[2, 0], [0, 3]])
    plot = matrix.visualize_eigenvectors(show=False, title="Principal directions")

    assert len(plot.axes) == 1
    assert plot.ax.get_title() == "Principal directions"
    assert np.allclose(
        plot.data["transformed_vectors"],
        plot.data["matrix"] @ plot.data["eigenvectors"],
    )
    assert sorted(plot.data["eigenvalues"]) == [2, 3]

    rotation = Matrix([[0, -1], [1, 0]])
    with pytest.raises(ValueError, match="complex eigenpairs"):
        rotation.visualize_eigenvectors(show=False)


def test_visualize_svd_teaches_each_stage_and_reconstructs_matrix(tmp_path):
    matrix = Matrix([[3, 1], [1, 2]])
    plot = matrix.visualize_svd(show=False)

    assert len(plot.axes) == 4
    assert len(plot.data["stages"]) == 4
    reconstructed = plot.data["u"] @ np.diag(plot.data["singular_values"]) @ plot.data["vt"]
    assert np.allclose(reconstructed, matrix.to_numpy())
    assert np.allclose(plot.data["stages"][-1], matrix.to_numpy() @ plot.data["stages"][0])

    output = tmp_path / "svd.png"
    assert plot.save(output) == output
    assert output.exists()

    with pytest.raises(ValueError, match="2x2"):
        Matrix.identity(3).visualize_svd(show=False)


def test_visualize_least_squares_shows_fit_and_residuals():
    design = Matrix([[1, 0], [1, 1], [1, 2], [1, 3]])
    plot = design.visualize_least_squares([1, 2.8, 5.2, 6.9], show=False)

    assert len(plot.axes) == 1
    assert plot.ax.get_xlabel() == "x"
    assert plot.data["coefficients"].shape == (2,)
    assert np.allclose(
        plot.data["observed"] - plot.data["fitted"],
        plot.data["residuals"],
    )
    assert plot.data["solution"].method == "least_squares"
    assert len(plot.artists) == 3


def test_visualize_least_squares_accepts_explicit_x_and_has_friendly_errors():
    design = Matrix([[1, 0, 0], [1, 1, 1], [1, 2, 4]])
    plot = design.visualize_least_squares([1, 2, 5], x=[0, 1, 2], show=False)
    assert np.allclose(plot.data["x"], [0, 1, 2])

    with pytest.raises(ValueError, match="Pass x"):
        design.visualize_least_squares([1, 2, 5], show=False)
    with pytest.raises(ValueError, match="exactly 3"):
        design.visualize_least_squares([1, 2, 5], x=[0, 1], show=False)
    with pytest.raises(ValueError, match="one response column"):
        design.visualize_least_squares([[1, 2], [2, 3], [5, 6]], x=[0, 1, 2], show=False)


def test_show_false_does_not_call_pyplot_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("show() should not be called"))
    Matrix.identity(2).visualize_transformation(show=False)
    Matrix.diagonal([2, 3]).visualize_eigenvectors(show=False)
    Matrix.identity(2).visualize_svd(show=False)
    Matrix([[1, 0], [1, 1]]).visualize_least_squares([1, 2], show=False)
    Matrix([[1, 2], [3, 4]]).explain_rref().plot(show=False)
