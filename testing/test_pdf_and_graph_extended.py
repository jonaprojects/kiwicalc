import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw
from kiwicalc.pdf.worksheet import (
    PDFCalculusExercise,
    PDFCubicEquation,
    PDFCubicFunction,
    PDFEquationExercise,
    PDFLinearEquation,
    PDFLinearFromPointAndSlope,
    PDFLinearFromPoints,
    PDFLinearFunction,
    PDFLinearSystem,
    PDFPolyEquation,
    PDFQuadraticEquation,
    PDFQuadraticFunction,
    PDFQuarticEquation,
    PDFQuarticFunction,
)


def test_pdf_exercise_base_contract():
    exercise = kw.PDFExercise("Solve x+1=2", "equation", "linear", solution=1, number=3, lang="en")
    assert exercise.exercise == "Solve x+1=2"
    assert exercise.number == 3
    assert exercise.dtype == "linear"
    assert exercise.solution == 1
    assert exercise.has_solution
    assert exercise.lang == "en"
    exercise.number = 4
    assert str(exercise) == "Solve x+1=2"
    assert exercise.number == 4

    assert PDFCalculusExercise("differentiate x", "derivative").dtype == "derivative"
    assert PDFEquationExercise("x=1", "linear", 1).solution == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PDFLinearFunction(with_solution=False),
        lambda: PDFLinearSystem(with_solution=False, num_of_equations=2),
        lambda: PDFLinearFromPoints(with_solution=False),
        lambda: PDFLinearFromPointAndSlope(with_solution=False),
        lambda: PDFQuadraticFunction(with_solution=False),
        lambda: PDFCubicFunction(with_solution=False),
        lambda: PDFQuarticFunction(with_solution=False),
        lambda: PDFLinearEquation(with_solution=False),
        lambda: PDFQuadraticEquation(with_solution=False),
        lambda: PDFCubicEquation(with_solution=False),
        lambda: PDFQuarticEquation(with_solution=False),
        lambda: PDFPolyEquation(with_solution=False),
    ],
)
def test_generated_pdf_exercise_types(factory):
    exercise = factory()
    assert str(exercise)
    assert not exercise.has_solution


def test_linear_system_exercise_rejects_too_many_variables():
    with pytest.raises(ValueError):
        PDFLinearSystem(num_of_equations=27)


def test_pdf_page_iteration_and_worksheet_lifecycle(tmp_path):
    first = kw.PDFExercise("x+1=2", "equation", "linear", solution=1)
    second = kw.PDFExercise("x+2=4\nShow work", "equation", "linear", solution=[2])
    page = kw.PDFPage("Algebra", [first, second])
    assert page.title == "Algebra"
    assert list(page) == [first, second]
    page.add(kw.PDFExercise("x=3", "equation", "linear"))
    assert len(page.exercises) == 3

    worksheet = kw.PDFWorksheet("Algebra", ordered=True)
    worksheet.add_exercise(first)
    worksheet.add_exercise(second)
    assert worksheet.current_page.exercises == [first, second]
    worksheet.end_page()
    assert worksheet.num_of_pages == 2
    worksheet.next_page("More")
    assert worksheet.current_page.title == "More"

    output = tmp_path / "extended-worksheet.pdf"
    worksheet.create(output)
    assert output.read_bytes().startswith(b"%PDF")
    worksheet.del_last_page()
    assert worksheet.num_of_pages == 2


def test_graph_base_and_2d_plotting():
    fig, ax = plt.subplots()
    graph = kw.Graph([], fig, ax)
    assert graph.is_empty()
    graph.add(kw.Function("f(x)=x"))
    assert len(graph.items) == 1
    with pytest.raises(NotImplementedError):
        graph.plot()
    with pytest.raises(NotImplementedError):
        graph.scatter()
    plt.close(fig)

    graph2d = kw.Graph2D([kw.Function("f(x)=x"), kw.Circle(1)])
    graph2d.plot(values=[-1, 0, 1], show=False)
    assert not graph2d.is_empty()
    plt.close("all")


def test_plot_multiple_and_graph3d(monkeypatch):
    kw.plot_multiple([lambda x: x, lambda x: x * x], values=[-1, 0, 1], show=False)
    plt.close("all")

    called = {}
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_functions_3d", lambda **kwargs: called.setdefault("plot", kwargs))
    monkeypatch.setattr("kiwicalc.plotting.plots.scatter_functions_3d", lambda **kwargs: called.setdefault("scatter", kwargs))
    graph = kw.Graph3D()
    graph.plot([lambda x, y: x + y])
    graph.scatter([lambda x, y: x + y])
    assert "plot" in called and "scatter" in called
    plt.close("all")
