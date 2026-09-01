import importlib

import pytest

import kiwicalc as kw


worksheet_module = importlib.import_module("kiwicalc.pdf.worksheet")


def test_linear_exercise_generators_deterministically(monkeypatch):
    values = iter([0, 0, 2, 4])
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: next(values))
    exercise, solution = worksheet_module.linear_from_points_exercise(get_solution=True)
    assert "passes through" in exercise and "Increasing" in solution

    values = iter([0, 3, 2, 1])
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: next(values))
    assert isinstance(worksheet_module.linear_from_points_exercise(get_solution=False), str)

    values = iter([2, 5, 0, -2])
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: next(values))
    exercise, solution = worksheet_module.linearFromPointAndSlope_exercise(get_solution=True)
    assert "slope" in exercise and "y =" in solution

    values = iter([2, 5, 3])
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: next(values))
    assert isinstance(worksheet_module.linearFromPointAndSlope_exercise(get_solution=False), str)

    # Equal y coordinates trigger the horizontal-line avoidance branch.
    values = iter([0, 1, 2, 1, 2])
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: next(values))
    assert isinstance(worksheet_module.linear_from_points_exercise(get_solution=False), str)
    assert worksheet_module.linear_intersection_exercise() is None


def test_linear_system_exercise_solution_and_prompt_paths(monkeypatch):
    monkeypatch.setattr(
        worksheet_module,
        "random_linear_system",
        lambda variables, get_solutions, digits_after: (["x+y=3", "x-y=1"], [2, 1])
        if get_solutions
        else ["x+y=3", "x-y=1"],
    )
    prompt, solution = worksheet_module.linear_system_exercise(["x", "y"], get_solution=True)
    assert "Solve" in prompt and solution == "x=2, y=1"
    assert isinstance(worksheet_module.linear_system_exercise(["x", "y"], get_solution=False), str)


def test_path_generation_and_pdf_creation(tmp_path, monkeypatch):
    existing = {"worksheet1.pdf", "worksheet2.pdf"}
    monkeypatch.setattr(worksheet_module.os.path, "isfile", lambda path: path in existing)
    assert worksheet_module.generate_pdf_path() == "worksheet3.pdf"

    pdf_path = tmp_path / "single.pdf"
    assert worksheet_module.create_pdf(pdf_path, title="Practice", lines=["x=1", "x=2"])
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    pages_path = tmp_path / "pages.pdf"
    worksheet_module.create_pages(pages_path, 2, ["One", "Two"], [["a"], ["b"]])
    assert pages_path.exists()
    with pytest.warns(UserWarning):
        assert not worksheet_module.create_pdf(tmp_path / "missing" / "bad.pdf")


def test_worksheet_dtype_dispatch(monkeypatch, tmp_path):
    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)

    for cls in (
        worksheet_module.LinearEquation,
        worksheet_module.QuadraticEquation,
        worksheet_module.CubicEquation,
        worksheet_module.QuarticEquation,
        worksheet_module.PolyEquation,
    ):
        monkeypatch.setattr(cls, "random_worksheets", recorder)

    for dtype in ("linear", "quadratic", "cubic", "quartic", "polynomial", "trigo", "log"):
        worksheet_module.worksheet(path=str(tmp_path / f"{dtype}.pdf"), dtype=dtype, num_of_pages=1, equations_per_page=1)
    assert len(calls) == 5
    with pytest.raises(ValueError):
        worksheet_module.worksheet(path=str(tmp_path / "bad.pdf"), dtype="unknown")

    monkeypatch.setattr(worksheet_module, "generate_pdf_path", lambda: str(tmp_path / "generated.pdf"))
    worksheet_module.worksheet(path=None, dtype="linear", num_of_pages=1, equations_per_page=1)


def test_pdf_exercise_properties_and_simple_subclasses():
    exercise = worksheet_module.PDFExercise("solve", "equation", "linear", solution=2, lang="he")
    assert exercise.exercise == "solve"
    assert exercise.dtype == "linear"
    assert exercise.solution == 2 and exercise.has_solution
    assert exercise.lang == "he"
    exercise.number = 7
    assert exercise.number == 7 and str(exercise) == "solve"
    assert not worksheet_module.PDFExercise("q", "equation", "linear").has_solution
    assert isinstance(worksheet_module.PDFCalculusExercise("q", "derivative"), worksheet_module.PDFExercise)
    assert isinstance(worksheet_module.PDFAnalyzeFunction("q", "linear"), worksheet_module.PDFCalculusExercise)
    assert isinstance(worksheet_module.PDFEquationExercise("q", "linear"), worksheet_module.PDFExercise)


def test_linear_function_constructor_branches(monkeypatch):
    monkeypatch.setattr(worksheet_module, "random_linear", lambda **_: ("2x+1", -0.5, [2, 1]))
    positive = worksheet_module.PDFLinearFunction(with_solution=True)
    assert positive.has_solution and "Increasing" in positive.solution
    monkeypatch.setattr(worksheet_module, "random_linear", lambda **_: ("-2x+1", 0.5, [-2, 1]))
    negative = worksheet_module.PDFLinearFunction(with_solution=True)
    assert "Decreasing" in negative.solution
    assert not worksheet_module.PDFLinearFunction(with_solution=False).has_solution


def test_linear_system_and_linear_prompt_subclasses(monkeypatch):
    monkeypatch.setattr(
        worksheet_module,
        "linear_system_exercise",
        lambda variables, get_solution, digits_after: ("system", "answer") if get_solution else "system",
    )
    assert worksheet_module.PDFLinearSystem(with_solution=True, num_of_equations=2).has_solution
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: 2)
    assert worksheet_module.PDFLinearSystem(with_solution=False, num_of_equations=None).exercise == "system"
    assert not worksheet_module.PDFLinearSystem(with_solution=False, num_of_equations=3).has_solution
    assert worksheet_module.PDFLinearSystem(with_solution=False, num_of_equations=9).exercise == "system"
    with pytest.raises(ValueError):
        worksheet_module.PDFLinearSystem(num_of_equations=27)

    monkeypatch.setattr(worksheet_module, "linear_from_points_exercise", lambda get_solution, lang: ("points", "solution") if get_solution else "points")
    monkeypatch.setattr(worksheet_module, "linearFromPointAndSlope_exercise", lambda get_solution, lang: ("slope", "solution") if get_solution else "slope")
    assert worksheet_module.PDFLinearFromPoints(True).has_solution
    assert not worksheet_module.PDFLinearFromPoints(False).has_solution
    assert worksheet_module.PDFLinearFromPointAndSlope(True).has_solution
    assert not worksheet_module.PDFLinearFromPointAndSlope(False).has_solution
    assert isinstance(worksheet_module.PDFLinearIntersection(), worksheet_module.PDFLinearIntersection)


def test_polynomial_function_subclasses(monkeypatch):
    monkeypatch.setattr(worksheet_module, "random_polynomial", lambda degree, get_solutions=True: ("x^2-1", [-1, 1]))
    monkeypatch.setattr(
        worksheet_module.Poly,
        "data",
        lambda self, no_roots: {
            "derivative": kw.Poly("2x"),
            "extremums": [],
            "up": "x>0",
            "down": "x<0",
        },
    )
    assert worksheet_module.PDFPolyFunction(with_solution=True, degree=2).has_solution
    monkeypatch.setattr(worksheet_module.random, "randint", lambda *_: 2)
    assert not worksheet_module.PDFPolyFunction(with_solution=False, degree=None).has_solution
    assert not worksheet_module.PDFPolyFunction(with_solution=False, degree=3).has_solution
    assert worksheet_module.PDFQuadraticFunction(False).dtype == "poly"
    assert worksheet_module.PDFCubicFunction(False).dtype == "poly"
    assert worksheet_module.PDFQuarticFunction(False).dtype == "poly"


def test_equation_exercise_subclasses(monkeypatch):
    monkeypatch.setattr(worksheet_module.LinearEquation, "random_equation", lambda digits_after, get_solution: ("x=1", 1) if get_solution else "x=1")
    assert worksheet_module.PDFLinearEquation(True, number=1).has_solution
    assert not worksheet_module.PDFLinearEquation(False).has_solution

    monkeypatch.setattr(
        worksheet_module,
        "random_polynomial",
        lambda degree, get_solutions=False: (f"x^{degree}-1", [1]) if get_solutions else f"x^{degree}-1",
    )
    for cls in (
        worksheet_module.PDFQuadraticEquation,
        worksheet_module.PDFCubicEquation,
        worksheet_module.PDFQuarticEquation,
        worksheet_module.PDFPolyEquation,
    ):
        assert cls(True).has_solution
        assert not cls(False).has_solution


def test_pdf_page_iteration_and_worksheet_layout(tmp_path):
    first = worksheet_module.PDFExercise("first", "equation", "linear", solution=[1, 2])
    second = worksheet_module.PDFExercise("multi\nline", "equation", "linear", solution="a\nb")
    third = worksheet_module.PDFExercise("third", "equation", "linear", solution=None)
    page = worksheet_module.PDFPage("Page", [first, second])
    assert page.title == "Page" and list(page) == [first, second]
    page.add(third)
    assert page.exercises[-1] is third

    worksheet = worksheet_module.PDFWorksheet("Practice", ordered=True)
    worksheet.add_exercise(first)
    worksheet.add_exercise(second)
    worksheet.add_exercise(third)
    worksheet.end_page()
    assert worksheet.num_of_pages == 2
    assert len(worksheet.pages) == 2
    worksheet.next_page()
    worksheet.next_page("Custom")
    assert worksheet.current_page.title == "Custom"
    worksheet.del_last_page()
    output = tmp_path / "worksheet.pdf"
    worksheet.create(output)
    assert output.exists()

    unordered = worksheet_module.PDFWorksheet("Loose", ordered=False)
    unordered.add_exercise(worksheet_module.PDFExercise("plain", "equation", "linear", solution=3))
    unordered.add_exercise(worksheet_module.PDFExercise("multiple\nlines", "equation", "linear", solution=None))
    unordered.add_exercise(worksheet_module.PDFExercise("many", "equation", "linear", solution=[1, 2]))
    unordered.end_page()
    assert unordered.num_of_pages == 2


def test_worksheet_create_uses_generated_path(monkeypatch, tmp_path):
    generated = tmp_path / "automatic.pdf"
    calls = []
    monkeypatch.setattr(worksheet_module, "generate_pdf_path", lambda: str(generated))
    monkeypatch.setattr(worksheet_module, "create_pages", lambda path, num_of_pages, titles, lines: calls.append((path, num_of_pages)))
    worksheet_module.PDFWorksheet("Auto").create(path=None)
    assert calls == [(str(generated), 1)]
