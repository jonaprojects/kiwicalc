import pytest

import kiwicalc as kw


def test_root_evaluation_assignment_variables_and_serialization_branches():
    x = kw.Var("x")
    expression = kw.Root(x + 7, root_by=3, coefficient=2)
    assert expression.variables == {"x"}
    assert expression.when(x=1).try_evaluate() == pytest.approx(4)
    assert kw.Root.from_dict(expression.to_dict()) == expression
    assert kw.Root(9, coefficient=0).try_evaluate() == 0
    assert isinstance(kw.Root(9, root_by=0).try_evaluate(), ValueError)
    assert kw.Root(x).try_evaluate() is None


def test_root_addition_and_subtraction_dispatch_branches():
    x = kw.Var("x")
    root = kw.Root(x)
    assert root + 0 == root
    assert root + kw.Root(x) == kw.Root(x, coefficient=2)
    assert root - kw.Root(x) == 0
    assert isinstance(root + 3, kw.ExpressionSum)
    assert isinstance(root - 3, kw.ExpressionSum)
    assert isinstance(root + kw.Sin(x), kw.ExpressionSum)
    assert kw.Root.dependant_roots(kw.Root(x, 2), kw.Root(x, 3)) is None


def test_root_multiplication_dispatch_branches():
    x = kw.Var("x")
    assert kw.Root(x) * 3 == kw.Root(x, coefficient=3)
    assert kw.Root(x) * kw.Root(4) == kw.Root(x, coefficient=2)
    same_index = kw.Root(x) * kw.Root(x + 1)
    assert isinstance(same_index, kw.Root)
    assert same_index.inside == x * (x + 1)
    assert isinstance(kw.Root(x, 2) * kw.Root(x, 3), kw.ExpressionMul)
    assert isinstance(kw.Root(x) * kw.Sin(x), kw.Root)
    with pytest.raises(TypeError):
        kw.Root(x) * object()


def test_root_power_branches():
    x = kw.Var("x")
    root = kw.Root(x, root_by=4)
    assert root**1 == root
    assert root**0 == 1
    assert root**4 == x
    assert root**8 == x**2
    reduced = root**2
    assert isinstance(reduced, kw.Root) and reduced.root == 2
    symbolic = kw.Root(x, root_by=kw.Var("n")) ** 2
    assert isinstance(symbolic, kw.Root)


def test_root_division_dispatch_branches():
    x = kw.Var("x")
    with pytest.raises(ZeroDivisionError):
        kw.Root(x) / 0
    assert kw.Root(x, coefficient=4) / 2 == kw.Root(x, coefficient=2)
    assert kw.Root(x) / kw.Mono(2) == kw.Root(x, coefficient=0.5)
    assert kw.Root(x) / kw.Root(x) == 1
    changed_index = kw.Root(x, 2) / kw.Root(x, 4)
    assert isinstance(changed_index, kw.Root)
    assert changed_index.root == 4
    assert isinstance(kw.Root(x) / kw.Root(x + 1), kw.Fraction)
    assert isinstance(kw.Root(x) / kw.Sin(x), kw.Fraction)


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        (kw.Root(4, coefficient=0), "0"),
        (kw.Root(4), "√(4)"),
        (kw.Root(4, coefficient=-1), "-√(4)"),
        (kw.Root(8, root_by=3, coefficient=2), "2 * 3^√(8)"),
    ],
)
def test_root_string_branches(root, expected):
    assert str(root) == expected


def test_root_equality_branches():
    x = kw.Var("x")
    assert kw.Root(4) == 2
    assert kw.Root(x) == kw.Root(x)
    assert kw.Root(x) != kw.Root(x + 1)
    assert kw.Root(x) != kw.Sin(x)
    assert kw.Root(x) != None  # noqa: E711


def test_root_derivative_branches():
    x = kw.Var("x")
    assert kw.Root(4).derivative() == 0
    assert kw.Root(x, coefficient=0).derivative() == 0
    square_derivative = kw.Root(x).derivative()
    assert square_derivative.when(x=4).try_evaluate() == pytest.approx(0.25)
    cube_derivative = kw.Root(x, root_by=3).derivative()
    assert cube_derivative.when(x=8).try_evaluate() == pytest.approx(1 / 12)
    assert kw.Root(x, root_by=1).derivative() == 1
    assert kw.Root(x, root_by=0.5).derivative().when(x=3).try_evaluate() == pytest.approx(6)


def test_root_python_syntax_branches():
    x = kw.Var("x")
    assert "** (1/" in kw.Root(x).python_syntax()
    expression = kw.Root(kw.Log(x), root_by=kw.Log(x + 1), coefficient=kw.Log(x + 2))
    syntax = expression.python_syntax()
    assert syntax.count("log(") == 3
