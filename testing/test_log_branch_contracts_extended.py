import math

import pytest

import kiwicalc as kw


def test_log_constructor_validation_and_constant_branches():
    with pytest.raises(ValueError):
        kw.Log(-1)
    with pytest.raises(ValueError):
        kw.Log(0)
    with pytest.raises(TypeError):
        kw.Log(kw.Var("x"), coefficient=object())
    assert kw.Log(100) == 2
    assert kw.Ln(math.e) == 1
    assert kw.Log("2log(x,10)").coefficient == 2


def test_log_metadata_simplification_and_assignment_branches():
    x = kw.Var("x")
    expression = kw.Log([[x, 10, 2], [x + 1, 2, 1]], coefficient=3)
    assert expression.index_of([x, 10, 2]) == 0
    assert expression.index_of([x, 3, 1]) == -1
    assert expression.all_bases() == {2, 10}
    assert expression.biggest_power() == 2
    assert expression.variables == {"x"}
    expression.simplify()
    assigned = expression.when(x=3)
    assert assigned.try_evaluate() == pytest.approx(3 * math.log(3, 10) ** 2 * math.log(4, 2), abs=1e-5)


def test_log_addition_branches():
    x = kw.Var("x")
    assert kw.Log(x) + 0 == kw.Log(x)
    assert kw.Log(100) + 3 == 5
    assert isinstance(kw.Log(x) + 3, kw.ExpressionSum)
    assert kw.Log(x) + kw.Log(x + 1) == kw.Log(x * (x + 1))
    unequal = kw.Log(x, coefficient=2) + kw.Log(x + 1, coefficient=3)
    assert isinstance(unequal, kw.Log)
    assert unequal == kw.Log(x**2 * (x + 1) ** 3)
    assert isinstance(kw.Log(x) + kw.Sin(x), kw.ExpressionSum)


def test_log_subtraction_branches():
    x = kw.Var("x")
    assert kw.Log(100) - 1 == 1
    symbolic_number = kw.Log(x) - 2
    assert symbolic_number.when(x=100).try_evaluate() == pytest.approx(0)
    quotient = kw.Log(x) - kw.Log(x + 1)
    assert quotient.when(x=9).try_evaluate() == pytest.approx(math.log(9 / 10, 10), abs=1e-5)
    unequal = kw.Log(x, coefficient=2) - kw.Log(x + 1, coefficient=3)
    assert isinstance(unequal, kw.ExpressionSum)


def test_log_multiplication_branches():
    x = kw.Var("x")
    assert kw.Log(x) * kw.Log(100) == kw.Log(x, coefficient=2)
    squared = kw.Log(x) * kw.Log(x)
    assert squared._expressions[0][2] == 2
    product = kw.Log(x) * kw.Log(x + 1)
    assert len(product._expressions) == 2
    assert (kw.Log(x) * 3).coefficient == 3
    assert (3 * kw.Log(x)).coefficient == 3


def test_log_division_branches():
    x = kw.Var("x")
    with pytest.raises(ZeroDivisionError):
        kw.Log(x) / 0
    assert (kw.Log(x, coefficient=4) / 2).coefficient == 2
    assert kw.Log(100) / kw.Mono(2) == 1
    with pytest.raises(ZeroDivisionError):
        kw.Log(100) / kw.Mono(0)
    assert (kw.Log(x, coefficient=4) / kw.Mono(2)).coefficient == 2
    assert isinstance(kw.Log(x) / kw.Sin(x), kw.Fraction)


def test_log_power_negation_and_copy_branches():
    x = kw.Var("x")
    original = kw.Log(x, coefficient=2)
    powered = original**2
    assert powered.coefficient == 4 and powered._expressions[0][2] == 2
    negative = -original
    assert negative.coefficient == -2
    assert original.coefficient == 2
    copied = original.__copy__()
    assert copied == original and copied is not original


def test_log_evaluation_unknown_base_and_power_branches():
    x, b, p = kw.Var("x"), kw.Var("b"), kw.Var("p")
    assert kw.Log(x).try_evaluate() is None
    assert kw.Log(x, base=b).when(x=8, b=2).try_evaluate() == 3
    powered = kw.Log([[kw.Mono(8), 2, p]])
    assert powered.try_evaluate() is None
    assert powered.when(p=2).try_evaluate() == 9
    constant = kw.Log(100)
    assert constant.try_evaluate() == 2


def test_log_string_python_and_serialization_branches():
    x = kw.Var("x")
    assert kw.Log([[x, 10, 0]])._single_log_str(x, 10, 0) == "1"
    assert kw.Log([[x, math.e, 1]])._single_log_str(x, math.e, 1) == "ln(x)"
    assert "^2" in kw.Log([[x, 10, 2]])._single_log_str(x, 10, 2)
    assert str(kw.Log(x, coefficient=0)) == "0"
    assert str(kw.Log(100)) == "2"
    assert kw.Log(x, coefficient=-1).python_syntax().startswith("-")
    assert "** 2" in kw.Log([[x, 10, kw.Mono(2)]]).python_syntax()
    expression = kw.Log([[x, 10, kw.Mono(2)]], coefficient=3)
    assert kw.Log.from_dict(expression.to_dict()) == expression


def test_log_equality_error_and_symbolic_branches():
    x = kw.Var("x")
    assert kw.Log(x) == kw.Log(x)
    assert kw.Log(x) != kw.Log(x + 1)
    assert kw.Log(x, coefficient=2) != kw.Log(x, coefficient=3)
    assert kw.Log(x) != kw.Sin(x)
    assert kw.Log(x) != None  # noqa: E711
    with pytest.raises(TypeError):
        kw.Log(x) == object()
