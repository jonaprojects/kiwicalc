import json
import math

import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    'expression',
    [
        kw.Mono('3x^2*y'),
        kw.Poly('x^2+6x+8'),
        kw.Sin(kw.Var('x')),
        kw.Log(kw.Var('x')),
        kw.Root(kw.Var('x')),
        kw.Abs(kw.Var('x')),
        kw.Exponent(2, kw.Var('x')),
    ],
)
def test_expression_dictionary_round_trip(expression):
    assert kw.create_from_dict(expression.to_dict()) == expression


def test_expression_json_is_valid_and_describes_type():
    payload = json.loads(kw.Poly('x^2+1').to_json())
    assert payload['type'].lower() == 'poly'


def test_when_does_not_mutate_original_expression():
    x = kw.Var('x')
    expression = x**2 + 2*x + 1

    assigned = expression.when(x=3)

    assert expression.variables == {'x'}
    assert assigned.try_evaluate() == 16


@pytest.mark.parametrize('value', [-3, -0.5, 0, 2, math.pi])
def test_symbolic_evaluation_matches_generated_lambda(value):
    x = kw.Var('x')
    expression = x**2 + kw.Sin(x) + 3

    symbolic = expression.when(x=value).try_evaluate()
    executable = expression.to_lambda()(value)

    assert symbolic == pytest.approx(executable)
