"""Safe algebraic and string-defined univariate probability distributions."""

from __future__ import annotations

import ast
import io
import math
import tokenize
from dataclasses import dataclass
from functools import cached_property
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from kiwicalc.core.interfaces import IExpression, IVariable
from kiwicalc.numeric.api import find_root, integrate
from kiwicalc.probability.distributions import (
    ContinuousDistribution,
    DiscreteDistribution,
    _generator,
    _input,
    _quantiles,
    _result,
    _size,
)


_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
_UNARY_FUNCTIONS = {
    'abs': np.abs,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'ln': np.log,
    'log10': np.log10,
    'log2': np.log2,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'asin': np.arcsin,
    'acos': np.arccos,
    'atan': np.arctan,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'asinh': np.arcsinh,
    'acosh': np.arccosh,
    'atanh': np.arctanh,
}
_FUNCTION_NAMES = frozenset(_UNARY_FUNCTIONS) | {'log'}
_RESERVED_NAMES = frozenset(_CONSTANTS) | _FUNCTION_NAMES
_BINARY_OPERATORS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}
_OPERATOR_TEXT = {
    ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**',
}
_NEGATIVE_TOLERANCE = 1e-12
_INTEGRATION_TOLERANCE = 1e-9
_MAX_EVALUATIONS = 20000


def _real_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _name(value, label):
    if not isinstance(value, str) or not value:
        raise TypeError(f'{label} must be a non-empty string')
    if not value.isidentifier():
        raise ValueError(f'{label} must be a valid identifier')
    return value


def _implicit_multiplication(source):
    """Insert multiplication for common mathematical adjacency forms."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError) as exc:
        raise ValueError('formula contains invalid syntax') from exc
    significant = []
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE,
               tokenize.NL, tokenize.INDENT, tokenize.DEDENT}
    for token in tokens:
        if token.type not in ignored:
            significant.append(token)
    result = []
    previous = None
    for token in significant:
        if previous is not None:
            previous_ends_value = (
                previous.type in (tokenize.NUMBER, tokenize.NAME)
                or (previous.type == tokenize.OP and previous.string == ')')
            )
            current_starts_value = (
                token.type in (tokenize.NUMBER, tokenize.NAME)
                or (token.type == tokenize.OP and token.string == '(')
            )
            function_call = (
                previous.type == tokenize.NAME
                and previous.string in _FUNCTION_NAMES
                and token.type == tokenize.OP and token.string == '('
            )
            if previous_ends_value and current_starts_value and not function_call:
                result.append((tokenize.OP, '*'))
        result.append((token.type, token.string))
        previous = token
    return tokenize.untokenize(result).strip()


def _parse(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('formula must not be empty')
    source = source.replace('^', '**')
    source = _implicit_multiplication(source)
    try:
        parsed = ast.parse(source, mode='eval')
    except SyntaxError as exc:
        raise ValueError('formula contains invalid syntax') from exc
    _validate_node(parsed.body)
    return parsed.body


def _validate_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError('formula constants must be real numbers')
        return
    if isinstance(node, ast.Name):
        if node.id in _FUNCTION_NAMES:
            raise ValueError(f'{node.id} must be called with parentheses')
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_NAMES:
            raise ValueError('formula contains an unknown or unsupported function')
        if node.keywords:
            raise ValueError('formula functions do not accept keyword arguments')
        expected = (1, 2) if node.func.id == 'log' else (1,)
        if len(node.args) not in expected:
            count = 'one or two' if node.func.id == 'log' else 'one'
            raise ValueError(f'{node.func.id}() expects {count} argument(s)')
        for argument in node.args:
            _validate_node(argument)
        return
    raise ValueError(f'formula contains unsupported syntax: {type(node).__name__}')


def _symbols(node):
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and item.id not in _RESERVED_NAMES
    }


def _evaluate(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in values:
            return values[node.id]
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else np.negative(value)
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS[type(node.op)]
        return operation(_evaluate(node.left, values), _evaluate(node.right, values))
    arguments = [_evaluate(argument, values) for argument in node.args]
    if node.func.id == 'log':
        if len(arguments) == 1:
            return np.log(arguments[0])
        return np.log(arguments[0]) / np.log(arguments[1])
    return _UNARY_FUNCTIONS[node.func.id](arguments[0])


def _render_number(value):
    return repr(value) if isinstance(value, int) else format(float(value), '.12g')


def _render(node, bindings=None):
    bindings = {} if bindings is None else bindings
    if isinstance(node, ast.Constant):
        return _render_number(node.value)
    if isinstance(node, ast.Name):
        return _render_number(bindings[node.id]) if node.id in bindings else node.id
    if isinstance(node, ast.UnaryOp):
        sign = '+' if isinstance(node.op, ast.UAdd) else '-'
        return f'{sign}({_render(node.operand, bindings)})'
    if isinstance(node, ast.BinOp):
        operator = _OPERATOR_TEXT[type(node.op)]
        return f'({_render(node.left, bindings)} {operator} {_render(node.right, bindings)})'
    arguments = ', '.join(_render(argument, bindings) for argument in node.args)
    return f'{node.func.id}({arguments})'


@dataclass(frozen=True)
class _Formula:
    node: ast.AST
    source: str
    variable: str
    bindings: Mapping

    def evaluate(self, value, *, nonnegative=True):
        input_array = np.asarray(value)
        values = dict(self.bindings)
        values[self.variable] = value
        with np.errstate(all='ignore'):
            result = np.asarray(_evaluate(self.node, values))
        if np.iscomplexobj(result):
            if np.any(np.abs(np.imag(result)) > _NEGATIVE_TOLERANCE):
                raise ValueError('formula must produce real values on its support')
            result = np.real(result)
        try:
            result = np.asarray(result, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('formula must produce real numeric values') from exc
        if result.ndim == 0 and input_array.ndim:
            result = np.full(input_array.shape, result.item(), dtype=float)
        else:
            try:
                result = np.broadcast_to(result, input_array.shape or ()).astype(float, copy=True)
            except ValueError as exc:
                raise ValueError('formula result does not match the input shape') from exc
        if np.any(~np.isfinite(result)):
            raise ValueError('formula must produce finite values across its support')
        if nonnegative:
            if np.any(result < -_NEGATIVE_TOLERANCE):
                raise ValueError('formula must be nonnegative across its support')
            result[result < 0] = 0.0
        return result.item() if result.ndim == 0 else result


def _prepare_formula(formula, variable, parameters):
    declared = None
    if isinstance(formula, IExpression):
        snapshot = formula.__copy__()
        declared = {str(item) for item in snapshot.variables}
        source = snapshot.python_syntax()
    elif isinstance(formula, str):
        source = formula
    elif isinstance(formula, Real) and not isinstance(formula, (bool, np.bool_)):
        source = repr(_real_number(formula, 'formula'))
    else:
        raise TypeError('formula must be an IExpression, a string, or a real number')

    node = _parse(source)
    symbols = _symbols(node)
    if declared is not None and symbols != declared:
        raise ValueError('expression variables do not match its executable formula')

    if parameters is None:
        bindings = {}
    elif not isinstance(parameters, Mapping):
        raise TypeError('parameters must be a mapping of names to real values')
    else:
        bindings = {}
        for key, value in parameters.items():
            key = _name(key, 'parameter name')
            if key in _RESERVED_NAMES:
                raise ValueError(f"parameter name '{key}' is reserved")
            if key not in symbols:
                raise ValueError(f"unknown parameter '{key}'")
            bindings[key] = _real_number(value, f"parameter '{key}'")

    if variable is None:
        unbound = symbols - set(bindings)
        if len(unbound) > 1:
            names = ', '.join(sorted(unbound))
            raise ValueError(
                f'formula has multiple unbound symbols: {names}; '
                'provide variable= and parameters='
            )
        variable_name = next(iter(unbound), 'x')
    elif isinstance(variable, str):
        variable_name = _name(variable, 'variable')
    elif isinstance(variable, IVariable):
        variable_name = _name(variable.name, 'variable name')
    else:
        raise TypeError('variable must be a string or an IVariable-compatible object')

    if variable_name in _RESERVED_NAMES:
        raise ValueError(f"variable name '{variable_name}' is reserved")
    if variable_name in bindings:
        raise ValueError('the distribution variable cannot also be a parameter')
    unresolved = symbols - set(bindings) - {variable_name}
    if unresolved:
        names = ', '.join(sorted(unresolved))
        raise ValueError(f'unresolved parameter(s): {names}')
    canonical = _render(node)
    return _Formula(node, canonical, variable_name, MappingProxyType(dict(bindings)))


class _FormulaDistribution:
    def _set_formula_metadata(self, prepared, normalization, name):
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise TypeError('name must be a non-empty string or None')
        self._formula_data = prepared
        self._normalization_constant = float(normalization)
        self._name = name.strip() if name is not None else None

    @property
    def formula(self):
        return self._formula_data.source

    @property
    def variable(self):
        return self._formula_data.variable

    @property
    def parameters(self):
        return dict(self._formula_data.bindings)

    @property
    def normalization_constant(self):
        return self._normalization_constant

    @property
    def was_normalized(self):
        return math.isclose(self.normalization_constant, 1.0, rel_tol=1e-9, abs_tol=1e-12)

    @property
    def normalized_formula(self):
        bound = _render(self._formula_data.node, self._formula_data.bindings)
        if self.was_normalized:
            return bound
        constant = format(self.normalization_constant, '.12g')
        return f'({bound}) / {constant}'

    @property
    def name(self):
        return self._name


class ContinuousFormulaDistribution(_FormulaDistribution, ContinuousDistribution):
    """A normalized density created from a formula on finite bounds."""

    def __init__(self, formula, between, *, variable=None, parameters=None, name=None):
        try:
            lower, upper = between
        except (TypeError, ValueError) as exc:
            raise ValueError('between must contain exactly two finite bounds') from exc
        lower = _real_number(lower, 'lower bound')
        upper = _real_number(upper, 'upper bound')
        if lower >= upper:
            raise ValueError('the lower bound must be smaller than the upper bound')
        prepared = _prepare_formula(formula, variable, parameters)
        prepared.evaluate(np.linspace(lower, upper, 513))
        normalization = integrate(
            lambda value: prepared.evaluate(value), lower, upper,
            method='adaptive_simpson', tolerance=_INTEGRATION_TOLERANCE,
            max_evaluations=_MAX_EVALUATIONS,
        )
        if not math.isfinite(normalization) or normalization <= 0:
            raise ValueError('formula must have a positive finite integral over its support')
        self._support = (lower, upper)
        self._set_formula_metadata(prepared, normalization, name)

    @property
    def support(self):
        return self._support

    def _kernel(self, value):
        return self._formula_data.evaluate(value) / self.normalization_constant

    def pdf(self, value):
        values, scalar = _input(value)
        result = np.zeros(values.shape, dtype=float)
        nan = np.isnan(values)
        result[nan] = np.nan
        inside = (~nan) & (values >= self.support[0]) & (values <= self.support[1])
        if np.any(inside):
            result[inside] = self._kernel(values[inside])
        return _result(result, scalar)

    def _cdf_scalar(self, value):
        if value <= self.support[0]:
            return 0.0
        if value >= self.support[1]:
            return 1.0
        result = integrate(
            self._kernel, self.support[0], value,
            method='adaptive_simpson', tolerance=_INTEGRATION_TOLERANCE,
            max_evaluations=_MAX_EVALUATIONS,
        )
        return min(1.0, max(0.0, float(result)))

    def cdf(self, value):
        values, scalar = _input(value)
        result = np.fromiter(
            (math.nan if math.isnan(float(item)) else self._cdf_scalar(float(item))
             for item in values.reshape(-1)),
            dtype=float, count=values.size,
        ).reshape(values.shape)
        return _result(result, scalar)

    def ppf(self, probability):
        probabilities, scalar = _quantiles(probability)

        def inverse(item):
            item = float(item)
            if item == 0:
                return self.support[0]
            if item == 1:
                return self.support[1]
            return find_root(
                lambda value: self._cdf_scalar(value) - item,
                bracket=self.support, method='brent', tolerance=1e-10,
                max_iterations=200,
            )

        result = np.fromiter(
            (inverse(item) for item in probabilities.reshape(-1)),
            dtype=float, count=probabilities.size,
        ).reshape(probabilities.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        probabilities = _generator(random_state).random(size=_size(size))
        return self.ppf(probabilities)

    rvs = sample

    @cached_property
    def mean(self):
        return integrate(
            lambda value: value * self._kernel(value), *self.support,
            method='adaptive_simpson', tolerance=_INTEGRATION_TOLERANCE,
            max_evaluations=_MAX_EVALUATIONS,
        )

    @cached_property
    def variance(self):
        center = self.mean
        return integrate(
            lambda value: (value - center) ** 2 * self._kernel(value), *self.support,
            method='adaptive_simpson', tolerance=_INTEGRATION_TOLERANCE,
            max_evaluations=_MAX_EVALUATIONS,
        )

    def __repr__(self):
        name = f', name={self.name!r}' if self.name is not None else ''
        return (
            f'ContinuousFormulaDistribution(formula={self.formula!r}, '
            f'between={self.support!r}, variable={self.variable!r}{name})'
        )


class DiscreteFormulaDistribution(_FormulaDistribution, DiscreteDistribution):
    """A normalized probability mass function on finite numeric outcomes."""

    def __init__(self, formula, over, *, variable=None, parameters=None, name=None):
        if isinstance(over, (str, bytes)):
            raise TypeError('over must be a finite iterable of real outcomes')
        try:
            raw_values = list(over)
        except TypeError as exc:
            raise TypeError('over must be a finite iterable of real outcomes') from exc
        if not raw_values:
            raise ValueError('over cannot be empty')
        values = [_real_number(item, 'support value') for item in raw_values]
        if len(set(values)) != len(values):
            raise ValueError('support values must be distinct')
        values = np.asarray(sorted(values), dtype=float)
        prepared = _prepare_formula(formula, variable, parameters)
        weights = np.asarray(prepared.evaluate(values), dtype=float)
        normalization = float(np.sum(weights))
        if not math.isfinite(normalization) or normalization <= 0:
            raise ValueError('formula must have a positive finite sum over its support')
        self._values = values
        self._probabilities = weights / normalization
        self._cumulative = np.cumsum(self._probabilities)
        self._cumulative[-1] = 1.0
        self._lookup = dict(zip(self._values.tolist(), self._probabilities.tolist()))
        self._set_formula_metadata(prepared, normalization, name)

    @property
    def support(self):
        return float(self._values[0]), float(self._values[-1])

    @property
    def values(self):
        return tuple(self._values.tolist())

    @property
    def probabilities(self):
        return tuple(self._probabilities.tolist())

    @property
    def mean(self):
        return float(np.dot(self._values, self._probabilities))

    @property
    def variance(self):
        return float(np.dot((self._values - self.mean) ** 2, self._probabilities))

    def pmf(self, value):
        values, scalar = _input(value)
        result = np.fromiter(
            (math.nan if math.isnan(float(item)) else self._lookup.get(float(item), 0.0)
             for item in values.reshape(-1)),
            dtype=float, count=values.size,
        ).reshape(values.shape)
        return _result(result, scalar)

    def cdf(self, value):
        values, scalar = _input(value)

        def cumulative(item):
            item = float(item)
            if math.isnan(item):
                return math.nan
            index = int(np.searchsorted(self._values, item, side='right')) - 1
            return 0.0 if index < 0 else float(self._cumulative[index])

        result = np.fromiter(
            (cumulative(item) for item in values.reshape(-1)),
            dtype=float, count=values.size,
        ).reshape(values.shape)
        return _result(result, scalar)

    def ppf(self, probability):
        probabilities, scalar = _quantiles(probability)
        indices = np.searchsorted(self._cumulative, probabilities, side='left')
        indices = np.minimum(indices, len(self._values) - 1)
        return _result(self._values[indices], scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).choice(
            self._values, size=_size(size), p=self._probabilities,
        )

    rvs = sample

    def probability_between(self, lower, upper, *, inclusive='both'):
        if inclusive not in {'both', 'left', 'right', 'neither'}:
            raise ValueError("inclusive must be 'both', 'left', 'right', or 'neither'")
        lower = _real_number(lower, 'lower')
        upper = _real_number(upper, 'upper')
        if lower > upper:
            raise ValueError('lower cannot exceed upper')
        left = self._values >= lower if inclusive in {'both', 'left'} else self._values > lower
        right = self._values <= upper if inclusive in {'both', 'right'} else self._values < upper
        return float(np.sum(self._probabilities[left & right]))

    def __repr__(self):
        name = f', name={self.name!r}' if self.name is not None else ''
        return (
            f'DiscreteFormulaDistribution(formula={self.formula!r}, '
            f'over={self.values!r}, variable={self.variable!r}{name})'
        )


def distribution(
        formula: Union[IExpression, str, Real], *,
        between: Optional[Tuple[Real, Real]] = None,
        over: Optional[Iterable[Real]] = None,
        variable: Optional[Union[IVariable, str]] = None,
        parameters: Optional[Mapping[str, Real]] = None,
        name: Optional[str] = None,
) -> Union[ContinuousFormulaDistribution, DiscreteFormulaDistribution]:
    """Create a normalized distribution from a friendly mathematical formula.

    Pass ``between=(lower, upper)`` for a bounded continuous density or
    ``over=values`` for a finite discrete probability mass function.
    """
    if (between is None) == (over is None):
        raise ValueError('provide exactly one of between= or over=')
    if between is not None:
        return ContinuousFormulaDistribution(
            formula, between, variable=variable, parameters=parameters, name=name,
        )
    return DiscreteFormulaDistribution(
        formula, over, variable=variable, parameters=parameters, name=name,
    )


__all__ = [
    'ContinuousFormulaDistribution', 'DiscreteFormulaDistribution', 'distribution',
]
