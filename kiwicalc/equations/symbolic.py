"""Native structured equation solving.

This module is deliberately additive.  Legacy equation functions retain their
historical signatures and containers; :func:`solve_equation` provides the
exact, structured API.
"""
from __future__ import annotations

import cmath
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction as Rational
from functools import reduce
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from kiwicalc.parsing.errors import (
    AmbiguousVariableError,
    EquationParseError,
    UnsupportedExpressionError,
)


# ---------------------------------------------------------------------------
# Immutable symbolic representation


class SymbolicExpression:
    """Marker base for the immutable symbolic representation."""

    def evaluate(self, values: Optional[Mapping[str, complex]] = None):
        return _evaluate(self, values or {})

    def substitute(self, values: Mapping[str, Any]):
        return _substitute(self, values)

    @property
    def variables(self):
        return frozenset(_variables(self))

    def to_dict(self):
        return _expression_to_dict(self)


@dataclass(frozen=True)
class ExactNumber(SymbolicExpression):
    value: Rational

    def __init__(self, numerator=0, denominator=None):
        value = numerator if isinstance(numerator, Rational) and denominator is None else Rational(numerator, denominator) if denominator is not None else Rational(numerator)
        object.__setattr__(self, "value", value)

    @property
    def numerator(self):
        return self.value.numerator

    @property
    def denominator(self):
        return self.value.denominator

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(float(self.value))

    def __str__(self):
        return str(self.numerator) if self.denominator == 1 else f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class Symbol(SymbolicExpression):
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not re.fullmatch(r"[A-Za-z_]\w*", self.name):
            raise ValueError("Symbol names must be non-empty identifiers")

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class SymbolicConstant(SymbolicExpression):
    name: str

    def __post_init__(self):
        if self.name not in {"pi", "e", "i"}:
            raise ValueError(f"Unknown symbolic constant {self.name!r}")

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Add(SymbolicExpression):
    terms: Tuple[SymbolicExpression, ...]

    def __post_init__(self):
        terms = tuple(self.terms)
        if not terms or any(not isinstance(term, SymbolicExpression) for term in terms):
            raise TypeError("Add terms must be a nonempty sequence of symbolic expressions")
        object.__setattr__(self, "terms", terms)

    def __str__(self):
        text = ""
        for index, term in enumerate(self.terms):
            rendered = str(term)
            if index and rendered.startswith("-"):
                text += f" - {rendered[1:]}"
            else:
                text += (" + " if index else "") + rendered
        return text or "0"


@dataclass(frozen=True)
class Multiply(SymbolicExpression):
    factors: Tuple[SymbolicExpression, ...]

    def __post_init__(self):
        factors = tuple(self.factors)
        if not factors or any(not isinstance(factor, SymbolicExpression) for factor in factors):
            raise TypeError("Multiply factors must be a nonempty sequence of symbolic expressions")
        object.__setattr__(self, "factors", factors)

    def __str__(self):
        def render(item):
            return f"({item})" if isinstance(item, Add) else str(item)
        return "*".join(render(item) for item in self.factors) or "1"


@dataclass(frozen=True)
class Power(SymbolicExpression):
    base: SymbolicExpression
    exponent: SymbolicExpression

    def __post_init__(self):
        if not isinstance(self.base, SymbolicExpression) or not isinstance(self.exponent, SymbolicExpression):
            raise TypeError("Power base and exponent must be symbolic expressions")

    def __str__(self):
        base = f"({self.base})" if isinstance(self.base, (Add, Multiply)) else str(self.base)
        exponent = f"({self.exponent})" if isinstance(self.exponent, (Add, Multiply)) else str(self.exponent)
        return f"{base}^{exponent}"


@dataclass(frozen=True)
class SymbolicFunction(SymbolicExpression):
    name: str
    arguments: Tuple[SymbolicExpression, ...]

    def __post_init__(self):
        arguments = tuple(self.arguments)
        if self.name not in _FUNCTIONS:
            raise ValueError(f"Unsupported symbolic function {self.name!r}")
        if not arguments or any(not isinstance(argument, SymbolicExpression) for argument in arguments):
            raise TypeError("Function arguments must be a nonempty sequence of symbolic expressions")
        if self.name == "log" and len(arguments) not in {1, 2} or self.name != "log" and len(arguments) != 1:
            raise ValueError(f"Invalid number of arguments for {self.name}")
        object.__setattr__(self, "arguments", arguments)

    def __str__(self):
        return f"{self.name}({', '.join(map(str, self.arguments))})"


@dataclass(frozen=True)
class RootOf(SymbolicExpression):
    coefficients: Tuple[ExactNumber, ...]
    index: int
    interval: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        coefficients = tuple(self.coefficients)
        if len(coefficients) < 2 or any(not isinstance(value, ExactNumber) for value in coefficients):
            raise TypeError("RootOf coefficients must contain at least two exact numbers")
        if coefficients[0] == ZERO:
            raise ValueError("RootOf leading coefficient cannot be zero")
        degree = len(coefficients) - 1
        if isinstance(self.index, bool) or not isinstance(self.index, int) or not 0 <= self.index < degree:
            raise ValueError("RootOf index must identify a polynomial root")
        interval = None if self.interval is None else tuple(self.interval)
        if interval is not None and (len(interval) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in interval) or interval[0] >= interval[1]):
            raise ValueError("RootOf interval must be an increasing finite pair")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "interval", interval)

    def __str__(self):
        return f"RootOf(({', '.join(map(str, self.coefficients))}), {self.index})"


ZERO, ONE, NEG_ONE = ExactNumber(0), ExactNumber(1), ExactNumber(-1)
PI, E, I = SymbolicConstant("pi"), SymbolicConstant("e"), SymbolicConstant("i")


# ---------------------------------------------------------------------------
# Structural assumptions and domain conditions


class Condition:
    """Immutable predicate used to qualify symbolic transformations.

    ``evaluate`` deliberately returns ``None`` when the supplied assignments do
    not determine the predicate.  This three-valued behaviour prevents a
    parameter condition from being mistaken for either true or false.
    """

    def evaluate(self, values: Optional[Mapping[str, Any]] = None, *, tolerance=1e-12):
        raise NotImplementedError

    def substitute(self, values: Mapping[str, Any]):
        raise NotImplementedError

    @property
    def variables(self):
        return frozenset()

    def to_dict(self):
        return _condition_to_dict(self)


def _condition_tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError("condition tolerance must be nonnegative and finite")
    return float(value)


def _condition_values(values):
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("condition values must be a mapping")
    return values


@dataclass(frozen=True)
class TruthCondition(Condition):
    value: bool

    def __post_init__(self):
        if not isinstance(self.value, bool):
            raise TypeError("TruthCondition value must be a boolean")

    def evaluate(self, values=None, *, tolerance=1e-12):
        _condition_tolerance(tolerance)
        _condition_values(values)
        return self.value

    def substitute(self, values):
        _condition_values(values)
        return self

    def __str__(self):
        return "True" if self.value else "False"


@dataclass(frozen=True)
class RelationCondition(Condition):
    left: SymbolicExpression
    relation: str
    right: SymbolicExpression = ZERO
    display: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.left, SymbolicExpression) or not isinstance(self.right, SymbolicExpression):
            raise TypeError("RelationCondition operands must be symbolic expressions")
        if self.relation not in {"=", "!=", ">", ">=", "<", "<="}:
            raise ValueError("Unsupported relation operator")
        if self.display is not None and (not isinstance(self.display, str) or not self.display):
            raise ValueError("RelationCondition display must be nonempty text")

    @property
    def variables(self):
        return frozenset(_variables(self.left) | _variables(self.right))

    def evaluate(self, values=None, *, tolerance=1e-12):
        tolerance = _condition_tolerance(tolerance)
        values = _condition_values(values)
        if not self.variables.issubset(values):
            return None
        try:
            left = complex(_evaluate(self.left, values))
            right = complex(_evaluate(self.right, values))
        except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(item) for item in (left.real, left.imag, right.real, right.imag)):
            return False
        difference = left - right
        scale = max(1.0, abs(left), abs(right))
        threshold = float(tolerance) * scale
        if self.relation == "=":
            return abs(difference) <= threshold
        if self.relation == "!=":
            return abs(difference) > threshold
        if abs(left.imag) > threshold or abs(right.imag) > threshold:
            return False
        if self.relation == ">":
            return left.real > right.real + threshold
        if self.relation == ">=":
            return left.real >= right.real - threshold
        if self.relation == "<":
            return left.real < right.real - threshold
        return left.real <= right.real + threshold

    def substitute(self, values):
        values = _condition_values(values)
        return simplify_condition(RelationCondition(
            _substitute(self.left, values), self.relation, _substitute(self.right, values)
        ))

    def __str__(self):
        return self.display or f"{self.left} {self.relation} {self.right}"


@dataclass(frozen=True)
class DefinedCondition(Condition):
    expression: SymbolicExpression
    domain: str = "real"

    def __post_init__(self):
        if not isinstance(self.expression, SymbolicExpression):
            raise TypeError("DefinedCondition expression must be symbolic")
        if self.domain not in {"real", "complex"}:
            raise ValueError("DefinedCondition domain must be 'real' or 'complex'")

    @property
    def variables(self):
        return frozenset(_variables(self.expression))

    def evaluate(self, values=None, *, tolerance=1e-12):
        tolerance = _condition_tolerance(tolerance)
        values = _condition_values(values)
        if not self.variables.issubset(values):
            return None
        try:
            value = complex(_evaluate(self.expression, values))
        except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            return False
        return self.domain == "complex" or abs(value.imag) <= tolerance * max(1.0, abs(value))

    def substitute(self, values):
        values = _condition_values(values)
        result = DefinedCondition(_substitute(self.expression, values), self.domain)
        evaluated = result.evaluate({}, tolerance=0.0)
        return result if evaluated is None else TruthCondition(evaluated)

    def __str__(self):
        return f"{self.expression} is defined in the {self.domain} domain"


@dataclass(frozen=True)
class BetweenCondition(Condition):
    expression: SymbolicExpression
    lower: SymbolicExpression
    upper: SymbolicExpression
    lower_closed: bool = True
    upper_closed: bool = True
    display: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        if any(not isinstance(value, SymbolicExpression) for value in (self.expression, self.lower, self.upper)):
            raise TypeError("BetweenCondition values must be symbolic expressions")
        if not isinstance(self.lower_closed, bool) or not isinstance(self.upper_closed, bool):
            raise TypeError("BetweenCondition closure flags must be booleans")
        if self.display is not None and (not isinstance(self.display, str) or not self.display):
            raise ValueError("BetweenCondition display must be nonempty text")

    @property
    def variables(self):
        return frozenset(_variables(self.expression) | _variables(self.lower) | _variables(self.upper))

    def evaluate(self, values=None, *, tolerance=1e-12):
        tolerance = _condition_tolerance(tolerance)
        values = _condition_values(values)
        lower_relation = ">=" if self.lower_closed else ">"
        upper_relation = "<=" if self.upper_closed else "<"
        first = RelationCondition(self.expression, lower_relation, self.lower).evaluate(values, tolerance=tolerance)
        second = RelationCondition(self.expression, upper_relation, self.upper).evaluate(values, tolerance=tolerance)
        return False if False in (first, second) else None if None in (first, second) else True

    def substitute(self, values):
        values = _condition_values(values)
        result = BetweenCondition(
            _substitute(self.expression, values), _substitute(self.lower, values), _substitute(self.upper, values),
            self.lower_closed, self.upper_closed,
        )
        evaluated = result.evaluate({}, tolerance=0.0)
        return result if evaluated is None else TruthCondition(evaluated)

    def __str__(self):
        if self.display:
            return self.display
        lower = "<=" if self.lower_closed else "<"
        upper = "<=" if self.upper_closed else "<"
        return f"{self.lower} {lower} {self.expression} {upper} {self.upper}"


@dataclass(frozen=True)
class OpaqueCondition(Condition):
    """Compatibility wrapper for conditions serialized by older releases."""

    text: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("OpaqueCondition text must be a nonempty string")

    def evaluate(self, values=None, *, tolerance=1e-12):
        _condition_tolerance(tolerance)
        _condition_values(values)
        if self.text == "True":
            return True
        if self.text == "False":
            return False
        return None

    def substitute(self, values):
        _condition_values(values)
        return self

    def __str__(self):
        return self.text


@dataclass(frozen=True)
class CompoundCondition(Condition):
    operator: str
    conditions: Tuple[Condition, ...]

    def __post_init__(self):
        conditions = tuple(self.conditions)
        if self.operator not in {"and", "or", "not"}:
            raise ValueError("Compound condition operator must be 'and', 'or', or 'not'")
        if not conditions or any(not isinstance(value, Condition) for value in conditions):
            raise TypeError("Compound conditions must contain predicates")
        if self.operator == "not" and len(conditions) != 1:
            raise ValueError("A 'not' condition must contain exactly one predicate")
        object.__setattr__(self, "conditions", conditions)

    @property
    def variables(self):
        return frozenset().union(*(condition.variables for condition in self.conditions))

    def evaluate(self, values=None, *, tolerance=1e-12):
        tolerance = _condition_tolerance(tolerance)
        values = _condition_values(values)
        results = tuple(condition.evaluate(values, tolerance=tolerance) for condition in self.conditions)
        if self.operator == "not":
            return None if results[0] is None else not results[0]
        if self.operator == "and":
            return False if False in results else None if None in results else True
        return True if True in results else None if None in results else False

    def substitute(self, values):
        values = _condition_values(values)
        return simplify_condition(CompoundCondition(
            self.operator, tuple(condition.substitute(values) for condition in self.conditions)
        ))

    def __str__(self):
        if self.operator == "not":
            return f"not ({self.conditions[0]})"
        separator = f" {self.operator} "
        return separator.join(f"({condition})" for condition in self.conditions)


def parse_condition(text):
    """Parse a simple relational or domain predicate.

    This intentionally accepts only atomic predicates.  Boolean composition is
    represented explicitly with :class:`CompoundCondition`, avoiding ambiguous
    precedence in user-provided condition strings.
    """
    if not isinstance(text, str) or not text.strip():
        raise EquationParseError("Condition must be nonempty text")
    text = text.strip()
    if text in {"True", "False"}:
        return TruthCondition(text == "True")
    defined = re.fullmatch(r"(.+?)\s+is defined in the\s+(real|complex)\s+domain", text)
    if defined:
        return DefinedCondition(parse_symbolic(defined.group(1)), defined.group(2))
    chained = re.fullmatch(r"(.+?)\s*(<=|<)\s*(.+?)\s*(<=|<)\s*(.+)", text)
    if chained:
        lower, lower_operator, expression, upper_operator, upper = chained.groups()
        return BetweenCondition(
            parse_symbolic(expression), parse_symbolic(lower), parse_symbolic(upper),
            lower_operator == "<=", upper_operator == "<=", text,
        )
    relation = re.fullmatch(r"(.+?)\s*(<=|>=|!=|=|<|>)\s*(.+)", text)
    if relation:
        left, operator, right = relation.groups()
        return RelationCondition(parse_symbolic(left), operator, parse_symbolic(right), text)
    raise EquationParseError("Condition must be an atomic relation or domain predicate")


def _coerce_condition(value):
    if isinstance(value, Condition):
        return value
    if isinstance(value, str) and value:
        try:
            return parse_condition(value)
        except (EquationParseError, UnsupportedExpressionError, TypeError, ValueError):
            return OpaqueCondition(value)
    raise ValueError("conditions must contain predicates or nonempty strings")


def simplify_condition(condition, values=None, *, tolerance=1e-12):
    """Simplify a predicate structurally and with optional assignments."""
    tolerance = _condition_tolerance(tolerance)
    condition = _coerce_condition(condition)
    if values:
        condition = condition.substitute(values)
    if isinstance(condition, RelationCondition):
        result = condition.evaluate({}, tolerance=tolerance)
        return condition if result is None else TruthCondition(result)
    if isinstance(condition, DefinedCondition):
        result = condition.evaluate({}, tolerance=tolerance)
        return condition if result is None else TruthCondition(result)
    if isinstance(condition, BetweenCondition):
        lower, upper = _numeric_value(condition.lower), _numeric_value(condition.upper)
        if lower is not None and upper is not None and not isinstance(lower, complex) and not isinstance(upper, complex):
            if lower > upper or lower == upper and (not condition.lower_closed or not condition.upper_closed):
                return TruthCondition(False)
        result = condition.evaluate({}, tolerance=tolerance)
        return condition if result is None else TruthCondition(result)
    if not isinstance(condition, CompoundCondition):
        return condition
    simplified = tuple(simplify_condition(item, tolerance=tolerance) for item in condition.conditions)
    if condition.operator == "not":
        item = simplified[0]
        return TruthCondition(not item.value) if isinstance(item, TruthCondition) else CompoundCondition("not", (item,))
    flat = []
    for item in simplified:
        if isinstance(item, CompoundCondition) and item.operator == condition.operator:
            flat.extend(item.conditions)
        else:
            flat.append(item)
    if condition.operator == "and":
        if any(isinstance(item, TruthCondition) and not item.value for item in flat):
            return TruthCondition(False)
        flat = [item for item in flat if not isinstance(item, TruthCondition)]
    else:
        if any(isinstance(item, TruthCondition) and item.value for item in flat):
            return TruthCondition(True)
        flat = [item for item in flat if not isinstance(item, TruthCondition)]
    unique = tuple(dict.fromkeys(flat))
    if not unique:
        return TruthCondition(condition.operator == "and")
    return unique[0] if len(unique) == 1 else CompoundCondition(condition.operator, unique)


_RELATION_NEGATIONS = {"=": "!=", "!=": "=", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}


def _relation_implies(given, target):
    if given.left != target.left:
        if given.relation == target.relation == "!=":
            given_right, target_right = _numeric_value(given.right), _numeric_value(target.right)
            variables = given.left.variables | target.left.variables
            if given_right == target_right == 0 and len(variables) == 1:
                variable = next(iter(variables))
                try:
                    given_poly = _poly_fraction(given.left, variable)
                    target_poly = _poly_fraction(target.left, variable)
                    if (
                        given_poly is not None and target_poly is not None
                        and given_poly[1] and target_poly[1]
                        and max(given_poly[1]) == max(target_poly[1]) == 0
                        and target_poly[0]
                    ):
                        _, remainder = _poly_divmod(given_poly[0], target_poly[0])
                        if not remainder:
                            return True
                except (UnsupportedExpressionError, ZeroDivisionError):
                    pass
        return False
    given_right, target_right = _numeric_value(given.right), _numeric_value(target.right)
    if given_right is None or target_right is None or isinstance(given_right, complex) or isinstance(target_right, complex):
        return given == target
    given_right, target_right = float(given_right), float(target_right)
    if given.relation == "=":
        probe = RelationCondition(ExactNumber(str(given_right)), target.relation, ExactNumber(str(target_right)))
        return bool(probe.evaluate({}, tolerance=0.0))
    if given.relation == "!=":
        return target.relation == "!=" and given_right == target_right
    if given.relation == ">":
        return (
            target.relation in {">", ">="} and given_right >= target_right
            or target.relation == "!=" and target_right <= given_right
        )
    if given.relation == ">=":
        return (
            target.relation == ">=" and given_right >= target_right
            or target.relation == ">" and given_right > target_right
            or target.relation == "!=" and target_right < given_right
        )
    if given.relation == "<":
        return (
            target.relation in {"<", "<="} and given_right <= target_right
            or target.relation == "!=" and target_right >= given_right
        )
    return (
        target.relation == "<=" and given_right <= target_right
        or target.relation == "<" and given_right < target_right
        or target.relation == "!=" and target_right > given_right
    )


@dataclass(frozen=True)
class AssumptionSet:
    """Canonical immutable conjunction of structural conditions."""

    conditions: Tuple[Condition, ...] = ()

    def __post_init__(self):
        flattened = []
        for raw in self.conditions:
            condition = simplify_condition(raw)
            if isinstance(condition, CompoundCondition) and condition.operator == "and":
                flattened.extend(condition.conditions)
            elif isinstance(condition, TruthCondition) and condition.value:
                continue
            else:
                flattened.append(condition)
        object.__setattr__(self, "conditions", tuple(dict.fromkeys(flattened)))

    def __iter__(self):
        return iter(self.conditions)

    def __len__(self):
        return len(self.conditions)

    def __bool__(self):
        return bool(self.conditions)

    def __contains__(self, value):
        if isinstance(value, str):
            return value in self.rendered
        return value in self.conditions

    @property
    def rendered(self):
        return tuple(str(condition) for condition in self.conditions)

    @property
    def variables(self):
        return frozenset().union(*(condition.variables for condition in self.conditions)) if self.conditions else frozenset()

    @property
    def substitutions(self):
        """Known symbol values implied by explicit equality predicates."""
        result = {}
        for condition in self.conditions:
            if not isinstance(condition, RelationCondition) or condition.relation != "=":
                continue
            if isinstance(condition.left, Symbol) and condition.left.name not in _variables(condition.right):
                result[condition.left.name] = condition.right
            elif isinstance(condition.right, Symbol) and condition.right.name not in _variables(condition.left):
                result[condition.right.name] = condition.left
        # Resolve finite acyclic chains such as ``a = b, b = 2``.  Cycles are
        # retained symbolically rather than expanded indefinitely.
        for _ in range(len(result)):
            changed = False
            for name, expression in tuple(result.items()):
                replacements = {key: value for key, value in result.items() if key != name}
                candidate = _substitute(expression, replacements)
                if name not in _variables(candidate) and candidate != expression:
                    result[name] = candidate
                    changed = True
            if not changed:
                break
        return MappingProxyType(result)

    @property
    def contradictory(self):
        if any(isinstance(condition, TruthCondition) and not condition.value for condition in self.conditions):
            return True
        for condition in self.conditions:
            if isinstance(condition, CompoundCondition) and condition.operator == "not" and condition.conditions[0] in self.conditions:
                return True
        relations = [condition for condition in self.conditions if isinstance(condition, RelationCondition)]
        for first in relations:
            for second in relations:
                if first.left == second.left and first.right == second.right and _RELATION_NEGATIONS[first.relation] == second.relation:
                    return True
        # Detect incompatible exact numeric bounds on the same expression.
        grouped = {}
        for relation in relations:
            right = _numeric_value(relation.right)
            if right is None or isinstance(right, complex):
                continue
            grouped.setdefault(relation.left, []).append((relation.relation, float(right)))
        for bounds in grouped.values():
            equals = {value for relation, value in bounds if relation == "="}
            excluded = {value for relation, value in bounds if relation == "!="}
            if len(equals) > 1 or equals & excluded:
                return True
            lower_bounds = [(value, relation == ">") for relation, value in bounds if relation in {">", ">="}]
            upper_bounds = [(value, relation == "<") for relation, value in bounds if relation in {"<", "<="}]
            lower = None if not lower_bounds else (
                max(value for value, _ in lower_bounds),
                any(strict for value, strict in lower_bounds if value == max(item[0] for item in lower_bounds)),
            )
            upper = None if not upper_bounds else (
                min(value for value, _ in upper_bounds),
                any(strict for value, strict in upper_bounds if value == min(item[0] for item in upper_bounds)),
            )
            if lower and upper and (lower[0] > upper[0] or lower[0] == upper[0] and (lower[1] or upper[1])):
                return True
            if equals:
                value = next(iter(equals))
                if lower and (value < lower[0] or value == lower[0] and lower[1]):
                    return True
                if upper and (value > upper[0] or value == upper[0] and upper[1]):
                    return True
        return False

    def merge(self, *others):
        combined = list(self.conditions)
        for other in others:
            combined.extend(_coerce_assumption_set(other).conditions)
        return AssumptionSet(tuple(combined))

    def substitute(self, values, *, tolerance=1e-12):
        values = _condition_values(values)
        tolerance = _condition_tolerance(tolerance)
        return AssumptionSet(tuple(simplify_condition(condition.substitute(values), tolerance=tolerance) for condition in self.conditions))

    def evaluate(self, values=None, *, tolerance=1e-12):
        values = _condition_values(values)
        tolerance = _condition_tolerance(tolerance)
        if self.contradictory:
            return False
        results = tuple(condition.evaluate(values, tolerance=tolerance) for condition in self.conditions)
        return False if False in results else None if None in results else True

    def entails(self, condition):
        condition = simplify_condition(condition)
        if condition in self.conditions or isinstance(condition, TruthCondition) and condition.value:
            return True
        if self.contradictory:
            return True
        if isinstance(condition, CompoundCondition):
            if condition.operator == "and":
                return all(self.entails(item) for item in condition.conditions)
            if condition.operator == "or":
                return any(self.entails(item) for item in condition.conditions)
        if isinstance(condition, RelationCondition):
            relations = [item for item in self.conditions if isinstance(item, RelationCondition)]
            if any(_relation_implies(item, condition) for item in relations):
                return True
            # x >= c together with x != c entails x > c (and symmetrically).
            if condition.relation in {">", "<"}:
                weak = ">=" if condition.relation == ">" else "<="
                boundary = RelationCondition(condition.left, weak, condition.right)
                excluded = RelationCondition(condition.left, "!=", condition.right)
                if self.entails(boundary) and self.entails(excluded):
                    return True
            inferred = _relation_truth_from_signs(condition, _possible_signs(condition.left, self))
            if inferred is True and _defined_under_assumptions(condition.left, self, "real") is True:
                return True
        return False

    def refutes(self, condition):
        condition = simplify_condition(condition)
        if isinstance(condition, TruthCondition):
            return not condition.value
        negated = negate_condition(condition)
        return self.entails(negated)

    def infer_sign(self, expression):
        """Return the strongest supported sign class for ``expression``.

        The result is one of ``positive``, ``negative``, ``zero``,
        ``nonnegative``, ``nonpositive``, ``nonzero``, ``unknown``, or
        ``undefined``. Inference is exact and conservative; it never samples.
        """
        expression = simplify_symbolic(expression)
        return _SIGN_NAMES[_possible_signs(expression, self)]

    def is_defined(self, expression, domain="real"):
        """Return ``True``, ``False``, or ``None`` for definedness in a domain."""
        return _defined_under_assumptions(simplify_symbolic(expression), self, domain)

    def infer_domain(self, expression):
        """Infer ``real``, ``complex``, ``undefined``, or ``unknown``."""
        expression = simplify_symbolic(expression)
        real = _defined_under_assumptions(expression, self, "real")
        if real is True:
            return "real"
        complex_defined = _defined_under_assumptions(expression, self, "complex")
        if real is False and complex_defined is True:
            return "complex"
        if real is False and complex_defined is False:
            return "undefined"
        return "unknown"

    def to_dict(self):
        return {"type": "assumptions", "conditions": [condition.to_dict() for condition in self.conditions]}

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "assumptions":
            raise ValueError("Invalid assumption-set payload")
        return cls(tuple(_condition_from_dict(item) for item in data.get("conditions", ())))

    def __str__(self):
        return " and ".join(self.rendered) if self.conditions else "True"


def _coerce_assumption_set(values):
    if isinstance(values, AssumptionSet):
        return values
    return AssumptionSet(tuple(_coerce_condition(value) for value in values or ()))


def _normalize_user_assumptions(values):
    def user_condition(value):
        if isinstance(value, Condition):
            return value
        if isinstance(value, str):
            return parse_condition(value)
        raise TypeError("each assumption must be a Condition or atomic condition string")

    if values is None:
        return AssumptionSet()
    if isinstance(values, AssumptionSet):
        return values
    if isinstance(values, (Condition, str)):
        return AssumptionSet((user_condition(values),))
    if isinstance(values, Mapping):
        conditions = []
        for name, value in values.items():
            symbol = name if isinstance(name, Symbol) else Symbol(name)
            if isinstance(value, str) and value in {"real", "complex"}:
                conditions.append(DefinedCondition(symbol, value))
            else:
                conditions.append(RelationCondition(symbol, "=", _coerce(value)))
        return AssumptionSet(tuple(conditions))
    try:
        return AssumptionSet(tuple(user_condition(value) for value in values))
    except TypeError as error:
        raise TypeError("assumptions must be predicates, strings, a mapping, or an iterable") from error


def _condition_to_dict(condition):
    if isinstance(condition, TruthCondition):
        return {"type": "truth", "value": condition.value}
    if isinstance(condition, RelationCondition):
        return {"type": "relation", "left": condition.left.to_dict(), "relation": condition.relation, "right": condition.right.to_dict(), "display": condition.display}
    if isinstance(condition, DefinedCondition):
        return {"type": "defined", "expression": condition.expression.to_dict(), "domain": condition.domain}
    if isinstance(condition, BetweenCondition):
        return {"type": "between", "expression": condition.expression.to_dict(), "lower": condition.lower.to_dict(), "upper": condition.upper.to_dict(), "lower_closed": condition.lower_closed, "upper_closed": condition.upper_closed, "display": condition.display}
    if isinstance(condition, OpaqueCondition):
        return {"type": "opaque", "text": condition.text}
    if isinstance(condition, CompoundCondition):
        return {"type": "compound", "operator": condition.operator, "conditions": [item.to_dict() for item in condition.conditions]}
    raise TypeError(f"Unsupported condition {type(condition).__name__}")


def negate_condition(condition):
    """Return the structural logical negation of one predicate."""
    condition = simplify_condition(condition)
    if isinstance(condition, TruthCondition):
        return TruthCondition(not condition.value)
    if isinstance(condition, RelationCondition):
        return RelationCondition(condition.left, _RELATION_NEGATIONS[condition.relation], condition.right)
    if isinstance(condition, CompoundCondition) and condition.operator == "not":
        return condition.conditions[0]
    return CompoundCondition("not", (condition,))


def _condition_from_dict(data):
    kind = data.get("type")
    if kind == "truth":
        return TruthCondition(data["value"])
    if kind == "relation":
        return RelationCondition(symbolic_from_dict(data["left"]), data["relation"], symbolic_from_dict(data["right"]), data.get("display"))
    if kind == "defined":
        return DefinedCondition(symbolic_from_dict(data["expression"]), data["domain"])
    if kind == "between":
        return BetweenCondition(symbolic_from_dict(data["expression"]), symbolic_from_dict(data["lower"]), symbolic_from_dict(data["upper"]), data.get("lower_closed", True), data.get("upper_closed", True), data.get("display"))
    if kind == "opaque":
        return OpaqueCondition(data["text"])
    if kind == "compound":
        return CompoundCondition(data["operator"], tuple(_condition_from_dict(item) for item in data["conditions"]))
    raise ValueError(f"Unknown condition type {kind!r}")


def condition_from_dict(data):
    """Restore a structural condition from :meth:`Condition.to_dict` data."""
    return _condition_from_dict(data)


def _coerce(value):
    if isinstance(value, SymbolicExpression):
        return value
    if isinstance(value, (Rational, int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ExactNumber(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return ExactNumber(str(float(value)))
    raise TypeError(f"Cannot convert {type(value).__name__} to a symbolic expression")


def _add(*items):
    flat, number = [], Rational(0)
    for item in items:
        item = _coerce(item)
        values = item.terms if isinstance(item, Add) else (item,)
        for value in values:
            if isinstance(value, ExactNumber):
                number += value.value
            else:
                flat.append(value)
    grouped = {}
    for value in flat:
        coefficient, base = Rational(1), value
        if isinstance(value, Multiply) and value.factors and isinstance(value.factors[0], ExactNumber):
            coefficient = value.factors[0].value
            remainder = value.factors[1:]
            base = remainder[0] if len(remainder) == 1 else Multiply(remainder)
        grouped[base] = grouped.get(base, Rational(0)) + coefficient
    flat = []
    for base, coefficient in grouped.items():
        if coefficient:
            flat.append(base if coefficient == 1 else _mul(ExactNumber(coefficient), base))
        elif _requires_domain_guard(base):
            # Cancellation is algebraically zero only where the original term
            # is defined.  Keep a guarded zero so its domain survives parsing.
            flat.append(_mul(ZERO, base))
    flat.sort(key=str)
    if number:
        flat.insert(0, ExactNumber(number))
    if not flat:
        return ZERO
    return flat[0] if len(flat) == 1 else Add(tuple(flat))


def _neg(item):
    return _mul(NEG_ONE, item)


def _mul(*items):
    flat, number = [], Rational(1)
    for item in items:
        item = _coerce(item)
        values = item.factors if isinstance(item, Multiply) else (item,)
        for value in values:
            if isinstance(value, ExactNumber):
                number *= value.value
            else:
                flat.append(value)
    if not number:
        guarded = sorted((value for value in flat if _requires_domain_guard(value)), key=str)
        return Multiply((ZERO, *guarded)) if guarded else ZERO
    flat.sort(key=str)
    if number != 1 or not flat:
        flat.insert(0, ExactNumber(number))
    if len(flat) == 1:
        return flat[0]
    return Multiply(tuple(flat))


def _requires_domain_guard(expression):
    """Whether evaluating *expression* can be undefined on its natural domain."""
    if isinstance(expression, Power):
        if isinstance(expression.exponent, ExactNumber):
            exponent = expression.exponent.value
            if exponent < 0 or exponent.denominator % 2 == 0:
                return True
        else:
            return True
        return _requires_domain_guard(expression.base)
    if isinstance(expression, SymbolicFunction):
        if expression.name in {"sqrt", "ln", "log", "tan", "asin", "acos"}:
            return True
        return any(_requires_domain_guard(argument) for argument in expression.arguments)
    if isinstance(expression, Add):
        return any(_requires_domain_guard(term) for term in expression.terms)
    if isinstance(expression, Multiply):
        return any(_requires_domain_guard(factor) for factor in expression.factors)
    return False


def _pow(base, exponent):
    base, exponent = _coerce(base), _coerce(exponent)
    if isinstance(exponent, ExactNumber):
        if exponent.value == 0:
            # ``u**0`` is one only where ``u`` itself is defined.  Retain the
            # power node for partial expressions so domain collection can see
            # exclusions such as x != 0 in ``(1/x)**0``.
            return Power(base, exponent) if _requires_domain_guard(base) else ONE
        if exponent.value == 1:
            return base
        if isinstance(base, ExactNumber) and exponent.denominator == 1:
            try:
                return ExactNumber(base.value ** exponent.numerator)
            except ZeroDivisionError:
                raise ZeroDivisionError("zero cannot be raised to a negative power")
    return Power(base, exponent)


def _function(name, *arguments):
    name = "ln" if name == "log" and len(arguments) == 1 else name
    arguments = tuple(_coerce(value) for value in arguments)
    if name == "sqrt":
        value = arguments[0]
        if isinstance(value, ExactNumber) and value.value >= 0:
            n, d = math.isqrt(value.numerator), math.isqrt(value.denominator)
            if n * n == value.numerator and d * d == value.denominator:
                return ExactNumber(n, d)
    if name == "abs" and isinstance(arguments[0], ExactNumber):
        return ExactNumber(abs(arguments[0].value))
    if name in {"sin", "cos", "tan"}:
        coefficient = _pi_coefficient(arguments[0])
        if coefficient is not None:
            coefficient %= 2
            exact = _exact_trig_value(name, coefficient)
            if exact is not None:
                return exact
    return SymbolicFunction(name, arguments)


def _pi_coefficient(expression):
    if expression == PI:
        return Rational(1)
    if isinstance(expression, Multiply):
        number, pi_count = Rational(1), 0
        for factor in expression.factors:
            if isinstance(factor, ExactNumber): number *= factor.value
            elif factor == PI: pi_count += 1
            else: return None
        return number if pi_count == 1 else None
    return None


def _exact_trig_value(name, coefficient):
    half_sqrt2 = _mul(ExactNumber(1, 2), _function("sqrt", ExactNumber(2)))
    half_sqrt3 = _mul(ExactNumber(1, 2), _function("sqrt", ExactNumber(3)))
    sin_values = {
        Rational(0): ZERO, Rational(1, 6): ExactNumber(1, 2),
        Rational(1, 4): half_sqrt2, Rational(1, 3): half_sqrt3,
        Rational(1, 2): ONE, Rational(2, 3): half_sqrt3,
        Rational(3, 4): half_sqrt2, Rational(5, 6): ExactNumber(1, 2),
        Rational(1): ZERO, Rational(7, 6): ExactNumber(-1, 2),
        Rational(5, 4): _neg(half_sqrt2), Rational(4, 3): _neg(half_sqrt3),
        Rational(3, 2): NEG_ONE, Rational(5, 3): _neg(half_sqrt3),
        Rational(7, 4): _neg(half_sqrt2), Rational(11, 6): ExactNumber(-1, 2),
    }
    cos_values = {value % 2: sin_values.get((value + Rational(1, 2)) % 2) for value in sin_values}
    if name == "sin": return sin_values.get(coefficient)
    if name == "cos": return cos_values.get(coefficient)
    sine, cosine = sin_values.get(coefficient), cos_values.get(coefficient)
    if sine is None or cosine is None or cosine == ZERO:
        return None
    if sine == cosine:
        return ONE
    if sine == _neg(cosine):
        return NEG_ONE
    if isinstance(sine, ExactNumber) and isinstance(cosine, ExactNumber):
        return ExactNumber(sine.value / cosine.value)
    return _mul(sine, _pow(cosine, NEG_ONE))


def _variables(expression):
    if isinstance(expression, Symbol):
        return {expression.name}
    if isinstance(expression, (ExactNumber, SymbolicConstant, RootOf)):
        return set()
    if isinstance(expression, Add):
        return set().union(*(_variables(item) for item in expression.terms))
    if isinstance(expression, Multiply):
        return set().union(*(_variables(item) for item in expression.factors))
    if isinstance(expression, Power):
        return _variables(expression.base) | _variables(expression.exponent)
    return set().union(*(_variables(item) for item in expression.arguments))


def _substitute(expression, values):
    if isinstance(expression, Symbol) and expression.name in values:
        return _coerce(values[expression.name])
    if isinstance(expression, (ExactNumber, Symbol, SymbolicConstant, RootOf)):
        return expression
    if isinstance(expression, Add):
        return _add(*(_substitute(item, values) for item in expression.terms))
    if isinstance(expression, Multiply):
        return _mul(*(_substitute(item, values) for item in expression.factors))
    if isinstance(expression, Power):
        return _pow(_substitute(expression.base, values), _substitute(expression.exponent, values))
    return _function(expression.name, *(_substitute(item, values) for item in expression.arguments))


def _evaluate(expression, values):
    if isinstance(expression, ExactNumber):
        return float(expression.value)
    if isinstance(expression, Symbol):
        if expression.name not in values:
            raise ValueError(f"No value supplied for {expression.name!r}")
        return values[expression.name]
    if isinstance(expression, SymbolicConstant):
        return {"pi": math.pi, "e": math.e, "i": 1j}[expression.name]
    if isinstance(expression, Add):
        return sum(_evaluate(item, values) for item in expression.terms)
    if isinstance(expression, Multiply):
        return math.prod(_evaluate(item, values) for item in expression.factors)
    if isinstance(expression, Power):
        return _evaluate(expression.base, values) ** _evaluate(expression.exponent, values)
    if isinstance(expression, RootOf):
        if expression.interval is not None:
            lower = Rational.from_float(float(expression.interval[0]))
            upper = Rational.from_float(float(expression.interval[1]))
            coefficients = [value.value for value in expression.coefficients]

            def evaluate_exact(point):
                result = Rational(0)
                for coefficient in coefficients:
                    result = result * point + coefficient
                return result

            lower_value = evaluate_exact(lower)
            if lower_value == 0:
                return float(lower)
            upper_value = evaluate_exact(upper)
            if upper_value == 0:
                return float(upper)
            for _ in range(160):
                midpoint = (lower + upper) / 2
                middle_value = evaluate_exact(midpoint)
                if middle_value == 0:
                    return float(midpoint)
                if (lower_value < 0) != (middle_value < 0):
                    upper, upper_value = midpoint, middle_value
                else:
                    lower, lower_value = midpoint, middle_value
            return float((lower + upper) / 2)
        roots = np.roots([float(value.value) for value in expression.coefficients])
        roots = sorted(roots, key=lambda root: (root.real, root.imag))
        return complex(roots[expression.index])
    args = [_evaluate(item, values) for item in expression.arguments]
    functions = {
        "sqrt": cmath.sqrt, "abs": abs, "exp": cmath.exp, "ln": cmath.log,
        "sin": cmath.sin, "cos": cmath.cos, "tan": cmath.tan,
        "asin": cmath.asin, "acos": cmath.acos, "atan": cmath.atan,
    }
    if expression.name == "log" and len(args) == 2:
        return cmath.log(args[1], args[0])
    if expression.name not in functions:
        raise UnsupportedExpressionError(f"Cannot evaluate function {expression.name!r}")
    return functions[expression.name](*args)


# ---------------------------------------------------------------------------
# Parser


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_TOKEN_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|\*\*|[+\-*/^(),|]")
_FUNCTIONS = {"sqrt", "abs", "exp", "ln", "log", "sin", "cos", "tan", "asin", "acos", "atan"}


def _tokenize(text, *, max_size=10000):
    if not isinstance(text, str):
        raise TypeError("Symbolic expressions must be strings")
    if not text.strip():
        raise EquationParseError("Expression cannot be empty")
    if len(text) > max_size:
        raise UnsupportedExpressionError(f"Expression exceeds the {max_size}-character limit")
    result, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _TOKEN_RE.match(text, position)
        if not match:
            raise EquationParseError(f"Unexpected character {text[position]!r} at position {position}")
        value = match.group(0)
        kind = "number" if value[0].isdigit() or value[0] == "." else "identifier" if value[0].isalpha() or value[0] == "_" else "operator"
        result.append(_Token(kind, value, position))
        position = match.end()
    result.append(_Token("eof", "", len(text)))
    return result


class _ExpressionParser:
    def __init__(self, text, *, max_depth=100, max_nodes=10000):
        self.tokens = _tokenize(text, max_size=max_nodes * 8)
        self.index, self.depth, self.nodes = 0, 0, 0
        self.max_depth, self.max_nodes = max_depth, max_nodes

    @property
    def current(self):
        return self.tokens[self.index]

    def take(self, value=None):
        token = self.current
        if value is not None and token.value != value:
            raise EquationParseError(f"Expected {value!r} at position {token.position}")
        self.index += 1
        return token

    def node(self, value):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise UnsupportedExpressionError("Symbolic expression exceeds the node limit")
        return value

    def push_depth(self):
        self.depth += 1
        if self.depth > self.max_depth:
            raise UnsupportedExpressionError("Symbolic expression exceeds the nesting limit")

    def pop_depth(self):
        self.depth -= 1

    def parse(self):
        result = self.sum()
        if self.current.kind != "eof":
            raise EquationParseError(f"Unexpected token {self.current.value!r} at position {self.current.position}")
        return result

    def sum(self):
        result = self.product()
        while self.current.value in {"+", "-"}:
            operation = self.take().value
            other = self.product()
            result = self.node(_add(result, other if operation == "+" else _neg(other)))
        return result

    def product(self):
        result = self.unary()
        while True:
            if self.current.value in {"*", "/"}:
                operation = self.take().value
                other = self.unary()
                result = self.node(_mul(result, other) if operation == "*" else _mul(result, _pow(other, NEG_ONE)))
            elif self.current.kind == "identifier" or self.current.value == "(":
                result = self.node(_mul(result, self.unary()))
            else:
                return result

    def unary(self):
        if self.current.value in {"+", "-"}:
            operation = self.take().value
            self.push_depth()
            try:
                value = self.unary()
            finally:
                self.pop_depth()
            return value if operation == "+" else self.node(_neg(value))
        return self.power()

    def power(self):
        result = self.primary()
        if self.current.value in {"^", "**"}:
            self.take()
            self.push_depth()
            try:
                exponent = self.unary()
            finally:
                self.pop_depth()
            result = self.node(_pow(result, exponent))
        return result

    def primary(self):
        token = self.current
        if token.kind == "number":
            self.take()
            return self.node(ExactNumber(token.value))
        if token.value == "|":
            self.take()
            value = self.sum()
            self.take("|")
            return self.node(_function("abs", value))
        if token.value == "(":
            self.take()
            self.push_depth()
            try:
                value = self.sum()
                self.take(")")
            finally:
                self.pop_depth()
            return value
        if token.kind == "identifier":
            name = self.take().value
            if self.current.value == "(":
                if name not in _FUNCTIONS:
                    raise UnsupportedExpressionError(f"Unsupported symbolic function {name!r}")
                self.take("(")
                arguments = [self.sum()]
                while self.current.value == ",":
                    self.take()
                    arguments.append(self.sum())
                self.take(")")
                if name == "log" and len(arguments) not in {1, 2} or name != "log" and len(arguments) != 1:
                    raise EquationParseError(f"Invalid number of arguments for {name}")
                return self.node(_function(name, *arguments))
            return self.node({"pi": PI, "e": E, "i": I}.get(name, Symbol(name)))
        raise EquationParseError(f"Expected an expression at position {token.position}")


def parse_symbolic(expression: str) -> SymbolicExpression:
    """Parse an expression using the safe native symbolic grammar."""
    return _ExpressionParser(expression).parse()


def to_symbolic(value) -> SymbolicExpression:
    """Adapt a number, string, or legacy KiwiCalc expression to the native IR."""
    if isinstance(value, SymbolicExpression):
        return value
    if isinstance(value, (int, float, Rational, np.number)) and not isinstance(value, (bool, np.bool_)):
        return _coerce(value)
    from kiwicalc.core.interfaces import IExpression
    if isinstance(value, (str, IExpression)):
        return parse_symbolic(str(value))
    raise TypeError(f"Cannot adapt {type(value).__name__} to a symbolic expression")


def to_legacy_expression(value):
    """Adapt a supported native expression back to KiwiCalc's legacy tree."""
    from kiwicalc.expressions.factory import create
    value = to_symbolic(value)
    if value.variables and any(len(name) != 1 for name in value.variables):
        raise UnsupportedExpressionError("The legacy expression parser cannot preserve multi-character variables")
    if not _legacy_expression_supported(value):
        raise UnsupportedExpressionError("The legacy expression model cannot preserve this native expression")
    try:
        return create(str(value))
    except Exception as error:
        raise UnsupportedExpressionError("The legacy expression model cannot represent this native expression") from error


def _legacy_expression_supported(value):
    if isinstance(value, (ExactNumber, Symbol)):
        return True
    if isinstance(value, Add):
        return all(_legacy_expression_supported(term) for term in value.terms)
    if isinstance(value, Multiply):
        return all(_legacy_expression_supported(factor) for factor in value.factors)
    if isinstance(value, Power):
        return (
            _legacy_expression_supported(value.base)
            and isinstance(value.exponent, ExactNumber)
            and value.exponent.denominator == 1
            and value.exponent.numerator >= 0
        )
    return False


def simplify_symbolic(value):
    """Return KiwiCalc's deterministic, domain-safe basic canonical form.

    Contract:

    * exact numeric constants are reduced and locally folded;
    * nested sums/products are flattened, like additive terms are collected,
      and commutative operands have deterministic lexical ordering;
    * additive zero and multiplicative one are removed;
    * multiplication by zero and cancellation retain guarded subexpressions
      whenever their original domain can be smaller than the ambient domain;
    * trivial powers and the documented exact elementary-function values are
      reduced; and
    * the operation is immutable and idempotent.

    This is deliberately a *basic* canonical form.  It does not promise
    expansion, factorization, rational cancellation, or general algebraic and
    transcendental identity rewriting.
    """
    value = to_symbolic(value)
    if isinstance(value, Add): return _add(*(simplify_symbolic(item) for item in value.terms))
    if isinstance(value, Multiply): return _mul(*(simplify_symbolic(item) for item in value.factors))
    if isinstance(value, Power): return _pow(simplify_symbolic(value.base), simplify_symbolic(value.exponent))
    if isinstance(value, SymbolicFunction): return _function(value.name, *(simplify_symbolic(item) for item in value.arguments))
    return value


def is_canonical_symbolic(value):
    """Return whether ``value`` already satisfies the basic canonical contract."""
    value = to_symbolic(value)
    return value == simplify_symbolic(value)


def structurally_equal(first, second):
    return simplify_symbolic(first) == simplify_symbolic(second)


# ---------------------------------------------------------------------------
# Guarded rewrite rules


@dataclass(frozen=True)
class RewriteContext:
    """Read-only context supplied to rewrite transforms and guards."""

    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    domain: str = "real"

    def __post_init__(self):
        if not isinstance(self.assumptions, AssumptionSet):
            raise TypeError("rewrite context assumptions must be an AssumptionSet")
        if self.domain not in {"real", "complex"}:
            raise ValueError("rewrite domain must be 'real' or 'complex'")


@dataclass(frozen=True)
class RewriteRule:
    """One deterministic local rewrite with an optional semantic guard.

    ``transform(expression, context)`` returns a replacement expression or
    ``None`` when the rule does not match. Once matched,
    ``guard(before, after, context)`` returns ``True``, ``False``, or one or
    more conditions required for the replacement to be valid.
    """

    identifier: str
    transform: Callable = field(repr=False, compare=False)
    guard: Optional[Callable] = field(default=None, repr=False, compare=False)
    explanation: str = ""
    domain_preserving: bool = False

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", self.identifier):
            raise ValueError("rewrite rule identifiers must use lowercase kebab-case")
        if not callable(self.transform):
            raise TypeError("rewrite rule transform must be callable")
        if self.guard is not None and not callable(self.guard):
            raise TypeError("rewrite rule guard must be callable")
        if not isinstance(self.explanation, str):
            raise TypeError("rewrite rule explanation must be text")
        if not isinstance(self.domain_preserving, bool):
            raise TypeError("domain_preserving must be boolean")


@dataclass(frozen=True)
class RewriteApplication:
    """Auditable record of one accepted local rewrite."""

    rule: str
    path: Tuple[int, ...]
    before: SymbolicExpression
    after: SymbolicExpression
    required: AssumptionSet = field(default_factory=AssumptionSet)
    introduced: AssumptionSet = field(default_factory=AssumptionSet)
    explanation: str = ""

    def __post_init__(self):
        if not isinstance(self.rule, str) or not self.rule:
            raise ValueError("rewrite application rule must be nonempty")
        path = tuple(self.path)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in path):
            raise ValueError("rewrite paths must contain nonnegative integer indices")
        if not isinstance(self.before, SymbolicExpression) or not isinstance(self.after, SymbolicExpression):
            raise TypeError("rewrite application expressions must be symbolic")
        if not isinstance(self.required, AssumptionSet) or not isinstance(self.introduced, AssumptionSet):
            raise TypeError("rewrite application conditions must be assumption sets")
        if not isinstance(self.explanation, str):
            raise TypeError("rewrite application explanation must be text")
        object.__setattr__(self, "path", path)

    @property
    def conditions(self):
        """Rendered view of all conditions required by the rule."""
        return self.required.rendered

    def to_dict(self):
        return {
            "type": "rewrite_application", "rule": self.rule,
            "path": list(self.path), "before": self.before.to_dict(),
            "after": self.after.to_dict(), "required": self.required.to_dict(),
            "introduced": self.introduced.to_dict(),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "rewrite_application":
            raise ValueError("Invalid rewrite-application payload")
        return cls(
            data["rule"], tuple(data["path"]), symbolic_from_dict(data["before"]),
            symbolic_from_dict(data["after"]), AssumptionSet.from_dict(data["required"]),
            AssumptionSet.from_dict(data["introduced"]), data.get("explanation", ""),
        )


@dataclass(frozen=True)
class RewriteResult:
    """Expression, validity assumptions, and termination diagnostics."""

    expression: SymbolicExpression
    assumptions: AssumptionSet
    applications: Tuple[RewriteApplication, ...] = ()
    status: str = "fixed_point"
    message: str = ""

    def __post_init__(self):
        if not isinstance(self.expression, SymbolicExpression):
            raise TypeError("rewrite result expression must be symbolic")
        if not isinstance(self.assumptions, AssumptionSet):
            raise TypeError("rewrite result assumptions must be an AssumptionSet")
        applications = tuple(self.applications)
        if any(not isinstance(item, RewriteApplication) for item in applications):
            raise TypeError("rewrite result applications must be rewrite records")
        if self.status not in {"fixed_point", "step_limit", "node_limit", "cycle"}:
            raise ValueError("invalid rewrite result status")
        object.__setattr__(self, "applications", applications)

    @property
    def converged(self):
        return self.status == "fixed_point"

    @property
    def conditions(self):
        return self.assumptions.rendered

    def to_dict(self):
        return {
            "type": "rewrite_result", "expression": self.expression.to_dict(),
            "assumptions": self.assumptions.to_dict(),
            "applications": [item.to_dict() for item in self.applications],
            "status": self.status, "message": self.message,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "rewrite_result":
            raise ValueError("Invalid rewrite-result payload")
        return cls(
            symbolic_from_dict(data["expression"]),
            AssumptionSet.from_dict(data["assumptions"]),
            tuple(RewriteApplication.from_dict(item) for item in data.get("applications", ())),
            data.get("status", "fixed_point"), data.get("message", ""),
        )


def _rewrite_children(expression):
    if isinstance(expression, Add):
        return expression.terms
    if isinstance(expression, Multiply):
        return expression.factors
    if isinstance(expression, Power):
        return expression.base, expression.exponent
    if isinstance(expression, SymbolicFunction):
        return expression.arguments
    return ()


def _rewrite_paths(expression, strategy):
    paths = []

    def visit(value, path):
        if strategy == "top_down":
            paths.append(path)
        for index, child in enumerate(_rewrite_children(value)):
            visit(child, path + (index,))
        if strategy == "bottom_up":
            paths.append(path)

    visit(expression, ())
    return tuple(paths)


def _rewrite_node_at(expression, path):
    value = expression
    for index in path:
        children = _rewrite_children(value)
        if index >= len(children):
            raise ValueError("rewrite path does not identify an expression node")
        value = children[index]
    return value


def _rewrite_replace_at(expression, path, replacement):
    if not path:
        return simplify_symbolic(replacement)
    index, tail = path[0], path[1:]
    children = list(_rewrite_children(expression))
    if index >= len(children):
        raise ValueError("rewrite path does not identify an expression node")
    children[index] = _rewrite_replace_at(children[index], tail, replacement)
    if isinstance(expression, Add):
        rebuilt = Add(tuple(children))
    elif isinstance(expression, Multiply):
        rebuilt = Multiply(tuple(children))
    elif isinstance(expression, Power):
        rebuilt = Power(*children)
    elif isinstance(expression, SymbolicFunction):
        rebuilt = SymbolicFunction(expression.name, tuple(children))
    else:  # pragma: no cover - protected by the path validation above
        raise ValueError("rewrite path descends through a leaf expression")
    return simplify_symbolic(rebuilt)


def _rewrite_node_count(expression):
    return 1 + sum(_rewrite_node_count(child) for child in _rewrite_children(expression))


def _structurally_real(expression):
    """Conservatively identify expressions that are real on their real domain."""
    if isinstance(expression, (ExactNumber, Symbol)):
        return True
    if isinstance(expression, SymbolicConstant):
        return expression.name != "i"
    if isinstance(expression, RootOf):
        return expression.interval is not None
    if isinstance(expression, Add):
        return all(_structurally_real(term) for term in expression.terms)
    if isinstance(expression, Multiply):
        return all(_structurally_real(factor) for factor in expression.factors)
    if isinstance(expression, Power):
        return (
            _structurally_real(expression.base)
            and isinstance(expression.exponent, ExactNumber)
            and expression.exponent.denominator == 1
        )
    if isinstance(expression, SymbolicFunction):
        return expression.name == "abs" or all(_structurally_real(argument) for argument in expression.arguments)
    return False


_ALL_SIGNS = frozenset((-1, 0, 1))
_SIGN_NAMES = {
    frozenset(): "undefined",
    frozenset((-1,)): "negative",
    frozenset((0,)): "zero",
    frozenset((1,)): "positive",
    frozenset((-1, 0)): "nonpositive",
    frozenset((0, 1)): "nonnegative",
    frozenset((-1, 1)): "nonzero",
    _ALL_SIGNS: "unknown",
}
_SIGNS_FOR_RELATION = {
    "=": frozenset((0,)), "!=": frozenset((-1, 1)),
    ">": frozenset((1,)), ">=": frozenset((0, 1)),
    "<": frozenset((-1,)), "<=": frozenset((-1, 0)),
}
_REVERSED_RELATION = {"=": "=", "!=": "!=", ">": "<", ">=": "<=", "<": ">", "<=": ">="}


def _normalized_relation_for_expression(condition, expression):
    if condition.left == expression:
        return condition
    if condition.right == expression:
        return RelationCondition(expression, _REVERSED_RELATION[condition.relation], condition.left)
    return None


def _direct_sign_constraints(expression, assumptions):
    possible = _ALL_SIGNS
    targets = tuple(
        (RelationCondition(expression, relation, ZERO), signs)
        for relation, signs in _SIGNS_FOR_RELATION.items()
    )
    for condition in assumptions.conditions:
        if isinstance(condition, RelationCondition):
            normalized = _normalized_relation_for_expression(condition, expression)
            if normalized is None:
                continue
            for target, signs in targets:
                if _relation_implies(normalized, target):
                    possible &= signs
        elif isinstance(condition, BetweenCondition) and condition.expression == expression:
            lower_relation = ">=" if condition.lower_closed else ">"
            upper_relation = "<=" if condition.upper_closed else "<"
            for normalized in (
                RelationCondition(expression, lower_relation, condition.lower),
                RelationCondition(expression, upper_relation, condition.upper),
            ):
                for target, signs in targets:
                    if _relation_implies(normalized, target):
                        possible &= signs
    return possible


def _possible_signs(expression, assumptions, _seen=None):
    expression = simplify_symbolic(expression)
    _seen = set() if _seen is None else _seen
    if expression in _seen or assumptions.contradictory:
        return frozenset() if assumptions.contradictory else _ALL_SIGNS
    seen = _seen | {expression}
    try:
        substituted = _substitute(expression, assumptions.substitutions)
    except (ValueError, ZeroDivisionError, OverflowError):
        return frozenset()
    if substituted != expression:
        return _possible_signs(substituted, assumptions, seen)

    if isinstance(expression, ExactNumber):
        structural = frozenset((0 if expression.value == 0 else 1 if expression.value > 0 else -1,))
    elif isinstance(expression, SymbolicConstant):
        structural = frozenset((1,)) if expression.name in {"pi", "e"} else frozenset()
    elif isinstance(expression, RootOf):
        if expression.interval is None:
            structural = _ALL_SIGNS
        elif expression.interval[0] >= 0:
            structural = frozenset((1,)) if expression.interval[0] > 0 else frozenset((0, 1))
        elif expression.interval[1] <= 0:
            structural = frozenset((-1,)) if expression.interval[1] < 0 else frozenset((-1, 0))
        else:
            structural = _ALL_SIGNS
    elif isinstance(expression, Symbol):
        structural = _ALL_SIGNS
    elif isinstance(expression, Add):
        signs = tuple(_possible_signs(term, assumptions, seen) for term in expression.terms)
        if any(not values for values in signs):
            structural = frozenset()
        elif all(values == frozenset((0,)) for values in signs):
            structural = frozenset((0,))
        elif all(values <= frozenset((0, 1)) for values in signs):
            structural = frozenset((1,)) if any(values == frozenset((1,)) for values in signs) else frozenset((0, 1))
        elif all(values <= frozenset((-1, 0)) for values in signs):
            structural = frozenset((-1,)) if any(values == frozenset((-1,)) for values in signs) else frozenset((-1, 0))
        else:
            structural = _ALL_SIGNS
    elif isinstance(expression, Multiply):
        structural = frozenset((1,))
        for factor in expression.factors:
            factor_signs = _possible_signs(factor, assumptions, seen)
            structural = frozenset(first * second for first in structural for second in factor_signs)
            if not structural:
                break
    elif isinstance(expression, Power) and isinstance(expression.exponent, ExactNumber):
        base = _possible_signs(expression.base, assumptions, seen)
        exponent = expression.exponent.value
        if exponent == 0:
            structural = frozenset((1,))
        else:
            if exponent < 0:
                base -= frozenset((0,))
            if exponent.denominator % 2 == 0:
                base &= frozenset((0, 1))
            if exponent.numerator % 2 == 0:
                structural = frozenset((0 if sign == 0 else 1 for sign in base))
            else:
                structural = base
    elif isinstance(expression, Power):
        structural = _ALL_SIGNS
    elif isinstance(expression, SymbolicFunction):
        argument = _possible_signs(expression.arguments[-1], assumptions, seen)
        if expression.name in {"abs", "sqrt"}:
            allowed = argument if expression.name == "abs" else argument & frozenset((0, 1))
            structural = frozenset((0 if sign == 0 else 1 for sign in allowed))
        elif expression.name == "exp":
            structural = frozenset((1,))
        elif expression.name == "ln":
            if assumptions.entails(RelationCondition(expression.arguments[0], ">", ONE)):
                structural = frozenset((1,))
            elif assumptions.entails(RelationCondition(expression.arguments[0], "=", ONE)):
                structural = frozenset((0,))
            elif (
                assumptions.entails(RelationCondition(expression.arguments[0], ">", ZERO))
                and assumptions.entails(RelationCondition(expression.arguments[0], "<", ONE))
            ):
                structural = frozenset((-1,))
            else:
                structural = _ALL_SIGNS
        else:
            structural = _ALL_SIGNS
    else:
        structural = _ALL_SIGNS
    return structural & _direct_sign_constraints(expression, assumptions)


def _relation_truth_from_signs(condition, signs):
    right = _numeric_value(condition.right)
    if right is None or isinstance(right, complex) or right != 0 or not signs:
        return None
    allowed = _SIGNS_FOR_RELATION[condition.relation]
    return True if signs <= allowed else False if signs.isdisjoint(allowed) else None


def _defined_under_assumptions(expression, assumptions, domain):
    if domain not in {"real", "complex"}:
        raise ValueError("domain must be 'real' or 'complex'")
    if assumptions.contradictory:
        return False
    try:
        substituted = _substitute(expression, assumptions.substitutions)
    except (ValueError, ZeroDivisionError, OverflowError):
        return False
    if substituted != expression:
        return _defined_under_assumptions(substituted, assumptions, domain)
    predicate = DefinedCondition(expression, domain)
    if predicate in assumptions.conditions:
        return True
    if assumptions.refutes(predicate):
        return False
    if domain == "real" and expression == I:
        return False
    variable = sorted(expression.variables)[0] if expression.variables else "x"
    undecided = False
    for condition in _domain_conditions(expression, domain, variable):
        condition = simplify_condition(condition, assumptions.substitutions)
        if isinstance(condition, TruthCondition):
            if not condition.value:
                return False
        elif assumptions.refutes(condition):
            return False
        elif not assumptions.entails(condition):
            undecided = True
    if undecided:
        return None
    if domain == "real" and not _structurally_real(expression):
        return None
    return True


def _rewrite_guard_conditions(outcome):
    if outcome is True:
        return True, AssumptionSet()
    if outcome is False:
        return False, AssumptionSet()
    if isinstance(outcome, AssumptionSet):
        return True, outcome
    if isinstance(outcome, (Condition, str)):
        return True, _normalize_user_assumptions(outcome)
    try:
        return True, _normalize_user_assumptions(tuple(outcome))
    except TypeError as error:
        raise TypeError("rewrite guards must return a boolean, condition, or iterable of conditions") from error


def _cancel_reciprocal_match(expression):
    if not isinstance(expression, Multiply):
        return None
    for reciprocal_index, reciprocal in enumerate(expression.factors):
        if not isinstance(reciprocal, Power) or reciprocal.exponent != NEG_ONE:
            continue
        for factor_index, factor in enumerate(expression.factors):
            if factor_index != reciprocal_index and factor == reciprocal.base:
                remaining = tuple(
                    value for index, value in enumerate(expression.factors)
                    if index not in {reciprocal_index, factor_index}
                )
                return reciprocal.base, _mul(*remaining)
    return None


def _cancel_reciprocal_transform(expression, context):
    matched = _cancel_reciprocal_match(expression)
    return None if matched is None else matched[1]


def _cancel_reciprocal_guard(expression, replacement, context):
    return RelationCondition(_cancel_reciprocal_match(expression)[0], "!=", ZERO)


def _exp_log_transform(expression, context):
    if (
        isinstance(expression, SymbolicFunction) and expression.name == "exp"
        and isinstance(expression.arguments[0], SymbolicFunction)
        and expression.arguments[0].name == "ln"
    ):
        return expression.arguments[0].arguments[0]
    return None


def _exp_log_guard(expression, replacement, context):
    relation = ">" if context.domain == "real" else "!="
    return RelationCondition(replacement, relation, ZERO)


def _sqrt_square_transform(expression, context):
    if context.domain != "real" or not (
        isinstance(expression, SymbolicFunction) and expression.name == "sqrt"
        and isinstance(expression.arguments[0], Power)
        and expression.arguments[0].exponent == ExactNumber(2)
        and _structurally_real(expression.arguments[0].base)
    ):
        return None
    return _function("abs", expression.arguments[0].base)


def _normalize_rational_rewrite_transform(expression, context):
    try:
        return _rational_normalization_data(expression, None, 100)[0]
    except (AmbiguousVariableError, UnsupportedExpressionError, ValueError, ZeroDivisionError):
        return None


def _normalize_rational_rewrite_guard(expression, replacement, context):
    try:
        _, _, condition = _rational_normalization_data(expression, None, 100)
    except (AmbiguousVariableError, UnsupportedExpressionError, ValueError, ZeroDivisionError):
        return False
    return condition or True


_STANDARD_REWRITE_RULES = {
    "cancel-reciprocal": RewriteRule(
        "cancel-reciprocal", _cancel_reciprocal_transform, _cancel_reciprocal_guard,
        "Cancel one factor against its reciprocal while retaining the nonzero condition.",
    ),
    "exp-log-inverse": RewriteRule(
        "exp-log-inverse", _exp_log_transform, _exp_log_guard,
        "Apply the exponential/logarithm inverse on the logarithm's domain.",
    ),
    "sqrt-square": RewriteRule(
        "sqrt-square", _sqrt_square_transform,
        explanation="Rewrite the real principal square root of a square as an absolute value.",
    ),
    "normalize-rational": RewriteRule(
        "normalize-rational", _normalize_rational_rewrite_transform,
        _normalize_rational_rewrite_guard,
        "Expand a univariate rational expression and cancel its exact polynomial GCD.",
        True,
    ),
}

_DEFAULT_REWRITE_RULES = (
    _STANDARD_REWRITE_RULES["cancel-reciprocal"],
    _STANDARD_REWRITE_RULES["exp-log-inverse"],
    _STANDARD_REWRITE_RULES["sqrt-square"],
)


def available_rewrite_rules():
    """Return the stable identifiers of KiwiCalc's built-in rewrite rules."""
    return tuple(_STANDARD_REWRITE_RULES)


def _coerce_rewrite_rules(rules):
    if rules is None:
        return _DEFAULT_REWRITE_RULES
    if isinstance(rules, (str, RewriteRule)):
        rules = (rules,)
    try:
        rules = tuple(rules)
    except TypeError as error:
        raise TypeError("rules must be rewrite rules or built-in rule identifiers") from error
    result = []
    for rule in rules:
        if isinstance(rule, str):
            try:
                rule = _STANDARD_REWRITE_RULES[rule]
            except KeyError as error:
                raise ValueError(f"Unknown rewrite rule {rule!r}") from error
        if not isinstance(rule, RewriteRule):
            raise TypeError("rules must contain RewriteRule objects or built-in identifiers")
        result.append(rule)
    identifiers = [rule.identifier for rule in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("rewrite rule identifiers must be unique within a run")
    return tuple(result)


def rewrite_symbolic(value, rules=None, assumptions=None, *, domain="real",
                     strategy="bottom_up", introduce_conditions=False,
                     max_steps=100, max_nodes=10000):
    """Apply guarded rules to a deterministic fixed point.

    Unknown guards are skipped by default. With ``introduce_conditions=True``
    they are accepted and recorded in the returned assumption set. Conditions
    needed merely to preserve the original expression's domain are collected
    automatically and therefore never disappear during a rewrite.
    """
    if domain not in {"real", "complex"}:
        raise ValueError("domain must be 'real' or 'complex'")
    if strategy not in {"bottom_up", "top_down"}:
        raise ValueError("strategy must be 'bottom_up' or 'top_down'")
    if not isinstance(introduce_conditions, bool):
        raise TypeError("introduce_conditions must be boolean")
    for name, limit in (("max_steps", max_steps), ("max_nodes", max_nodes)):
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    rules = _coerce_rewrite_rules(rules)
    expression = simplify_symbolic(value)
    if _rewrite_node_count(expression) > max_nodes:
        raise UnsupportedExpressionError(f"Expression exceeds the {max_nodes}-node rewrite limit")
    variable = sorted(expression.variables)[0] if expression.variables else "x"
    current_assumptions = _normalize_user_assumptions(assumptions).merge(
        _domain_conditions(expression, domain, variable)
    )
    applications = []
    seen = {(expression, current_assumptions)}

    for _ in range(int(max_steps)):
        applied = False
        context = RewriteContext(current_assumptions, domain)
        for path in _rewrite_paths(expression, strategy):
            before = _rewrite_node_at(expression, path)
            for rule in rules:
                replacement = rule.transform(before, context)
                if replacement is None:
                    continue
                replacement = simplify_symbolic(replacement)
                if replacement == before:
                    continue
                candidate = _rewrite_replace_at(expression, path, replacement)
                if candidate == expression:
                    continue
                if _rewrite_node_count(candidate) > max_nodes:
                    return RewriteResult(expression, current_assumptions, tuple(applications), "node_limit", f"A rewrite exceeded the {max_nodes}-node limit.")
                eligible, required = _rewrite_guard_conditions(
                    True if rule.guard is None else rule.guard(before, replacement, context)
                )
                candidate_variable = sorted(candidate.variables)[0] if candidate.variables else variable
                if not rule.domain_preserving:
                    required = required.merge(_domain_conditions(candidate, domain, candidate_variable))
                if not eligible or any(current_assumptions.refutes(condition) for condition in required):
                    continue
                introduced = AssumptionSet(tuple(
                    condition for condition in required
                    if not current_assumptions.entails(condition)
                ))
                if introduced and not introduce_conditions:
                    continue
                next_assumptions = current_assumptions.merge(introduced)
                if next_assumptions.contradictory:
                    continue
                state = candidate, next_assumptions
                if state in seen:
                    return RewriteResult(expression, current_assumptions, tuple(applications), "cycle", f"Rule {rule.identifier!r} would revisit an earlier rewrite state.")
                applications.append(RewriteApplication(
                    rule.identifier, path, before, replacement, required,
                    introduced, rule.explanation,
                ))
                expression, current_assumptions = candidate, next_assumptions
                seen.add(state)
                applied = True
                break
            if applied:
                break
        if not applied:
            return RewriteResult(expression, current_assumptions, tuple(applications), "fixed_point", "No applicable guarded rewrite remains.")
    return RewriteResult(expression, current_assumptions, tuple(applications), "step_limit", f"The {max_steps}-step rewrite limit was reached.")


def differentiate_symbolic(value, variable="x"):
    """Differentiate the supported native expression tree exactly."""
    value = to_symbolic(value)
    variable = variable.name if hasattr(variable, "name") else variable
    if isinstance(value, (ExactNumber, SymbolicConstant, RootOf)):
        return ZERO
    if isinstance(value, Symbol):
        return ONE if value.name == variable else ZERO
    if isinstance(value, Add):
        return _add(*(differentiate_symbolic(item, variable) for item in value.terms))
    if isinstance(value, Multiply):
        terms = []
        for index, factor in enumerate(value.factors):
            derivative = differentiate_symbolic(factor, variable)
            if derivative != ZERO:
                terms.append(_mul(derivative, *(item for position, item in enumerate(value.factors) if position != index)))
        return _add(*terms)
    if isinstance(value, Power) and isinstance(value.exponent, ExactNumber):
        return _mul(value.exponent, _pow(value.base, _add(value.exponent, NEG_ONE)), differentiate_symbolic(value.base, variable))
    if isinstance(value, SymbolicFunction) and len(value.arguments) == 1:
        argument = value.arguments[0]
        derivative = differentiate_symbolic(argument, variable)
        outer = {
            "sin": _function("cos", argument),
            "cos": _neg(_function("sin", argument)),
            "tan": _pow(_function("cos", argument), ExactNumber(-2)),
            "exp": _function("exp", argument),
            "ln": _pow(argument, NEG_ONE),
            "sqrt": _mul(ExactNumber(1, 2), _pow(argument, ExactNumber(-1, 2))),
        }.get(value.name)
        if outer is not None:
            return _mul(outer, derivative)
    raise UnsupportedExpressionError(f"Cannot symbolically differentiate {value}")


def _split_symbolic_equation(equation):
    if equation.count("=") != 1:
        raise EquationParseError("An equation must contain exactly one '='")
    left, right = equation.split("=", 1)
    if not left.strip() or not right.strip():
        raise EquationParseError("Both sides of an equation must be non-empty")
    return parse_symbolic(left), parse_symbolic(right)


# ---------------------------------------------------------------------------
# Structured solutions


class SolutionSet:
    def to_dict(self):
        return _solution_set_to_dict(self)


@dataclass(frozen=True)
class EmptySolutionSet(SolutionSet):
    def __str__(self): return "EmptySet"


@dataclass(frozen=True)
class UniversalSolutionSet(SolutionSet):
    domain: str = "real"
    def __post_init__(self):
        if self.domain not in {"real", "complex"}:
            raise ValueError("Universal solution domain must be 'real' or 'complex'")
    def __str__(self): return "Reals" if self.domain == "real" else "Complexes"


@dataclass(frozen=True)
class FiniteSolutionSet(SolutionSet):
    values: Tuple[Any, ...]
    multiplicities: Tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))
        if any(not _valid_solution_value(value) for value in self.values):
            raise TypeError("finite solutions must be immutable numeric or symbolic values")
        multiplicities = tuple(self.multiplicities) or (1,) * len(self.values)
        if len(multiplicities) != len(self.values) or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in multiplicities):
            raise ValueError("multiplicities must contain one positive integer per solution")
        object.__setattr__(self, "multiplicities", multiplicities)

    def __str__(self): return "{" + ", ".join(map(str, self.values)) + "}"


def _valid_solution_value(value):
    if isinstance(value, SymbolicExpression):
        return True
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, complex, np.number, Rational)):
        return False
    try:
        numeric = complex(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric.real) and math.isfinite(numeric.imag)


@dataclass(frozen=True)
class IntervalSolutionSet(SolutionSet):
    lower: Any
    upper: Any
    lower_closed: bool = True
    upper_closed: bool = True

    def __post_init__(self):
        if not isinstance(self.lower_closed, bool) or not isinstance(self.upper_closed, bool):
            raise TypeError("Interval closure flags must be booleans")
        lower, upper = _numeric_value(self.lower), _numeric_value(self.upper)
        if lower is not None and upper is not None:
            if isinstance(lower, complex) or isinstance(upper, complex) or lower > upper:
                raise ValueError("Interval bounds must be ordered real values")

    def __str__(self):
        lower = "-inf" if self.lower is None else str(self.lower)
        upper = "inf" if self.upper is None else str(self.upper)
        return (
            ("[" if self.lower_closed and self.lower is not None else "(")
            + f"{lower}, {upper}"
            + ("]" if self.upper_closed and self.upper is not None else ")")
        )


@dataclass(frozen=True)
class ParametricSolutionSet(SolutionSet):
    variable: str
    expression: SymbolicExpression
    parameter: str = "n"
    parameter_domain: str = "integers"

    def __post_init__(self):
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("Parametric solution variable must be a nonempty string")
        if not isinstance(self.parameter, str) or not self.parameter or self.parameter == self.variable:
            raise ValueError("Parametric solution parameter must be a distinct nonempty string")
        if not isinstance(self.expression, SymbolicExpression):
            raise TypeError("Parametric solution expression must be symbolic")
        if self.parameter_domain != "integers":
            raise ValueError("Only integer-parameterized families are currently supported")

    def __str__(self): return f"{self.variable} = {self.expression}, {self.parameter} in Z"


@dataclass(frozen=True)
class UnionSolutionSet(SolutionSet):
    sets: Tuple[SolutionSet, ...]

    def __post_init__(self):
        sets = tuple(self.sets)
        if not sets or any(not isinstance(value, SolutionSet) for value in sets):
            raise TypeError("A union must contain one or more solution sets")
        object.__setattr__(self, "sets", sets)
    def __str__(self): return " union ".join(map(str, self.sets))


@dataclass(frozen=True)
class ConditionalSolutionSet(SolutionSet):
    solution_set: SolutionSet
    conditions: Tuple[str, ...]
    _assumptions: AssumptionSet = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.solution_set, SolutionSet):
            raise TypeError("Conditional solution must wrap a solution set")
        assumptions = _coerce_assumption_set(self.conditions)
        if not assumptions:
            raise ValueError("Conditional solution conditions must be nonempty")
        object.__setattr__(self, "conditions", assumptions.rendered)
        object.__setattr__(self, "_assumptions", assumptions)

    @property
    def assumptions(self):
        return self._assumptions


@dataclass(frozen=True)
class EquationState:
    """Immutable structural equation used by derivation records."""

    left: SymbolicExpression
    right: SymbolicExpression

    def __post_init__(self):
        if not isinstance(self.left, SymbolicExpression) or not isinstance(self.right, SymbolicExpression):
            raise TypeError("EquationState sides must be symbolic expressions")

    @property
    def residual(self):
        return _add(self.left, _neg(self.right))

    def satisfied_by(self, values, tolerance=1e-10):
        residual = complex(_evaluate(self.residual, values))
        return math.isfinite(residual.real) and math.isfinite(residual.imag) and abs(residual) <= tolerance

    def __str__(self):
        return f"{self.left} = {self.right}"


def _step_value(value):
    if isinstance(value, (EquationState, SolutionSet)):
        return value
    if not isinstance(value, str):
        raise TypeError("Solution step states must be text, equations, or solution sets")
    if value.count("=") == 1:
        try:
            left, right = _split_symbolic_equation(value)
            return EquationState(left, right)
        except (EquationParseError, UnsupportedExpressionError, TypeError, ValueError):
            pass
    return value


@dataclass(frozen=True)
class SolutionStep:
    rule: str
    before: Any
    after: Any
    explanation: str
    conditions: Tuple[str, ...] = ()
    _assumptions: AssumptionSet = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.rule, str) or not isinstance(self.explanation, str):
            raise TypeError("Solution step rule and explanation must be strings")
        assumptions = _coerce_assumption_set(self.conditions)
        object.__setattr__(self, "before", _step_value(self.before))
        object.__setattr__(self, "after", _step_value(self.after))
        object.__setattr__(self, "conditions", assumptions.rendered)
        object.__setattr__(self, "_assumptions", assumptions)

    @property
    def assumptions(self):
        return self._assumptions

    def equivalent_at(self, values, tolerance=1e-10):
        """Replay an equation-to-equation step at one admissible assignment."""
        if not isinstance(self.before, EquationState) or not isinstance(self.after, EquationState):
            raise TypeError("This step does not contain two equation states")
        return self.before.satisfied_by(values, tolerance) == self.after.satisfied_by(values, tolerance)


@dataclass(frozen=True)
class EquationSolution:
    variable: str
    solution_set: SolutionSet
    status: str
    method: str
    exact: bool
    complete: bool
    conditions: Tuple[str, ...] = ()
    residuals: Tuple[float, ...] = ()
    steps: Tuple[SolutionStep, ...] = ()
    message: str = ""
    evaluations: int = 0
    _assumptions: AssumptionSet = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("Solution variable must be a nonempty string")
        if not isinstance(self.solution_set, SolutionSet):
            raise TypeError("solution_set must be a SolutionSet")
        if self.status not in {"solved", "unresolved"}:
            raise ValueError("Equation solution status must be 'solved' or 'unresolved'")
        if self.method not in {"symbolic", "numeric", "hybrid"}:
            raise ValueError("Equation solution method is invalid")
        if any(not isinstance(value, bool) for value in (self.exact, self.complete)):
            raise TypeError("exact and complete must be booleans")
        if isinstance(self.evaluations, bool) or not isinstance(self.evaluations, int) or self.evaluations < 0:
            raise ValueError("evaluations must be a nonnegative integer")
        assumptions = _coerce_assumption_set(self.conditions)
        conditions, residuals, steps = assumptions.rendered, tuple(self.residuals), tuple(self.steps)
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or value < 0) for value in residuals):
            raise ValueError("residuals must contain nonnegative finite values or None")
        if any(not isinstance(value, SolutionStep) for value in steps):
            raise TypeError("steps must contain SolutionStep values")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "_assumptions", assumptions)

    @property
    def assumptions(self):
        """Structural predicates qualifying this solution.

        ``conditions`` remains the backward-compatible tuple of display
        strings; this property is the machine-readable reasoning model.
        """
        return self._assumptions

    @property
    def solutions(self):
        return self.solution_set.values if isinstance(self.solution_set, FiniteSolutionSet) else self.solution_set

    @property
    def converged(self):
        return self.status == "solved"

    def __str__(self):
        if self.status != "solved":
            return f"{self.variable}: unresolved ({self.message})"
        return f"{self.variable} in {self.solution_set}"

    def formatted_steps(self):
        return tuple(f"{index}. {step.explanation}  {step.before} -> {step.after}" for index, step in enumerate(self.steps, 1))

    def to_dict(self):
        return {
            "variable": self.variable, "solution_set": self.solution_set.to_dict(),
            "status": self.status, "method": self.method, "exact": self.exact,
            "complete": self.complete, "conditions": list(self.conditions),
            "assumptions": self.assumptions.to_dict(),
            "residuals": list(self.residuals),
            "steps": [_step_to_dict(step) for step in self.steps],
            "message": self.message, "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["solution_set"] = _solution_set_from_dict(data["solution_set"])
        assumptions = data.pop("assumptions", None)
        data["conditions"] = (
            AssumptionSet.from_dict(assumptions).conditions
            if assumptions is not None else tuple(data.get("conditions", ()))
        )
        data["residuals"] = tuple(
            None if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isnan(float(value)) else value
            for value in data.get("residuals", ())
        )
        data["steps"] = tuple(_step_from_dict(item) for item in data.get("steps", ()))
        return cls(**data)


EMPTY = EmptySolutionSet()


def _step_to_dict(step):
    return {
        "rule": step.rule,
        "before": _step_value_to_dict(step.before),
        "after": _step_value_to_dict(step.after),
        "explanation": step.explanation,
        "conditions": list(step.conditions),
        "assumptions": step.assumptions.to_dict(),
    }


def _step_value_to_dict(value):
    if isinstance(value, EquationState):
        return {"kind": "equation", "left": value.left.to_dict(), "right": value.right.to_dict()}
    if isinstance(value, SolutionSet):
        return {"kind": "solution_set", "value": value.to_dict()}
    return {"kind": "text", "value": value}


def _step_value_from_dict(value):
    if not isinstance(value, Mapping) or "kind" not in value:
        return value  # Backward compatibility with the initial text-only format.
    if value["kind"] == "equation":
        return EquationState(symbolic_from_dict(value["left"]), symbolic_from_dict(value["right"]))
    if value["kind"] == "solution_set":
        return _solution_set_from_dict(value["value"])
    if value["kind"] == "text":
        return value["value"]
    raise ValueError(f"Unknown solution-step state kind {value['kind']!r}")


def _step_from_dict(item):
    result = dict(item)
    result["before"] = _step_value_from_dict(result["before"])
    result["after"] = _step_value_from_dict(result["after"])
    assumptions = result.pop("assumptions", None)
    result["conditions"] = AssumptionSet.from_dict(assumptions).conditions if assumptions is not None else tuple(result.get("conditions", ()))
    return SolutionStep(**result)


def _expression_to_dict(value):
    if isinstance(value, ExactNumber): return {"type": "number", "numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Symbol): return {"type": "symbol", "name": value.name}
    if isinstance(value, SymbolicConstant): return {"type": "constant", "name": value.name}
    if isinstance(value, Add): return {"type": "add", "terms": [_expression_to_dict(item) for item in value.terms]}
    if isinstance(value, Multiply): return {"type": "multiply", "factors": [_expression_to_dict(item) for item in value.factors]}
    if isinstance(value, Power): return {"type": "power", "base": _expression_to_dict(value.base), "exponent": _expression_to_dict(value.exponent)}
    if isinstance(value, SymbolicFunction): return {"type": "function", "name": value.name, "arguments": [_expression_to_dict(item) for item in value.arguments]}
    if isinstance(value, RootOf): return {"type": "root_of", "coefficients": [_expression_to_dict(item) for item in value.coefficients], "index": value.index, "interval": value.interval}
    raise TypeError(f"Unsupported symbolic expression {type(value).__name__}")


def symbolic_from_dict(data):
    kind = data.get("type")
    if kind == "number": return ExactNumber(data["numerator"], data["denominator"])
    if kind == "symbol": return Symbol(data["name"])
    if kind == "constant": return SymbolicConstant(data["name"])
    if kind == "add": return _add(*(symbolic_from_dict(item) for item in data["terms"]))
    if kind == "multiply": return _mul(*(symbolic_from_dict(item) for item in data["factors"]))
    if kind == "power": return _pow(symbolic_from_dict(data["base"]), symbolic_from_dict(data["exponent"]))
    if kind == "function": return _function(data["name"], *(symbolic_from_dict(item) for item in data["arguments"]))
    if kind == "root_of": return RootOf(tuple(symbolic_from_dict(item) for item in data["coefficients"]), data["index"], tuple(data["interval"]) if data.get("interval") else None)
    raise ValueError(f"Unknown symbolic expression type {kind!r}")


def _solution_set_to_dict(value):
    if isinstance(value, EmptySolutionSet): return {"type": "empty"}
    if isinstance(value, UniversalSolutionSet): return {"type": "universal", "domain": value.domain}
    if isinstance(value, FiniteSolutionSet): return {"type": "finite", "values": [_encode_value(item) for item in value.values], "multiplicities": list(value.multiplicities)}
    if isinstance(value, IntervalSolutionSet): return {"type": "interval", "lower": _encode_value(value.lower), "upper": _encode_value(value.upper), "lower_closed": value.lower_closed, "upper_closed": value.upper_closed}
    if isinstance(value, ParametricSolutionSet): return {"type": "parametric", "variable": value.variable, "expression": value.expression.to_dict(), "parameter": value.parameter, "parameter_domain": value.parameter_domain}
    if isinstance(value, UnionSolutionSet): return {"type": "union", "sets": [item.to_dict() for item in value.sets]}
    if isinstance(value, ConditionalSolutionSet): return {"type": "conditional", "solution_set": value.solution_set.to_dict(), "conditions": list(value.conditions), "assumptions": value.assumptions.to_dict()}
    raise TypeError(f"Unsupported solution set {type(value).__name__}")


def _encode_value(value):
    if isinstance(value, SymbolicExpression): return {"kind": "symbolic", "value": value.to_dict()}
    if isinstance(value, complex): return {"kind": "complex", "real": value.real, "imag": value.imag}
    return {"kind": "scalar", "value": value}


def _decode_value(data):
    if data["kind"] == "symbolic": return symbolic_from_dict(data["value"])
    if data["kind"] == "complex": return complex(data["real"], data["imag"])
    return data["value"]


def _solution_set_from_dict(data):
    kind = data.get("type")
    if kind == "empty": return EMPTY
    if kind == "universal": return UniversalSolutionSet(data["domain"])
    if kind == "finite": return FiniteSolutionSet(tuple(_decode_value(item) for item in data["values"]), tuple(data["multiplicities"]))
    if kind == "interval": return IntervalSolutionSet(_decode_value(data["lower"]), _decode_value(data["upper"]), data["lower_closed"], data["upper_closed"])
    if kind == "parametric": return ParametricSolutionSet(data["variable"], symbolic_from_dict(data["expression"]), data["parameter"], data["parameter_domain"])
    if kind == "union": return UnionSolutionSet(tuple(_solution_set_from_dict(item) for item in data["sets"]))
    if kind == "conditional":
        assumptions = data.get("assumptions")
        conditions = AssumptionSet.from_dict(assumptions).conditions if assumptions is not None else tuple(data["conditions"])
        return ConditionalSolutionSet(_solution_set_from_dict(data["solution_set"]), conditions)
    raise ValueError(f"Unknown solution set type {kind!r}")


# ---------------------------------------------------------------------------
# Algebra helpers and symbolic dispatch


def _poly_clean(polynomial):
    return {degree: coefficient for degree, coefficient in polynomial.items() if coefficient}


def _poly_add(first, second, factor=Rational(1)):
    result = dict(first)
    for degree, coefficient in second.items():
        result[degree] = result.get(degree, Rational(0)) + factor * coefficient
    return _poly_clean(result)


def _poly_mul(first, second, *, max_degree=100):
    result = {}
    for a, ca in first.items():
        for b, cb in second.items():
            degree = a + b
            if degree > max_degree:
                raise UnsupportedExpressionError(f"Polynomial degree exceeds the {max_degree} limit")
            result[degree] = result.get(degree, Rational(0)) + ca * cb
    return _poly_clean(result)


def _poly_pow(polynomial, exponent, *, max_degree=100):
    result, factor = {0: Rational(1)}, polynomial
    while exponent:
        if exponent & 1:
            result = _poly_mul(result, factor, max_degree=max_degree)
        exponent //= 2
        if exponent:
            factor = _poly_mul(factor, factor, max_degree=max_degree)
    return result


def _poly_fraction(expression, variable, *, max_degree=100):
    """Return exact numerator/denominator polynomials, or ``None``."""
    if isinstance(expression, ExactNumber):
        return {0: expression.value}, {0: Rational(1)}
    if isinstance(expression, Symbol):
        return ({1: Rational(1)}, {0: Rational(1)}) if expression.name == variable else None
    if isinstance(expression, SymbolicConstant):
        return None
    if isinstance(expression, Add):
        numerator, denominator = {}, {0: Rational(1)}
        for term in expression.terms:
            parsed = _poly_fraction(term, variable, max_degree=max_degree)
            if parsed is None:
                return None
            other_num, other_den = parsed
            numerator = _poly_add(_poly_mul(numerator, other_den, max_degree=max_degree), _poly_mul(other_num, denominator, max_degree=max_degree))
            denominator = _poly_mul(denominator, other_den, max_degree=max_degree)
        return numerator, denominator
    if isinstance(expression, Multiply):
        numerator, denominator = {0: Rational(1)}, {0: Rational(1)}
        for factor in expression.factors:
            parsed = _poly_fraction(factor, variable, max_degree=max_degree)
            if parsed is None:
                return None
            numerator = _poly_mul(numerator, parsed[0], max_degree=max_degree)
            denominator = _poly_mul(denominator, parsed[1], max_degree=max_degree)
        return numerator, denominator
    if isinstance(expression, Power) and isinstance(expression.exponent, ExactNumber) and expression.exponent.denominator == 1:
        exponent = expression.exponent.numerator
        parsed = _poly_fraction(expression.base, variable, max_degree=max_degree)
        if parsed is None:
            return None
        if exponent >= 0:
            return _poly_pow(parsed[0], exponent, max_degree=max_degree), _poly_pow(parsed[1], exponent, max_degree=max_degree)
        return _poly_pow(parsed[1], -exponent, max_degree=max_degree), _poly_pow(parsed[0], -exponent, max_degree=max_degree)
    return None


def _coefficient_list(polynomial):
    if not polynomial:
        return [Rational(0)]
    degree = max(polynomial)
    return [polynomial.get(power, Rational(0)) for power in range(degree, -1, -1)]


def _poly_eval(polynomial, value):
    result = Rational(0)
    for coefficient in _coefficient_list(polynomial):
        result = result * value + coefficient
    return result


def _synthetic(polynomial, root):
    coefficients = _coefficient_list(polynomial)
    result = [coefficients[0]]
    for coefficient in coefficients[1:]:
        result.append(coefficient + result[-1] * root)
    remainder = result.pop()
    degree = len(result) - 1
    return _poly_clean({degree - index: value for index, value in enumerate(result)}), remainder


def _poly_divmod(dividend, divisor):
    dividend, divisor = dict(_poly_clean(dividend)), _poly_clean(divisor)
    if not divisor:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = {}
    divisor_degree = max(divisor)
    while dividend and max(dividend) >= divisor_degree:
        degree = max(dividend) - divisor_degree
        coefficient = dividend[max(dividend)] / divisor[divisor_degree]
        quotient[degree] = quotient.get(degree, Rational(0)) + coefficient
        dividend = _poly_add(dividend, {power + degree: coefficient * value for power, value in divisor.items()}, factor=Rational(-1))
    return _poly_clean(quotient), _poly_clean(dividend)


def _poly_monic(polynomial):
    polynomial = _poly_clean(polynomial)
    if not polynomial:
        return {}
    leading = polynomial[max(polynomial)]
    return {degree: coefficient / leading for degree, coefficient in polynomial.items()}


def _poly_gcd(first, second):
    first, second = _poly_clean(first), _poly_clean(second)
    while second:
        _, remainder = _poly_divmod(first, second)
        first, second = second, remainder
    return _poly_monic(first)


def _normalization_variable(expression, variable):
    variables = expression.variables
    if isinstance(variable, Symbol):
        variable = variable.name
    if variable is not None and (not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z_]\w*", variable)):
        raise ValueError("normalization variable must be a valid identifier")
    if variable is None:
        if len(variables) > 1:
            raise AmbiguousVariableError("Specify variable= for multivariable normalization")
        variable = next(iter(variables), "x")
    unsupported = variables - {variable}
    if unsupported:
        raise UnsupportedExpressionError(
            "Polynomial normalization currently requires exact numeric coefficients; "
            f"unsupported symbols: {', '.join(sorted(unsupported))}"
        )
    return variable


def _normalization_degree(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("max_degree must be a nonnegative integer")
    return int(value)


def _polynomial_expression(polynomial, variable):
    terms = []
    symbol = Symbol(variable)
    for degree in sorted(polynomial, reverse=True):
        coefficient = polynomial[degree]
        if not coefficient:
            continue
        if degree == 0:
            terms.append(ExactNumber(coefficient))
            continue
        power = symbol if degree == 1 else _pow(symbol, ExactNumber(degree))
        if coefficient == 1:
            terms.append(power)
        elif coefficient == -1:
            terms.append(_neg(power))
        else:
            terms.append(_mul(ExactNumber(coefficient), power))
    return _add(*terms)


def _primitive_rational_pair(numerator, denominator):
    if not denominator:
        raise ZeroDivisionError("rational expression has an identically zero denominator")
    values = tuple(numerator.values()) + tuple(denominator.values())
    scale = reduce(
        lambda first, second: abs(first * second) // math.gcd(first, second),
        (value.denominator for value in values), 1,
    )
    integers = [int(value * scale) for value in values]
    content = reduce(math.gcd, (abs(value) for value in integers if value), 0) or 1
    factor = Rational(scale, content)
    numerator = {degree: coefficient * factor for degree, coefficient in numerator.items()}
    denominator = {degree: coefficient * factor for degree, coefficient in denominator.items()}
    if denominator[max(denominator)] < 0:
        numerator = {degree: -coefficient for degree, coefficient in numerator.items()}
        denominator = {degree: -coefficient for degree, coefficient in denominator.items()}
    return _poly_clean(numerator), _poly_clean(denominator)


def _rational_normalization_data(expression, variable, max_degree):
    expression = simplify_symbolic(expression)
    variable = _normalization_variable(expression, variable)
    max_degree = _normalization_degree(max_degree)
    parsed = _poly_fraction(expression, variable, max_degree=max_degree)
    if parsed is None:
        raise UnsupportedExpressionError("Expression is not a univariate rational polynomial")
    numerator, denominator = map(_poly_clean, parsed)
    if not denominator:
        raise ZeroDivisionError("rational expression has an identically zero denominator")
    common = _poly_gcd(numerator, denominator)
    cancelled = common if common and max(common) > 0 else None
    if cancelled is not None:
        numerator, numerator_remainder = _poly_divmod(numerator, cancelled)
        denominator, denominator_remainder = _poly_divmod(denominator, cancelled)
        if numerator_remainder or denominator_remainder:  # pragma: no cover - exact GCD invariant
            raise ArithmeticError("polynomial GCD did not divide exactly")
    numerator, denominator = _primitive_rational_pair(numerator, denominator)
    numerator_expression = _polynomial_expression(numerator, variable)
    denominator_expression = _polynomial_expression(denominator, variable)
    normalized = (
        numerator_expression
        if denominator == {0: Rational(1)}
        else _mul(numerator_expression, _pow(denominator_expression, NEG_ONE))
    )
    condition = None
    if cancelled is not None:
        condition = _condition_predicate(
            _polynomial_expression(cancelled, variable), "!= 0", variable,
        )
    return normalized, variable, condition


def normalize_polynomial_symbolic(value, variable=None, *, max_degree=100):
    """Return the exact expanded canonical form of one univariate polynomial."""
    expression = simplify_symbolic(value)
    variable = _normalization_variable(expression, variable)
    max_degree = _normalization_degree(max_degree)
    parsed = _poly_fraction(expression, variable, max_degree=max_degree)
    if parsed is None or not parsed[1] or any(degree != 0 for degree in parsed[1]):
        raise UnsupportedExpressionError("Expression is not a univariate polynomial")
    denominator = parsed[1].get(0, Rational(0))
    if not denominator:
        raise ZeroDivisionError("polynomial expression has a zero scalar denominator")
    polynomial = {degree: coefficient / denominator for degree, coefficient in parsed[0].items()}
    return _polynomial_expression(_poly_clean(polynomial), variable)


def normalize_rational_symbolic(value, variable=None, assumptions=None, *,
                                domain="real", max_degree=100):
    """Normalize and exactly cancel a univariate rational expression.

    The returned :class:`RewriteResult` retains the source domain and records
    the nonzero condition for every cancelled polynomial GCD.
    """
    if domain not in {"real", "complex"}:
        raise ValueError("domain must be 'real' or 'complex'")
    source = simplify_symbolic(value)
    normalized, variable, cancellation = _rational_normalization_data(
        source, variable, max_degree,
    )
    source_conditions = _domain_conditions(source, domain, variable)
    required = AssumptionSet(() if cancellation is None else (cancellation,))
    candidate_conditions = _domain_conditions(normalized, domain, variable)
    combined = _normalize_user_assumptions(assumptions).merge(
        source_conditions, candidate_conditions, required,
    )
    applications = ()
    if normalized != source:
        applications = (RewriteApplication(
            "normalize-rational", (), source, normalized, required,
            AssumptionSet(),
            "Expand the exact rational form and cancel its polynomial GCD while retaining excluded points.",
        ),)
    return RewriteResult(
        normalized, combined, applications, "fixed_point",
        "The expression is in primitive univariate rational form.",
    )


def _sturm_sequence(polynomial):
    """Return the exact Sturm chain for a square-free rational polynomial."""
    first = _poly_monic(polynomial)
    second = _poly_clean({degree - 1: degree * coefficient for degree, coefficient in first.items() if degree})
    sequence = [first]
    if not second:
        return sequence
    sequence.append(second)
    while second:
        _, remainder = _poly_divmod(first, second)
        remainder = {degree: -coefficient for degree, coefficient in remainder.items()}
        if not remainder:
            break
        sequence.append(remainder)
        first, second = second, remainder
    return sequence


def _sign_variations(values):
    signs = [1 if value > 0 else -1 for value in values if value]
    return sum(first != second for first, second in zip(signs, signs[1:]))


def _sturm_variations(sequence, point):
    return _sign_variations(_poly_eval(polynomial, point) for polynomial in sequence)


def _real_root_bound(polynomial):
    degree = max(polynomial)
    leading = abs(polynomial[degree])
    largest = max((abs(coefficient) / leading for power, coefficient in polynomial.items() if power != degree), default=Rational(0))
    return largest.numerator // largest.denominator + 2


def _isolate_real_roots(polynomial, *, max_splits=20000):
    """Return disjoint rational intervals, each containing exactly one root."""
    sequence = _sturm_sequence(polynomial)
    bound = Rational(_real_root_bound(polynomial))

    def count(lower, upper):
        return _sturm_variations(sequence, lower) - _sturm_variations(sequence, upper)

    total = count(-bound, bound)
    pending = [(-bound, bound, total)] if total else []
    isolated, splits = [], 0
    while pending:
        lower, upper, roots = pending.pop()
        if roots == 1:
            isolated.append((lower, upper))
            continue
        midpoint = (lower + upper) / 2
        if _poly_eval(polynomial, midpoint) == 0:
            # Rational roots are normally extracted before Sturm isolation.
            # Retain a small exact bracket if one reaches this defensive path.
            width = (upper - lower) / 4
            isolated.append((midpoint - width, midpoint + width))
            left_count = count(lower, midpoint - width)
            right_count = count(midpoint + width, upper)
            if left_count: pending.append((lower, midpoint - width, left_count))
            if right_count: pending.append((midpoint + width, upper, right_count))
        else:
            left_count = count(lower, midpoint)
            right_count = roots - left_count
            if left_count: pending.append((lower, midpoint, left_count))
            if right_count: pending.append((midpoint, upper, right_count))
        splits += 1
        if splits > max_splits:
            raise UnsupportedExpressionError("Exact real-root isolation exceeded its transformation limit")
    isolated.sort(key=lambda bounds: bounds[0])
    return tuple(isolated)


def _square_free_factors(polynomial):
    polynomial = _poly_monic(polynomial)
    if max(polynomial, default=0) <= 1:
        return [(polynomial, 1)]
    derivative = {degree - 1: degree * coefficient for degree, coefficient in polynomial.items() if degree}
    common = _poly_gcd(polynomial, derivative)
    square_free, remainder = _poly_divmod(polynomial, common)
    assert not remainder
    factors, multiplicity = [], 1
    while square_free and max(square_free, default=0) > 0:
        repeated = _poly_gcd(square_free, common)
        factor, remainder = _poly_divmod(square_free, repeated)
        assert not remainder
        if max(factor, default=0) > 0:
            factors.append((_poly_monic(factor), multiplicity))
        square_free = repeated
        common, remainder = _poly_divmod(common, repeated)
        assert not remainder
        multiplicity += 1
    return factors or [(polynomial, 1)]


def _integer_divisors(value):
    value = abs(int(value))
    if value == 0:
        return {0}
    factors = set()
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate == 0:
            factors.update((candidate, value // candidate, -candidate, -(value // candidate)))
    return factors


def _rational_candidates(polynomial):
    coefficients = _coefficient_list(polynomial)
    denominator = reduce(lambda first, second: abs(first * second) // math.gcd(first, second), (value.denominator for value in coefficients), 1)
    integers = [int(value * denominator) for value in coefficients]
    if integers[-1] == 0:
        return [Rational(0)]
    return sorted({Rational(p, q) for p in _integer_divisors(integers[-1]) for q in _integer_divisors(integers[0]) if q}, key=float)


def _sqrt_exact(value, domain):
    if value < 0:
        if domain == "real":
            return None
        return _mul(I, _sqrt_exact(-value, "real"))
    # Rationalize first, then extract the largest integer square.  This keeps
    # exact trigonometric roots recognizable (sqrt(32)/8 -> sqrt(2)/2) and
    # avoids needlessly large radical display forms.
    radicand = value.numerator * value.denominator
    outside, remaining, factor = 1, radicand, 2
    while factor <= 10000 and factor * factor <= remaining:
        square = factor * factor
        while remaining % square == 0:
            outside *= factor
            remaining //= square
        factor += 1
    remaining_root = math.isqrt(remaining)
    if remaining_root * remaining_root == remaining:
        outside *= remaining_root
        remaining = 1
    coefficient = ExactNumber(outside, value.denominator)
    return (
        coefficient if remaining == 1
        else _mul(coefficient, _function("sqrt", ExactNumber(remaining)))
    )


def _numeric_value(value):
    try:
        result = value.evaluate({}) if isinstance(value, SymbolicExpression) else value
        result = complex(result)
        return result.real if abs(result.imag) <= 1e-12 else result
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _collapse_guarded_zeros(expression):
    """Drop domain-preserving zero factors after restrictions were collected."""
    if isinstance(expression, Multiply):
        if any(factor == ZERO for factor in expression.factors):
            return ZERO
        return _mul(*(_collapse_guarded_zeros(factor) for factor in expression.factors))
    if isinstance(expression, Add):
        return _add(*(_collapse_guarded_zeros(term) for term in expression.terms))
    if isinstance(expression, Power):
        return _pow(_collapse_guarded_zeros(expression.base), _collapse_guarded_zeros(expression.exponent))
    if isinstance(expression, SymbolicFunction):
        return _function(expression.name, *(_collapse_guarded_zeros(argument) for argument in expression.arguments))
    return expression


def _condition_predicate(expression, relation, variable):
    relation_operator, reference = {
        "!= 0": ("!=", ZERO),
        "> 0": (">", ZERO),
        ">= 0": (">=", ZERO),
        "<= 1": ("<=", ONE),
        ">= -1": (">=", NEG_ONE),
        "!= 1": ("!=", ONE),
    }[relation]
    if (
        relation == ">= 0"
        and isinstance(expression, Power)
        and isinstance(expression.exponent, ExactNumber)
        and expression.exponent.denominator == 1
        and expression.exponent.numerator > 0
        and expression.exponent.numerator % 2 == 0
        and _structurally_real(expression.base)
    ):
        # A positive even integer power of a real-defined base is always
        # nonnegative. The base is visited separately, so any restrictions
        # required for the base itself are still retained.
        return None
    numeric = _numeric_value(expression)
    if numeric is not None:
        predicate = RelationCondition(expression, relation_operator, reference)
        return None if predicate.evaluate({}, tolerance=0.0) else TruthCondition(False)
    rendered = str(expression)
    parsed = _poly_fraction(expression, variable)
    if parsed is not None and parsed[1] == {0: Rational(1)}:
        rendered = _poly_string(parsed[0], variable)
    return RelationCondition(expression, relation_operator, reference, f"{rendered} {relation}")


def _domain_conditions(expression, domain, variable):
    conditions = []

    def add(value, relation):
        condition = _condition_predicate(value, relation, variable)
        if condition is not None and condition not in conditions:
            conditions.append(condition)

    def visit(value):
        if isinstance(value, Power):
            visit(value.base); visit(value.exponent)
            if isinstance(value.exponent, ExactNumber):
                exponent = value.exponent.value
                if exponent < 0:
                    add(value.base, "!= 0")
                if domain == "real" and exponent.denominator % 2 == 0:
                    add(value.base, ">= 0")
            else:
                condition = DefinedCondition(value, domain)
                if condition not in conditions:
                    conditions.append(condition)
            return
        if isinstance(value, SymbolicFunction):
            for argument in value.arguments:
                visit(argument)
            argument = value.arguments[-1]
            if value.name == "sqrt" and domain == "real":
                add(argument, ">= 0")
            elif value.name in {"ln", "log"}:
                add(argument, "> 0" if domain == "real" else "!= 0")
                if value.name == "log" and len(value.arguments) == 2:
                    base = value.arguments[0]
                    add(base, "> 0" if domain == "real" else "!= 0")
                    add(base, "!= 1")
            elif value.name == "tan":
                add(_function("cos", argument), "!= 0")
            elif value.name in {"asin", "acos"} and domain == "real":
                add(argument, ">= -1"); add(argument, "<= 1")
            return
        if isinstance(value, Add):
            for term in value.terms: visit(term)
        elif isinstance(value, Multiply):
            for factor in value.factors: visit(factor)

    visit(expression)
    return AssumptionSet(tuple(conditions))


def _merge_conditions(*groups):
    result = AssumptionSet()
    for group in groups:
        result = result.merge(group)
    return result


def _render_condition(expression, relation, variable):
    parsed = _poly_fraction(expression, variable)
    rendered = _poly_string(parsed[0], variable) if parsed is not None and parsed[1] == {0: Rational(1)} else str(expression)
    predicate = _condition_predicate(expression, relation, variable)
    # Keep historically exposed tautologies such as ``2 > 0`` in the display
    # conditions while structural domain collection is free to discard them.
    return predicate or OpaqueCondition(f"{rendered} {relation}")


def _in_interval(value, interval):
    if interval is None:
        return True
    numeric = _numeric_value(value)
    return numeric is not None and not isinstance(numeric, complex) and interval[0] - 1e-12 <= numeric <= interval[1] + 1e-12


def _solve_polynomial_exact(polynomial, variable, domain, interval=None):
    polynomial = _poly_clean(polynomial)
    if not polynomial:
        return UniversalSolutionSet(domain), True
    degree = max(polynomial)
    if degree == 0:
        return EMPTY, True
    square_free_factors = _square_free_factors(polynomial)
    if len(square_free_factors) > 1 or square_free_factors[0][1] > 1:
        combined = {}
        for factor, factor_multiplicity in square_free_factors:
            factor_set, _ = _solve_polynomial_exact(factor, variable, domain, interval)
            if isinstance(factor_set, FiniteSolutionSet):
                for root, multiplicity in zip(factor_set.values, factor_set.multiplicities):
                    combined[root] = combined.get(root, 0) + multiplicity * factor_multiplicity
        ordered = sorted(combined.items(), key=lambda pair: (complex(_numeric_value(pair[0]) or 0).real, complex(_numeric_value(pair[0]) or 0).imag, str(pair[0])))
        return FiniteSolutionSet(tuple(root for root, _ in ordered), tuple(value for _, value in ordered)) if ordered else EMPTY, True
    roots, multiplicities = [], []
    remaining = polynomial
    while remaining and max(remaining) > 2:
        found = None
        for candidate in _rational_candidates(remaining):
            if _poly_eval(remaining, candidate) == 0:
                found = candidate
                break
        if found is None:
            break
        multiplicity = 0
        while max(remaining) and _poly_eval(remaining, found) == 0:
            remaining, remainder = _synthetic(remaining, found)
            if remainder:
                break
            multiplicity += 1
        roots.append(ExactNumber(found)); multiplicities.append(multiplicity)
    degree = max(remaining) if remaining else 0
    if degree == 1:
        root = ExactNumber(-remaining.get(0, 0) / remaining[1])
        roots.append(root); multiplicities.append(1)
    elif degree == 2:
        a, b, c = remaining[2], remaining.get(1, 0), remaining.get(0, 0)
        discriminant = b * b - 4 * a * c
        radical = _sqrt_exact(discriminant, domain)
        if radical is not None:
            center, scale = ExactNumber(-b / (2 * a)), ExactNumber(Rational(1, 1) / (2 * a))
            first = _add(center, _mul(scale, radical))
            second = _add(center, _neg(_mul(scale, radical)))
            if first == second:
                roots.append(first); multiplicities.append(2)
            else:
                roots.extend((first, second)); multiplicities.extend((1, 1))
    elif degree > 2:
        coefficients = tuple(ExactNumber(value) for value in _coefficient_list(remaining))
        if domain == "real":
            for index, bounds in enumerate(_isolate_real_roots(remaining)):
                lower, upper = map(float, bounds)
                if not math.isfinite(lower) or not math.isfinite(upper):
                    raise UnsupportedExpressionError("A certified root interval exceeds floating-point range")
                roots.append(RootOf(coefficients, index, (lower, upper)))
                multiplicities.append(1)
        else:
            # The fundamental theorem of algebra certifies the number of
            # complex roots.  RootOf keeps the exact polynomial; numerical
            # approximation is used only to display/evaluate its stable index.
            roots.extend(RootOf(coefficients, index) for index in range(degree))
            multiplicities.extend([1] * degree)
    kept = [(root, multiplicity) for root, multiplicity in zip(roots, multiplicities) if _in_interval(root, interval)]
    kept.sort(key=lambda pair: (complex(_numeric_value(pair[0]) or 0).real, complex(_numeric_value(pair[0]) or 0).imag, str(pair[0])))
    return FiniteSolutionSet(tuple(root for root, _ in kept), tuple(multiplicity for _, multiplicity in kept)) if kept else EMPTY, True


def _affine(expression, variable):
    try:
        parsed = _poly_fraction(expression, variable, max_degree=1)
    except UnsupportedExpressionError:
        return None
    if parsed is None or parsed[1] != {0: Rational(1)} or max(parsed[0], default=0) > 1:
        return None
    return parsed[0].get(1, Rational(0)), parsed[0].get(0, Rational(0))


def _affine_symbolic(expression, variable):
    """Return symbolic ``(coefficient, constant)`` for an affine expression."""
    if variable not in _variables(expression):
        return ZERO, expression
    if isinstance(expression, Symbol) and expression.name == variable:
        return ONE, ZERO
    if isinstance(expression, Add):
        coefficient, constant = ZERO, ZERO
        for term in expression.terms:
            parsed = _affine_symbolic(term, variable)
            if parsed is None:
                return None
            coefficient, constant = _add(coefficient, parsed[0]), _add(constant, parsed[1])
        return coefficient, constant
    if isinstance(expression, Multiply):
        variable_factor, constants = None, []
        for factor in expression.factors:
            if variable in _variables(factor):
                if variable_factor is not None:
                    return None
                variable_factor = factor
            else:
                constants.append(factor)
        parsed = _affine_symbolic(variable_factor, variable) if variable_factor is not None else (ZERO, ONE)
        if parsed is None:
            return None
        scale = _mul(*constants) if constants else ONE
        return _mul(scale, parsed[0]), _mul(scale, parsed[1])
    return None


def _solve_affine_equal(expression, right, variable):
    affine = _affine(expression, variable)
    if affine is None or not isinstance(right, ExactNumber) or not affine[0]:
        return None
    return ExactNumber((right.value - affine[1]) / affine[0])


def _same_function_side(left, right):
    if isinstance(left, SymbolicFunction) and not isinstance(right, SymbolicFunction):
        return left, right
    if isinstance(right, SymbolicFunction) and not isinstance(left, SymbolicFunction):
        return right, left
    return None


def _log_terms(expression):
    terms = expression.terms if isinstance(expression, Add) else (expression,)
    if not terms or not all(isinstance(term, SymbolicFunction) and term.name in {"ln", "log"} for term in terms):
        return None
    signatures = [(term.name, term.arguments[:-1]) for term in terms]
    if any(signature != signatures[0] for signature in signatures[1:]):
        return None
    return signatures[0], tuple(term.arguments[-1] for term in terms)


def _trig_phase(name, value):
    half_sqrt2 = _mul(ExactNumber(1, 2), _function("sqrt", ExactNumber(2)))
    half_sqrt3 = _mul(ExactNumber(1, 2), _function("sqrt", ExactNumber(3)))
    table = {
        "sin": {
            NEG_ONE: _neg(_mul(ExactNumber(1, 2), PI)),
            ExactNumber(-1, 2): _neg(_mul(ExactNumber(1, 6), PI)),
            _neg(half_sqrt2): _neg(_mul(ExactNumber(1, 4), PI)),
            _neg(half_sqrt3): _neg(_mul(ExactNumber(1, 3), PI)),
            ZERO: ZERO, ExactNumber(1, 2): _mul(ExactNumber(1, 6), PI),
            half_sqrt2: _mul(ExactNumber(1, 4), PI),
            half_sqrt3: _mul(ExactNumber(1, 3), PI),
            ONE: _mul(ExactNumber(1, 2), PI),
        },
        "cos": {
            NEG_ONE: PI, ExactNumber(-1, 2): _mul(ExactNumber(2, 3), PI),
            _neg(half_sqrt2): _mul(ExactNumber(3, 4), PI),
            _neg(half_sqrt3): _mul(ExactNumber(5, 6), PI),
            ZERO: _mul(ExactNumber(1, 2), PI),
            ExactNumber(1, 2): _mul(ExactNumber(1, 3), PI),
            half_sqrt2: _mul(ExactNumber(1, 4), PI),
            half_sqrt3: _mul(ExactNumber(1, 6), PI), ONE: ZERO,
        },
        "tan": {NEG_ONE: _neg(_mul(ExactNumber(1, 4), PI)), ZERO: ZERO, ONE: _mul(ExactNumber(1, 4), PI)},
    }
    return table.get(name, {}).get(value)


def _parameter_family(variable, a, b, phase, period, sign=1):
    parameter = next(name for name in ("n", "k", "m", "j", "_n") if name != variable)
    index = Symbol(parameter)
    numerator = _add(_mul(ExactNumber(sign), phase), _mul(period, index), ExactNumber(-b))
    return ParametricSolutionSet(variable, _mul(ExactNumber(1 / a), numerator), parameter)


def _exact_log_ratio(base, value):
    if not isinstance(base, ExactNumber) or not isinstance(value, ExactNumber):
        return None
    if base.value <= 0 or base.value == 1 or value.value <= 0:
        return None
    for exponent in range(-64, 65):
        if base.value ** exponent == value.value:
            return ExactNumber(exponent)
    return None


def _filter_parametric(solution_set, interval):
    if interval is None:
        return solution_set
    families = solution_set.sets if isinstance(solution_set, UnionSolutionSet) else (solution_set,)
    values = []
    for family in families:
        at_zero = _numeric_value(family.expression.substitute({family.parameter: 0}))
        at_one = _numeric_value(family.expression.substitute({family.parameter: 1}))
        if at_zero is None or at_one is None or isinstance(at_zero, complex) or isinstance(at_one, complex) or at_one == at_zero:
            raise UnsupportedExpressionError("Cannot bound the parametric solution family")
        step = at_one - at_zero
        first = math.ceil(min((interval[0] - at_zero) / step, (interval[1] - at_zero) / step) - 1e-12)
        last = math.floor(max((interval[0] - at_zero) / step, (interval[1] - at_zero) / step) + 1e-12)
        if last - first + 1 > 10000:
            raise UnsupportedExpressionError("Bounded periodic solution exceeds the 10000-branch limit")
        for n in range(first, last + 1):
            value = family.expression.substitute({family.parameter: n})
            numeric = _numeric_value(value)
            if numeric is not None and not isinstance(numeric, complex) and interval[0] - 1e-12 <= numeric <= interval[1] + 1e-12:
                values.append(value)
    unique = {str(value): value for value in values}
    ordered = sorted(unique.values(), key=lambda value: float(_numeric_value(value)))
    return FiniteSolutionSet(tuple(ordered)) if ordered else EMPTY


def _interval_conditions(value, interval):
    expression = value if isinstance(value, SymbolicExpression) else _coerce(value)
    lower, upper = _coerce(interval[0]), _coerce(interval[1])
    return (BetweenCondition(
        expression, lower, upper, True, True,
        f"{interval[0]:g} <= {value} <= {interval[1]:g}",
    ),)


def _apply_interval(solution_set, interval):
    """Intersect every supported solution-set shape with a real interval."""
    if interval is None or isinstance(solution_set, EmptySolutionSet):
        return solution_set
    if isinstance(solution_set, UniversalSolutionSet):
        return IntervalSolutionSet(interval[0], interval[1])
    if isinstance(solution_set, ParametricSolutionSet):
        return _filter_parametric(solution_set, interval)
    if isinstance(solution_set, UnionSolutionSet):
        parts = []
        for subset in solution_set.sets:
            bounded = _apply_interval(subset, interval)
            if isinstance(bounded, EmptySolutionSet):
                continue
            if isinstance(bounded, UnionSolutionSet):
                parts.extend(bounded.sets)
            else:
                parts.append(bounded)
        if not parts:
            return EMPTY
        return parts[0] if len(parts) == 1 else UnionSolutionSet(tuple(parts))
    if isinstance(solution_set, ConditionalSolutionSet):
        bounded = _apply_interval(solution_set.solution_set, interval)
        if isinstance(bounded, EmptySolutionSet):
            return EMPTY
        conditions = solution_set.conditions
        if isinstance(bounded, ConditionalSolutionSet):
            conditions = _merge_conditions(conditions, bounded.conditions)
            bounded = bounded.solution_set
        return ConditionalSolutionSet(bounded, conditions)
    if isinstance(solution_set, IntervalSolutionSet):
        lower_numeric, upper_numeric = _numeric_value(solution_set.lower), _numeric_value(solution_set.upper)
        if solution_set.lower is None:
            lower_numeric = -math.inf
        if solution_set.upper is None:
            upper_numeric = math.inf
        if lower_numeric is None or upper_numeric is None:
            conditions = _interval_conditions(solution_set.lower, interval) + _interval_conditions(solution_set.upper, interval)
            return ConditionalSolutionSet(solution_set, conditions)
        lower, upper = max(float(lower_numeric), interval[0]), min(float(upper_numeric), interval[1])
        if lower > upper:
            return EMPTY
        return IntervalSolutionSet(
            lower, upper,
            True if interval[0] > float(lower_numeric) else solution_set.lower_closed,
            True if interval[1] < float(upper_numeric) else solution_set.upper_closed,
        )
    if isinstance(solution_set, FiniteSolutionSet):
        known_values, known_multiplicities, symbolic_parts = [], [], []
        for value, multiplicity in zip(solution_set.values, solution_set.multiplicities):
            numeric = _numeric_value(value)
            if numeric is None:
                symbolic_parts.append(ConditionalSolutionSet(
                    FiniteSolutionSet((value,), (multiplicity,)),
                    _interval_conditions(value, interval),
                ))
            elif not isinstance(numeric, complex) and interval[0] - 1e-12 <= numeric <= interval[1] + 1e-12:
                known_values.append(value); known_multiplicities.append(multiplicity)
        parts = list(symbolic_parts)
        if known_values:
            parts.insert(0, FiniteSolutionSet(tuple(known_values), tuple(known_multiplicities)))
        if not parts:
            return EMPTY
        return parts[0] if len(parts) == 1 else UnionSolutionSet(tuple(parts))
    raise TypeError(f"Unsupported solution set {type(solution_set).__name__}")


def _apply_assumptions(solution_set, assumptions):
    """Resolve conditional branches when caller assumptions prove a choice."""
    assumptions = _coerce_assumption_set(assumptions)
    if assumptions.contradictory or isinstance(solution_set, EmptySolutionSet):
        return EMPTY
    if isinstance(solution_set, ConditionalSolutionSet):
        branch = solution_set.assumptions
        if any(assumptions.refutes(condition) for condition in branch.conditions):
            return EMPTY
        unresolved = tuple(condition for condition in branch.conditions if not assumptions.entails(condition))
        nested = _apply_assumptions(solution_set.solution_set, assumptions.merge(branch))
        if isinstance(nested, EmptySolutionSet) or not unresolved:
            return nested
        return ConditionalSolutionSet(nested, unresolved)
    if isinstance(solution_set, UnionSolutionSet):
        parts = []
        for subset in solution_set.sets:
            resolved = _apply_assumptions(subset, assumptions)
            if isinstance(resolved, EmptySolutionSet):
                continue
            if isinstance(resolved, UnionSolutionSet):
                parts.extend(resolved.sets)
            else:
                parts.append(resolved)
        return EMPTY if not parts else parts[0] if len(parts) == 1 else UnionSolutionSet(tuple(parts))
    return solution_set


def _substitute_solution_set(solution_set, values):
    """Apply known parameter values without mutating a solution-set tree."""
    if not values or isinstance(solution_set, (EmptySolutionSet, UniversalSolutionSet)):
        return solution_set

    def substitute_value(value):
        return value.substitute(values) if isinstance(value, SymbolicExpression) else value

    if isinstance(solution_set, FiniteSolutionSet):
        return FiniteSolutionSet(tuple(substitute_value(value) for value in solution_set.values), solution_set.multiplicities)
    if isinstance(solution_set, IntervalSolutionSet):
        return IntervalSolutionSet(
            substitute_value(solution_set.lower), substitute_value(solution_set.upper),
            solution_set.lower_closed, solution_set.upper_closed,
        )
    if isinstance(solution_set, ParametricSolutionSet):
        safe_values = {name: value for name, value in values.items() if name != solution_set.parameter}
        return ParametricSolutionSet(
            solution_set.variable, solution_set.expression.substitute(safe_values),
            solution_set.parameter, solution_set.parameter_domain,
        )
    if isinstance(solution_set, UnionSolutionSet):
        parts = tuple(_substitute_solution_set(item, values) for item in solution_set.sets)
        parts = tuple(item for item in parts if not isinstance(item, EmptySolutionSet))
        return EMPTY if not parts else parts[0] if len(parts) == 1 else UnionSolutionSet(parts)
    if isinstance(solution_set, ConditionalSolutionSet):
        branch = solution_set.assumptions.substitute(values)
        if branch.contradictory:
            return EMPTY
        nested = _substitute_solution_set(solution_set.solution_set, values)
        return nested if not branch else ConditionalSolutionSet(nested, branch.conditions)
    raise TypeError(f"Unsupported solution set {type(solution_set).__name__}")


def _validate_interval(interval):
    if interval is None:
        return None
    if isinstance(interval, (str, bytes)):
        raise TypeError("interval must be a finite (lower, upper) pair")
    try:
        lower, upper = interval
    except (TypeError, ValueError) as error:
        raise ValueError("interval must contain exactly two bounds") from error
    if not all(isinstance(value, (int, float, np.number)) and np.isfinite(value) for value in (lower, upper)):
        raise ValueError("interval bounds must be finite real numbers")
    lower, upper = float(lower), float(upper)
    if lower >= upper:
        raise ValueError("interval lower bound must be smaller than its upper bound")
    return lower, upper


def _source_variables(equation):
    """Discover identifiers before canonicalization can cancel them."""
    if isinstance(equation, str):
        parts = equation.split("=")
        names = set()
        for part in parts:
            for token in _tokenize(part):
                if token.kind == "identifier" and token.value not in _FUNCTIONS | {"pi", "e", "i"}:
                    names.add(token.value)
        return names
    if isinstance(equation, Sequence) and not isinstance(equation, (str, bytes)):
        names = set()
        for value in equation:
            if isinstance(value, str):
                names.update(_source_variables(value))
            elif isinstance(value, SymbolicExpression):
                names.update(_variables(value))
            else:
                names.update(getattr(value, "variables", ()))
        return names
    names = getattr(equation, "variables", None)
    if names is not None:
        return set(names)
    try:
        return _source_variables(str(equation))
    except (EquationParseError, UnsupportedExpressionError):
        return set()


def _equation_input(equation):
    from kiwicalc.equations.single import Equation
    from kiwicalc.core.interfaces import IExpression
    if isinstance(equation, Equation):
        equation = str(equation)
    if isinstance(equation, str):
        return _split_symbolic_equation(equation)
    if isinstance(equation, Sequence) and not isinstance(equation, (str, bytes)) and len(equation) == 2:
        converted = []
        for side in equation:
            if isinstance(side, SymbolicExpression):
                converted.append(side)
            elif isinstance(side, IExpression) or isinstance(side, (str, int, float, Rational)):
                converted.append(parse_symbolic(str(side)) if not isinstance(side, (int, float, Rational)) else _coerce(side))
            else:
                raise TypeError("Equation sides must be strings, numbers, or expression objects")
        return tuple(converted)
    raise TypeError("equation must be a string, Equation, or (left, right) expression pair")


def _candidate_valid(left, right, variable, candidate, domain, tolerance, assumptions=AssumptionSet()):
    numeric = _numeric_value(candidate)
    assignments = {
        name: (_numeric_value(value) if _numeric_value(value) is not None else value)
        for name, value in assumptions.substitutions.items()
    }
    if numeric is not None:
        assignments[variable] = numeric
    if numeric is not None and assumptions.evaluate(assignments, tolerance=tolerance) is False:
        return False, math.inf
    # Parameterized equations often contain roots that can be certified
    # structurally even though not every parameter has a numeric assignment.
    # For example, x=1 is always a root of (a*x+b)*(x-1)=0.  Prefer that exact
    # proof before attempting numerical evaluation of the remaining symbols.
    symbolic_assignments = dict(assumptions.substitutions)
    symbolic_assignments[variable] = candidate
    if _substitute(_add(left, _neg(right)), symbolic_assignments) == ZERO:
        return True, 0.0
    if isinstance(candidate, RootOf):
        return True, None
    if numeric is None:
        return True, None
    if domain == "real" and isinstance(numeric, complex):
        return False, math.inf
    try:
        first = complex(_evaluate(left, assignments))
        second = complex(_evaluate(right, assignments))
    except (ArithmeticError, ValueError, OverflowError):
        return False, math.inf
    if not all(math.isfinite(value) for value in (first.real, first.imag, second.real, second.imag)):
        return False, math.inf
    residual = abs(first - second) / max(1.0, abs(first), abs(second))
    return residual <= max(tolerance * 100, 1e-8), float(residual)


def _verify_finite(solution_set, left, right, variable, domain, tolerance, assumptions=AssumptionSet()):
    """Verify finite branches against the equation and active assumptions."""
    assumptions = _coerce_assumption_set(assumptions)
    if assumptions.contradictory:
        return EMPTY, ()
    if isinstance(solution_set, ConditionalSolutionSet):
        combined = assumptions.merge(solution_set.assumptions)
        verified, residuals = _verify_finite(solution_set.solution_set, left, right, variable, domain, tolerance, combined)
        return (EMPTY if isinstance(verified, EmptySolutionSet) else ConditionalSolutionSet(verified, solution_set.assumptions.conditions)), residuals
    if isinstance(solution_set, UnionSolutionSet):
        parts, residuals = [], []
        for subset in solution_set.sets:
            verified, subset_residuals = _verify_finite(subset, left, right, variable, domain, tolerance, assumptions)
            if not isinstance(verified, EmptySolutionSet):
                parts.append(verified)
                residuals.extend(subset_residuals)
        return (EMPTY if not parts else parts[0] if len(parts) == 1 else UnionSolutionSet(tuple(parts))), tuple(residuals)
    if not isinstance(solution_set, FiniteSolutionSet):
        return solution_set, ()
    kept, multiplicities, residuals = [], [], []
    for value, multiplicity in zip(solution_set.values, solution_set.multiplicities):
        valid, residual = _candidate_valid(left, right, variable, value, domain, tolerance, assumptions)
        if valid:
            kept.append(value); multiplicities.append(multiplicity); residuals.append(residual)
    return (FiniteSolutionSet(tuple(kept), tuple(multiplicities)) if kept else EMPTY), tuple(residuals)


@dataclass(frozen=True)
class _SolverRuleContext:
    """Immutable context shared by native equation-transformation rules."""

    variable: str
    domain: str
    interval: Optional[Tuple[float, float]]
    assumptions: AssumptionSet


@dataclass(frozen=True)
class _SolverRuleOutcome:
    """One equation rewrite or terminal solution produced by a solver rule."""

    after: Any
    complete: bool = True
    method: str = "symbolic"
    explanation: str = ""
    conditions: AssumptionSet = field(default_factory=AssumptionSet)

    def __post_init__(self):
        if not isinstance(self.after, (EquationState, SolutionSet)):
            raise TypeError("solver-rule outcomes must contain an equation or solution set")
        if not isinstance(self.complete, bool):
            raise TypeError("solver-rule completeness must be boolean")
        if self.method not in {"symbolic"}:
            raise ValueError("native solver rules currently support the symbolic method")
        if not isinstance(self.explanation, str):
            raise TypeError("solver-rule explanations must be text")
        object.__setattr__(self, "conditions", _coerce_assumption_set(self.conditions))

    @property
    def terminal(self):
        return isinstance(self.after, SolutionSet)


@dataclass(frozen=True)
class _SolverRule:
    """Ordered guarded rule used by the unified symbolic solver."""

    identifier: str
    transform: Callable = field(repr=False, compare=False)
    guard: Optional[Callable] = field(default=None, repr=False, compare=False)
    explanation: str = ""
    domains: Tuple[str, ...] = ("real", "complex")

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.identifier):
            raise ValueError("solver-rule identifiers must use lowercase snake_case")
        if not callable(self.transform):
            raise TypeError("solver-rule transform must be callable")
        if self.guard is not None and not callable(self.guard):
            raise TypeError("solver-rule guard must be callable")
        if not isinstance(self.explanation, str):
            raise TypeError("solver-rule explanation must be text")
        domains = tuple(self.domains)
        if not domains or any(domain not in {"real", "complex"} for domain in domains):
            raise ValueError("solver-rule domains must contain 'real' and/or 'complex'")
        object.__setattr__(self, "domains", domains)


def _solver_residual(state):
    return _collapse_guarded_zeros(state.residual)


def _solver_clear_denominators(state, context):
    rational = _poly_fraction(_solver_residual(state), context.variable)
    if rational is None or rational[1] == {0: Rational(1)}:
        return None
    return _SolverRuleOutcome(EquationState(
        _polynomial_expression(rational[0], context.variable), ZERO,
    ))


def _replace_subexpressions(expression, replacements):
    replacement = replacements.get(expression)
    if replacement is not None:
        return replacement
    if isinstance(expression, Add):
        return _add(*(_replace_subexpressions(term, replacements) for term in expression.terms))
    if isinstance(expression, Multiply):
        return _mul(*(_replace_subexpressions(factor, replacements) for factor in expression.factors))
    if isinstance(expression, Power):
        return _pow(
            _replace_subexpressions(expression.base, replacements),
            _replace_subexpressions(expression.exponent, replacements),
        )
    if isinstance(expression, SymbolicFunction):
        return _function(
            expression.name,
            *(_replace_subexpressions(argument, replacements) for argument in expression.arguments),
        )
    return expression


def _expression_occurrences(expression, target):
    if expression == target:
        return 1
    return sum(_expression_occurrences(child, target) for child in _rewrite_children(expression))


def _substitution_kernels(expression, variable):
    """Return deterministic repeated/nonlinear kernels supported by solver rules."""
    kernels = set()

    def visit(value):
        if isinstance(value, SymbolicFunction) and variable in value.variables:
            if value.name in {"abs", "exp", "ln", "log", "sin", "cos", "tan", "sqrt"}:
                kernels.add(value)
        elif (
            isinstance(value, Power) and isinstance(value.exponent, ExactNumber)
            and value.exponent.denominator == 1 and value.exponent.numerator > 1
            and variable in value.base.variables
            and _affine_symbolic(value.base, variable) is None
        ):
            # A repeated nonlinear algebraic base, such as x^2+1 or x+1/x.
            kernels.add(value.base)
        elif (
            isinstance(value, Power) and variable in value.exponent.variables
            and variable not in value.base.variables
        ):
            kernels.add(value)
        for child in _rewrite_children(value):
            visit(child)

    visit(expression)
    return tuple(sorted(kernels, key=lambda value: (_rewrite_node_count(value), str(value))))


def _fresh_substitution_symbol(expression):
    name = "_kiwi_u"
    while name in expression.variables:
        name = "_" + name
    return Symbol(name)


def _integer_affine_ratio(expression, reference, variable):
    affine = _affine(expression, variable)
    reference_affine = _affine(reference, variable)
    if not affine or not reference_affine or not reference_affine[0]:
        return None
    ratio = affine[0] / reference_affine[0]
    if ratio.denominator != 1 or ratio <= 0 or affine[1] != reference_affine[1] * ratio:
        return None
    return int(ratio)


def _exact_positive_base_power(reference, value, max_power=16):
    if not isinstance(reference, ExactNumber) or not isinstance(value, ExactNumber):
        return None
    if reference.value <= 0 or reference.value == 1 or value.value <= 0:
        return None
    result = Rational(1)
    for power in range(1, max_power + 1):
        result *= reference.value
        if result == value.value:
            return power
    return None


def _family_substitution_maps(expression, variable, auxiliary):
    """Yield common exponential-family maps such as exp(2x)->u^2."""
    nodes = []

    def visit(value):
        if (
            isinstance(value, SymbolicFunction) and value.name == "exp"
            and variable in value.arguments[-1].variables
        ) or (
            isinstance(value, Power) and variable in value.exponent.variables
            and variable not in value.base.variables
        ):
            nodes.append(value)
        for child in _rewrite_children(value):
            visit(child)

    visit(expression)
    nodes = tuple(dict.fromkeys(nodes))
    for reference in sorted(nodes, key=str):
        mapping = {}
        for node in nodes:
            if isinstance(reference, SymbolicFunction) != isinstance(node, SymbolicFunction):
                break
            if isinstance(reference, SymbolicFunction):
                ratio = _integer_affine_ratio(
                    node.arguments[-1], reference.arguments[-1], variable,
                )
            else:
                base_power = _exact_positive_base_power(reference.base, node.base)
                exponent_ratio = _integer_affine_ratio(
                    node.exponent, reference.exponent, variable,
                )
                ratio = None if base_power is None or exponent_ratio is None else base_power * exponent_ratio
            if ratio is None:
                break
            mapping[node] = auxiliary if ratio == 1 else _pow(auxiliary, ExactNumber(ratio))
        else:
            yield reference, mapping


def _scale_solution_multiplicity(solution_set, factor):
    if factor == 1 or isinstance(solution_set, EmptySolutionSet):
        return solution_set
    if isinstance(solution_set, FiniteSolutionSet):
        return FiniteSolutionSet(
            solution_set.values,
            tuple(multiplicity * factor for multiplicity in solution_set.multiplicities),
        )
    if isinstance(solution_set, UnionSolutionSet):
        return UnionSolutionSet(tuple(
            _scale_solution_multiplicity(part, factor) for part in solution_set.sets
        ))
    if isinstance(solution_set, ConditionalSolutionSet):
        return ConditionalSolutionSet(
            _scale_solution_multiplicity(solution_set.solution_set, factor),
            solution_set.conditions,
        )
    return solution_set


def _combine_substitution_solutions(parts):
    flattened = []
    for part in parts:
        if isinstance(part, EmptySolutionSet):
            continue
        flattened.extend(part.sets if isinstance(part, UnionSolutionSet) else (part,))
    finite, symbolic = {}, []
    for part in flattened:
        if isinstance(part, FiniteSolutionSet):
            for value, multiplicity in zip(part.values, part.multiplicities):
                finite[value] = finite.get(value, 0) + multiplicity
        elif part not in symbolic:
            symbolic.append(part)
    if finite:
        ordered = sorted(
            finite.items(),
            key=lambda pair: (
                complex(_numeric_value(pair[0]) or 0).real,
                complex(_numeric_value(pair[0]) or 0).imag,
                str(pair[0]),
            ),
        )
        symbolic.insert(0, FiniteSolutionSet(
            tuple(value for value, _ in ordered),
            tuple(multiplicity for _, multiplicity in ordered),
        ))
    return EMPTY if not symbolic else symbolic[0] if len(symbolic) == 1 else UnionSolutionSet(tuple(symbolic))


def _solution_branch_count(solution_set):
    if isinstance(solution_set, EmptySolutionSet):
        return 0
    if isinstance(solution_set, FiniteSolutionSet):
        return len(solution_set.values)
    if isinstance(solution_set, UnionSolutionSet):
        return sum(_solution_branch_count(part) for part in solution_set.sets)
    if isinstance(solution_set, ConditionalSolutionSet):
        return _solution_branch_count(solution_set.solution_set)
    return 1


def _solve_substituted_polynomial(state, context, kernel, replaced, auxiliary):
    try:
        parsed = _poly_fraction(replaced, auxiliary.name, max_degree=32)
    except UnsupportedExpressionError:
        return None
    if parsed is None or not parsed[0] or max(parsed[0], default=0) < 1:
        return None
    numerator, denominator = parsed
    outer_degree = max(numerator)
    occurrences = _expression_occurrences(_solver_residual(state), kernel)
    direct_inner_equation = (
        state.left == kernel and context.variable not in _variables(state.right)
        or state.right == kernel and context.variable not in _variables(state.left)
    )
    if outer_degree == 1 and direct_inner_equation:
        # The dedicated inverse rule must finish kernel = constant. Replacing
        # that equation by an identical temporary-variable solve would recurse.
        return None
    if (
        outer_degree < 2 and denominator == {0: Rational(1)} and occurrences < 2
        and not isinstance(kernel, SymbolicFunction)
    ):
        return None
    outer, complete = _solve_polynomial_exact(
        numerator, auxiliary.name, context.domain, None,
    )
    if not complete or not isinstance(outer, (FiniteSolutionSet, EmptySolutionSet)):
        return None
    if isinstance(outer, EmptySolutionSet):
        return _SolverRuleOutcome(EMPTY, explanation=f"Substitute {auxiliary} = {kernel} and solve the outer polynomial.")
    parts, conditions = [], AssumptionSet()
    for target, multiplicity in zip(outer.values, outer.multiplicities):
        numeric = _numeric_value(target)
        if denominator != {0: Rational(1)} and numeric is not None:
            if abs(complex(_poly_eval_numeric(denominator, numeric))) <= 1e-12:
                continue
        inner, inner_conditions, inner_complete, _ = _symbolic_dispatch(
            kernel, target, context.variable, context.domain, context.interval,
            lambda *args: None, context.assumptions, max_transformations=16,
        )
        if inner is None or not inner_complete:
            return None
        parts.append(_scale_solution_multiplicity(inner, multiplicity))
        if sum(_solution_branch_count(part) for part in parts) > 128:
            return None
        conditions = conditions.merge(inner_conditions)
    result = _combine_substitution_solutions(parts)
    return _SolverRuleOutcome(
        result, explanation=f"Substitute {auxiliary} = {kernel}, solve the exact outer polynomial, then solve every inner branch.",
        conditions=conditions,
    )


def _solve_symbolic_substitution_branches(solution_set, kernel, context):
    """Lift a conditional outer solution set through ``kernel = value``."""
    if isinstance(solution_set, EmptySolutionSet):
        return EMPTY, AssumptionSet(), True
    if isinstance(solution_set, UniversalSolutionSet):
        return UniversalSolutionSet(context.domain), AssumptionSet(), True
    if isinstance(solution_set, ConditionalSolutionSet):
        nested, conditions, complete = _solve_symbolic_substitution_branches(
            solution_set.solution_set, kernel, context,
        )
        return (
            _condition_solution_set(nested, solution_set.assumptions),
            conditions,
            complete,
        )
    if isinstance(solution_set, UnionSolutionSet):
        parts, conditions = [], AssumptionSet()
        for subset in solution_set.sets:
            nested, nested_conditions, complete = _solve_symbolic_substitution_branches(
                subset, kernel, context,
            )
            if not complete:
                return None, conditions, False
            parts.append(nested)
            conditions = conditions.merge(nested_conditions)
        return _combine_substitution_solutions(parts), conditions, True
    if not isinstance(solution_set, FiniteSolutionSet):
        return None, AssumptionSet(), False
    parts, conditions = [], AssumptionSet()
    for target, multiplicity in zip(
        solution_set.values, solution_set.multiplicities,
    ):
        inner, inner_conditions, complete, _ = _symbolic_dispatch(
            kernel, target, context.variable, context.domain, context.interval,
            lambda *args: None, context.assumptions,
            max_transformations=16,
        )
        if inner is None or not complete:
            return None, conditions, False
        parts.append(_scale_solution_multiplicity(inner, multiplicity))
        conditions = conditions.merge(inner_conditions)
        if sum(_solution_branch_count(part) for part in parts) > 128:
            return None, conditions, False
    return _combine_substitution_solutions(parts), conditions, True


def _solver_algebraic_substitution(state, context):
    residual = _solver_residual(state)
    auxiliary = _fresh_substitution_symbol(residual)

    # Parameter-aware biquadratics and related sparse quartics.  The ordinary
    # exact-coefficient path below remains preferable when all coefficients are
    # rational because it produces the most compact RootOf/multiplicity data.
    symbolic_polynomial = _symbolic_polynomial_coefficients(
        residual, context.variable, max_degree=4,
    )
    if symbolic_polynomial is not None:
        positive_degrees = tuple(
            degree for degree in symbolic_polynomial if degree > 0
        )
        common_degree = reduce(math.gcd, positive_degrees, 0)
        outer_degree = max(positive_degrees, default=0) // max(common_degree, 1)
        symbolic_coefficients = set().union(*(
            _variables(coefficient) for coefficient in symbolic_polynomial.values()
        ))
        if common_degree > 1 and outer_degree == 2 and symbolic_coefficients:
            kernel = _pow(Symbol(context.variable), ExactNumber(common_degree))
            outer_expression = _add(*(
                coefficient if degree == 0 else _mul(
                    coefficient,
                    auxiliary if degree == common_degree else _pow(
                        auxiliary, ExactNumber(degree // common_degree),
                    ),
                )
                for degree, coefficient in symbolic_polynomial.items()
            ))
            outer, outer_conditions, complete, _ = _symbolic_dispatch(
                outer_expression, ZERO, auxiliary.name, context.domain, None,
                lambda *args: None, context.assumptions,
                max_transformations=16,
            )
            if outer is not None and complete:
                result, inner_conditions, inner_complete = (
                    _solve_symbolic_substitution_branches(outer, kernel, context)
                )
                if result is not None and inner_complete:
                    return _SolverRuleOutcome(
                        result,
                        explanation=(
                            f"Substitute {auxiliary} = {kernel}, solve the "
                            "parameter-aware outer quadratic, then solve every "
                            "conditional inner branch."
                        ),
                        conditions=outer_conditions.merge(inner_conditions),
                    )

    # Sparse polynomial powers: x^6 - 5*x^3 + 6 becomes u^2 - 5*u + 6.
    polynomial = _poly_fraction(residual, context.variable)
    if polynomial is not None and polynomial[1] == {0: Rational(1)}:
        positive_degrees = tuple(degree for degree in polynomial[0] if degree > 0)
        common_degree = reduce(math.gcd, positive_degrees, 0)
        outer_degree = max(positive_degrees, default=0) // max(common_degree, 1)
        if common_degree > 1 and outer_degree > 1:
            kernel = _pow(Symbol(context.variable), ExactNumber(common_degree))
            replaced = _polynomial_expression(
                {degree // common_degree: coefficient for degree, coefficient in polynomial[0].items()},
                auxiliary.name,
            )
            outcome = _solve_substituted_polynomial(
                state, context, kernel, replaced, auxiliary,
            )
            if outcome is not None:
                return outcome

    # Exact repeated subexpressions, including powers of one function call.
    if context.domain == "real":
        for kernel in _substitution_kernels(residual, context.variable):
            replaced = _replace_subexpressions(residual, {kernel: auxiliary})
            outcome = _solve_substituted_polynomial(
                state, context, kernel, replaced, auxiliary,
            )
            if outcome is not None:
                return outcome

        # Compatible exponential families need not share an identical tree:
        # exp(2*x) and exp(x), or 4^x and 2^x, map to powers of one u.
        for kernel, replacements in _family_substitution_maps(
            residual, context.variable, auxiliary,
        ):
            replaced = _replace_subexpressions(residual, replacements)
            outcome = _solve_substituted_polynomial(
                state, context, kernel, replaced, auxiliary,
            )
            if outcome is not None:
                return outcome
    return None


def _solver_clear_denominators_guard(state, outcome, context):
    denominator = _poly_fraction(_solver_residual(state), context.variable)[1]
    return RelationCondition(
        _polynomial_expression(denominator, context.variable), "!=", ZERO,
        f"{_poly_string(denominator, context.variable)} != 0",
    )


def _solver_polynomial(state, context):
    rational = _poly_fraction(_solver_residual(state), context.variable)
    if rational is None or rational[1] != {0: Rational(1)}:
        return None
    solution_set, complete = _solve_polynomial_exact(
        rational[0], context.variable, context.domain, context.interval,
    )
    return _SolverRuleOutcome(solution_set, complete)


def _symbolic_polynomial_coefficients(expression, variable, *, max_degree=4,
                                      max_terms=64):
    """Collect a bounded polynomial whose coefficients may be symbolic."""
    if variable not in _variables(expression):
        return {0: expression}
    if isinstance(expression, Symbol):
        return {1: ONE} if expression.name == variable else {0: expression}
    if isinstance(expression, Add):
        result = {}
        for term in expression.terms:
            parsed = _symbolic_polynomial_coefficients(
                term, variable, max_degree=max_degree, max_terms=max_terms,
            )
            if parsed is None:
                return None
            for degree, coefficient in parsed.items():
                result[degree] = _add(result.get(degree, ZERO), coefficient)
            result = {degree: value for degree, value in result.items() if value != ZERO}
            if len(result) > max_terms:
                return None
        return result
    if isinstance(expression, Multiply):
        result = {0: ONE}
        for factor in expression.factors:
            parsed = _symbolic_polynomial_coefficients(
                factor, variable, max_degree=max_degree, max_terms=max_terms,
            )
            if parsed is None:
                return None
            product_coefficients = {}
            for first_degree, first_coefficient in result.items():
                for second_degree, second_coefficient in parsed.items():
                    degree = first_degree + second_degree
                    if degree > max_degree:
                        return None
                    product_coefficients[degree] = _add(
                        product_coefficients.get(degree, ZERO),
                        _mul(first_coefficient, second_coefficient),
                    )
            result = {
                degree: value for degree, value in product_coefficients.items()
                if value != ZERO
            }
            if len(result) > max_terms:
                return None
        return result
    if (
        isinstance(expression, Power)
        and isinstance(expression.exponent, ExactNumber)
        and expression.exponent.denominator == 1
        and 0 <= expression.exponent.numerator <= max_degree
    ):
        parsed = _symbolic_polynomial_coefficients(
            expression.base, variable,
            max_degree=max_degree, max_terms=max_terms,
        )
        if parsed is None:
            return None
        result = {0: ONE}
        for _ in range(expression.exponent.numerator):
            next_result = {}
            for first_degree, first_coefficient in result.items():
                for second_degree, second_coefficient in parsed.items():
                    degree = first_degree + second_degree
                    if degree > max_degree:
                        return None
                    next_result[degree] = _add(
                        next_result.get(degree, ZERO),
                        _mul(first_coefficient, second_coefficient),
                    )
            result = {
                degree: value for degree, value in next_result.items()
                if value != ZERO
            }
            if len(result) > max_terms:
                return None
        return result
    return None


def _condition_solution_set(solution_set, conditions):
    conditions = _coerce_assumption_set(conditions)
    if isinstance(solution_set, EmptySolutionSet) or not conditions:
        return solution_set
    if isinstance(solution_set, UnionSolutionSet):
        parts = tuple(
            _condition_solution_set(part, conditions) for part in solution_set.sets
        )
        parts = tuple(part for part in parts if not isinstance(part, EmptySolutionSet))
        return EMPTY if not parts else parts[0] if len(parts) == 1 else UnionSolutionSet(parts)
    if isinstance(solution_set, ConditionalSolutionSet):
        combined = conditions.merge(solution_set.assumptions)
        return ConditionalSolutionSet(solution_set.solution_set, combined.conditions)
    return ConditionalSolutionSet(solution_set, conditions.conditions)


def _symbolic_linear_solution_set(coefficient, constant, domain):
    coefficient_value = _numeric_value(coefficient)
    constant_value = _numeric_value(constant)
    if coefficient_value is not None:
        if coefficient_value != 0:
            return FiniteSolutionSet((
                _mul(_neg(constant), _pow(coefficient, NEG_ONE)),
            ))
        if constant_value is not None:
            return UniversalSolutionSet(domain) if constant_value == 0 else EMPTY
    root = _mul(_neg(constant), _pow(coefficient, NEG_ONE))
    return UnionSolutionSet((
        ConditionalSolutionSet(
            FiniteSolutionSet((root,)),
            (RelationCondition(coefficient, "!=", ZERO),),
        ),
        ConditionalSolutionSet(
            UniversalSolutionSet(domain),
            (RelationCondition(coefficient, "=", ZERO),
             RelationCondition(constant, "=", ZERO)),
        ),
    ))


def _symbolic_quadratic_nonzero_leading(a, b, c, domain):
    discriminant = _add(
        _pow(b, ExactNumber(2)),
        _neg(_mul(ExactNumber(4), a, c)),
    )
    center = _mul(_neg(b), _pow(_mul(ExactNumber(2), a), NEG_ONE))
    radical_term = _mul(
        _function("sqrt", discriminant),
        _pow(_mul(ExactNumber(2), a), NEG_ONE),
    )
    distinct = FiniteSolutionSet((
        _add(center, radical_term),
        _add(center, _neg(radical_term)),
    ))
    repeated = FiniteSolutionSet((center,), (2,))
    discriminant_value = _numeric_value(discriminant)
    if discriminant_value is not None:
        discriminant_value = complex(discriminant_value)
        if domain == "real":
            if abs(discriminant_value.imag) > 1e-12 or discriminant_value.real < 0:
                return EMPTY
            return repeated if abs(discriminant_value.real) <= 1e-12 else distinct
        return repeated if abs(discriminant_value) <= 1e-12 else distinct
    if domain == "real":
        return UnionSolutionSet((
            ConditionalSolutionSet(
                distinct, (RelationCondition(discriminant, ">", ZERO),),
            ),
            ConditionalSolutionSet(
                repeated, (RelationCondition(discriminant, "=", ZERO),),
            ),
        ))
    return UnionSolutionSet((
        ConditionalSolutionSet(
            distinct, (RelationCondition(discriminant, "!=", ZERO),),
        ),
        ConditionalSolutionSet(
            repeated, (RelationCondition(discriminant, "=", ZERO),),
        ),
    ))


def _solver_symbolic_quadratic(state, context):
    polynomial = _symbolic_polynomial_coefficients(
        _solver_residual(state), context.variable, max_degree=2,
    )
    if polynomial is None or max(polynomial, default=0) != 2:
        return None
    a = polynomial[2]
    b = polynomial.get(1, ZERO)
    c = polynomial.get(0, ZERO)
    if not (_variables(a) | _variables(b) | _variables(c)):
        return None  # Exact numeric coefficients belong to solve_polynomial.
    leading_value = _numeric_value(a)
    if leading_value is not None:
        if leading_value == 0:
            solution_set = _symbolic_linear_solution_set(b, c, context.domain)
        else:
            solution_set = _symbolic_quadratic_nonzero_leading(
                a, b, c, context.domain,
            )
    else:
        quadratic = _condition_solution_set(
            _symbolic_quadratic_nonzero_leading(a, b, c, context.domain),
            (RelationCondition(a, "!=", ZERO),),
        )
        degenerate = _condition_solution_set(
            _symbolic_linear_solution_set(b, c, context.domain),
            (RelationCondition(a, "=", ZERO),),
        )
        solution_set = _combine_substitution_solutions((quadratic, degenerate))
    return _SolverRuleOutcome(
        solution_set,
        explanation=(
            "Apply the exact symbolic quadratic formula and retain the "
            "leading-coefficient, discriminant, and degenerate linear branches."
        ),
    )


def _solver_zero_product(state, context):
    residual = _solver_residual(state)
    if not isinstance(residual, Multiply):
        return None
    variable_factors = [
        factor for factor in residual.factors
        if context.variable in _variables(factor)
    ]
    parameter_factors = [
        factor for factor in residual.factors
        if context.variable not in _variables(factor)
        and not isinstance(factor, ExactNumber)
    ]
    if not variable_factors or len(variable_factors) + len(parameter_factors) < 2:
        return None
    parts = []
    conditions = AssumptionSet()
    for factor in variable_factors:
        solution_set, factor_conditions, complete, _ = _symbolic_dispatch(
            factor, ZERO, context.variable, context.domain, context.interval,
            lambda *args: None, context.assumptions,
            max_transformations=16,
        )
        if solution_set is None or not complete:
            return None
        parts.append(solution_set)
        conditions = conditions.merge(factor_conditions)
    parts.extend(
        ConditionalSolutionSet(
            UniversalSolutionSet(context.domain),
            (RelationCondition(factor, "=", ZERO),),
        )
        for factor in parameter_factors
    )
    return _SolverRuleOutcome(
        _combine_substitution_solutions(parts), conditions=conditions,
        explanation=(
            "Apply the zero-product property to every factor and retain "
            "parameter-only zero branches."
        ),
    )


def _solver_symbolic_linear(state, context):
    affine = _affine_symbolic(_solver_residual(state), context.variable)
    if affine is None or affine[0] == ZERO:
        return None
    coefficient, constant = affine
    root = _mul(_neg(constant), _pow(coefficient, NEG_ONE))
    coefficient_value = _numeric_value(coefficient)
    if coefficient_value is not None and coefficient_value != 0:
        result = FiniteSolutionSet((root,)) if _in_interval(root, context.interval) else EMPTY
        return _SolverRuleOutcome(
            result, explanation="Divide by the known nonzero symbolic coefficient.",
        )
    branches = UnionSolutionSet((
        ConditionalSolutionSet(
            FiniteSolutionSet((root,)),
            (RelationCondition(coefficient, "!=", ZERO),),
        ),
        ConditionalSolutionSet(
            UniversalSolutionSet(context.domain),
            (RelationCondition(coefficient, "=", ZERO),
             RelationCondition(constant, "=", ZERO)),
        ),
    ))
    return _SolverRuleOutcome(
        branches,
        explanation=(
            "Divide by a symbolic coefficient only on its nonzero branch; "
            "retain the identity branch."
        ),
    )


def _solver_combine_logarithms(state, context):
    left_logs, right_logs = _log_terms(state.left), _log_terms(state.right)
    if not left_logs or not right_logs or left_logs[0] != right_logs[0]:
        return None
    left_argument = _mul(*left_logs[1])
    right_argument = _mul(*right_logs[1])
    if _poly_fraction(_add(left_argument, _neg(right_argument)), context.variable) is None:
        return None
    return _SolverRuleOutcome(EquationState(left_argument, right_argument))


def _solver_combine_logarithms_guard(state, outcome, context):
    left_logs, right_logs = _log_terms(state.left), _log_terms(state.right)
    return AssumptionSet(tuple(
        _render_condition(argument, "> 0", context.variable)
        for argument in left_logs[1] + right_logs[1]
    ))


def _trigonometric_nodes(expression):
    nodes = []

    def visit(value):
        if (
            isinstance(value, SymbolicFunction)
            and value.name in {"sin", "cos"}
            and value not in nodes
        ):
            nodes.append(value)
        for child in _rewrite_children(value):
            visit(child)

    visit(expression)
    return tuple(sorted(nodes, key=lambda value: (str(value.arguments[-1]), value.name)))


def _replace_even_trigonometric_powers(expression, source, target):
    if (
        isinstance(expression, Power) and expression.base == source
        and isinstance(expression.exponent, ExactNumber)
        and expression.exponent.denominator == 1
        and expression.exponent.numerator > 0
        and expression.exponent.numerator % 2 == 0
    ):
        identity = _add(ONE, _neg(_pow(target, ExactNumber(2))))
        return _pow(identity, ExactNumber(expression.exponent.numerator // 2))
    if isinstance(expression, Add):
        return _add(*(
            _replace_even_trigonometric_powers(term, source, target)
            for term in expression.terms
        ))
    if isinstance(expression, Multiply):
        return _mul(*(
            _replace_even_trigonometric_powers(factor, source, target)
            for factor in expression.factors
        ))
    if isinstance(expression, Power):
        return _pow(
            _replace_even_trigonometric_powers(expression.base, source, target),
            _replace_even_trigonometric_powers(expression.exponent, source, target),
        )
    if isinstance(expression, SymbolicFunction):
        return _function(expression.name, *(
            _replace_even_trigonometric_powers(argument, source, target)
            for argument in expression.arguments
        ))
    return expression


def _factor_common_symbolic_term(expression):
    if not isinstance(expression, Add) or len(expression.terms) < 2:
        return expression

    def factors(term):
        return term.factors if isinstance(term, Multiply) else (term,)

    candidates = [
        factor for factor in factors(expression.terms[0])
        if not isinstance(factor, ExactNumber)
    ]
    common = next((
        candidate for candidate in candidates
        if all(candidate in factors(term) for term in expression.terms[1:])
    ), None)
    if common is None:
        return expression
    remainders = []
    for term in expression.terms:
        values = list(factors(term))
        values.remove(common)
        remainders.append(_mul(*values))
    return _mul(common, _add(*remainders))


def _multiple_angle_replacement(node, reference, ratio):
    argument = reference.arguments[-1]
    sine = _function("sin", argument)
    cosine = _function("cos", argument)
    if ratio == 2 and node.name == "sin":
        return _mul(ExactNumber(2), sine, cosine)
    if ratio == 2 and node.name == "cos":
        return (
            _add(ONE, _neg(_mul(ExactNumber(2), _pow(sine, ExactNumber(2)))))
            if reference.name == "sin"
            else _add(_mul(ExactNumber(2), _pow(cosine, ExactNumber(2))), NEG_ONE)
        )
    if ratio == 3 and node.name == "sin":
        return _add(
            _mul(ExactNumber(3), sine),
            _neg(_mul(ExactNumber(4), _pow(sine, ExactNumber(3)))),
        )
    if ratio == 3 and node.name == "cos":
        return _add(
            _mul(ExactNumber(4), _pow(cosine, ExactNumber(3))),
            _neg(_mul(ExactNumber(3), cosine)),
        )
    return None


def _solver_trigonometric_reduction(state, context):
    residual = _solver_residual(state)
    nodes = _trigonometric_nodes(residual)

    # Prefer reducing a higher angle to a function already present.  This
    # strictly lowers the largest affine angle multiplier, preventing cycles.
    for node in nodes:
        for reference in nodes:
            if node == reference:
                continue
            ratio = _integer_affine_ratio(
                node.arguments[-1], reference.arguments[-1], context.variable,
            )
            if ratio not in {2, 3} or ratio == 3 and node.name != reference.name:
                continue
            replacement = _multiple_angle_replacement(node, reference, ratio)
            if replacement is None:
                continue
            reduced = _replace_subexpressions(residual, {node: replacement})
            reduced = _factor_common_symbolic_term(reduced)
            if reduced != residual and _rewrite_node_count(reduced) <= 512:
                return _SolverRuleOutcome(
                    EquationState(reduced, ZERO),
                    explanation=(
                        f"Reduce the {ratio}-angle {node.name} expression to "
                        "the existing base angle and factor any common term."
                    ),
                )

    # If one of sin(u), cos(u) occurs only through even powers, replace it by
    # 1 minus the square of the other.  Accept only candidates that become a
    # univariate rational polynomial in the retained trig function.
    by_argument = {}
    for node in nodes:
        by_argument.setdefault(node.arguments[-1], {})[node.name] = node
    auxiliary = _fresh_substitution_symbol(residual)
    for pair in by_argument.values():
        if set(pair) != {"sin", "cos"}:
            continue
        for source_name, target_name in (("sin", "cos"), ("cos", "sin")):
            candidate = _replace_even_trigonometric_powers(
                residual, pair[source_name], pair[target_name],
            )
            if candidate == residual:
                continue
            replaced = _replace_subexpressions(
                candidate, {pair[target_name]: auxiliary},
            )
            try:
                polynomial = _poly_fraction(replaced, auxiliary.name, max_degree=16)
            except UnsupportedExpressionError:
                polynomial = None
            if polynomial is None or polynomial[1] != {0: Rational(1)}:
                continue
            candidate = _factor_common_symbolic_term(candidate)
            return _SolverRuleOutcome(
                EquationState(candidate, ZERO),
                explanation=(
                    f"Use {source_name}(u)^2 + {target_name}(u)^2 = 1 "
                    f"to reduce the equation to a polynomial in {target_name}(u)."
                ),
            )
    return None


def _solver_invert_exponential(state, context):
    function_side = _same_function_side(state.left, state.right)
    if not function_side:
        return None
    function, constant = function_side
    if function.name != "exp" or _variables(constant):
        return None
    argument = function.arguments[-1]
    affine = _affine_symbolic(argument, context.variable)
    if affine is None or affine[0] == ZERO:
        return None
    numeric = _numeric_value(constant)
    if numeric is not None and (isinstance(numeric, complex) or numeric <= 0):
        return _SolverRuleOutcome(EMPTY)
    return _SolverRuleOutcome(EquationState(argument, _function("ln", constant)))


def _solver_invert_exponential_guard(state, outcome, context):
    if outcome.terminal:
        return True
    function_side = _same_function_side(state.left, state.right)
    if function_side and function_side[0].name == "exp":
        return RelationCondition(function_side[1], ">", ZERO)
    return True


def _solver_split_absolute(state, context):
    function_side = _same_function_side(state.left, state.right)
    if not function_side:
        return None
    function, constant = function_side
    if function.name != "abs" or not isinstance(constant, ExactNumber):
        return None
    if constant.value < 0:
        return _SolverRuleOutcome(EMPTY)
    roots = []
    for target in (constant, ExactNumber(-constant.value)):
        root = _solve_affine_equal(function.arguments[-1], target, context.variable)
        if root is not None and root not in roots:
            roots.append(root)
    return _SolverRuleOutcome(FiniteSolutionSet(tuple(roots))) if roots else None


def _solver_isolate_radical(state, context):
    function_side = _same_function_side(state.left, state.right)
    if not function_side:
        return None
    function, constant = function_side
    if function.name != "sqrt":
        return None
    if (
        context.domain == "real" and isinstance(constant, ExactNumber)
        and constant.value < 0
    ):
        return _SolverRuleOutcome(EMPTY)
    try:
        squared_target = _expand_square(constant)
    except UnsupportedExpressionError:
        return None
    if _rewrite_node_count(squared_target) > 512:
        return None
    return _SolverRuleOutcome(EquationState(
        function.arguments[-1], squared_target,
    ))


def _solver_isolate_radical_guard(state, outcome, context):
    if outcome.terminal or context.domain != "real":
        return True
    _, constant = _same_function_side(state.left, state.right)
    return RelationCondition(constant, ">=", ZERO)


def _square_factor(expression):
    if isinstance(expression, SymbolicFunction) and expression.name == "sqrt":
        return expression.arguments[-1]
    if isinstance(expression, Multiply):
        return _mul(*(_expand_square(factor) for factor in expression.factors))
    return _pow(expression, ExactNumber(2))


def _expand_square(expression):
    """Expand a bounded square while reducing principal ``sqrt(u)^2``."""
    if not isinstance(expression, Add):
        return _square_factor(expression)
    terms = expression.terms
    expanded = [_square_factor(term) for term in terms]
    for first in range(len(terms)):
        for second in range(first + 1, len(terms)):
            expanded.append(_mul(ExactNumber(2), terms[first], terms[second]))
            if len(expanded) > 64:
                raise UnsupportedExpressionError(
                    "Radical square expansion exceeds the 64-term limit"
                )
    return _add(*expanded)


def _expand_distributive(expression, *, max_terms=64):
    """Expand only additive products needed to expose radical terms."""
    if isinstance(expression, Add):
        return _add(*(
            _expand_distributive(term, max_terms=max_terms)
            for term in expression.terms
        ))
    if not isinstance(expression, Multiply):
        return expression
    terms = [ONE]
    for factor in expression.factors:
        factor = _expand_distributive(factor, max_terms=max_terms)
        choices = factor.terms if isinstance(factor, Add) else (factor,)
        if len(terms) * len(choices) > max_terms:
            raise UnsupportedExpressionError(
                f"Radical expansion exceeds the {max_terms}-term limit"
            )
        terms = [_mul(term, choice) for term in terms for choice in choices]
    return _add(*terms)


def _additive_radical_term(term, variable):
    if isinstance(term, SymbolicFunction) and term.name == "sqrt":
        return (term, ONE) if variable in _variables(term) else None
    if not isinstance(term, Multiply):
        return None
    radicals = [
        factor for factor in term.factors
        if isinstance(factor, SymbolicFunction) and factor.name == "sqrt"
    ]
    if len(radicals) != 1:
        return None
    radical = radicals[0]
    if variable not in _variables(radical):
        return None
    coefficient_factors = tuple(
        factor for factor in term.factors if factor is not radical
    )
    coefficient = _mul(*coefficient_factors)
    coefficient_value = _numeric_value(coefficient)
    if (
        variable in _variables(coefficient)
        or coefficient_value is None or coefficient_value == 0
        or any(
            isinstance(factor, SymbolicFunction) and factor.name == "sqrt"
            for factor in coefficient_factors
        )
    ):
        return None
    return radical, coefficient


def _solver_isolate_additive_radical(state, context):
    try:
        residual = _expand_distributive(_solver_residual(state))
    except UnsupportedExpressionError:
        return None
    terms = residual.terms if isinstance(residual, Add) else (residual,)
    for index, term in enumerate(terms):
        parsed = _additive_radical_term(term, context.variable)
        if parsed is None:
            continue
        radical, coefficient = parsed
        rest = _add(*(value for position, value in enumerate(terms) if position != index))
        target = _mul(_neg(rest), _pow(coefficient, NEG_ONE))
        candidate = EquationState(radical, target)
        if candidate == state or _rewrite_node_count(target) > 512:
            continue
        return _SolverRuleOutcome(
            candidate,
            explanation=(
                "Isolate one additive principal square root before guarded "
                "squaring; repeat until an algebraic equation remains."
            ),
        )
    return None


def _solver_invert_logarithm(state, context):
    function_side = _same_function_side(state.left, state.right)
    if not function_side:
        return None
    function, constant = function_side
    if function.name not in {"ln", "log"} or len(function.arguments) not in {1, 2}:
        return None
    argument = function.arguments[-1]
    affine = _affine(argument, context.variable)
    if not affine or not affine[0] or _variables(constant):
        return None
    inverse = (
        _function("exp", constant)
        if function.name == "ln" or len(function.arguments) == 1
        else _pow(function.arguments[0], constant)
    )
    return _SolverRuleOutcome(EquationState(argument, inverse))


def _solver_invert_logarithm_guard(state, outcome, context):
    function, _ = _same_function_side(state.left, state.right)
    return _render_condition(function.arguments[-1], "> 0", context.variable)


def _solver_invert_trigonometric(state, context):
    function_side = _same_function_side(state.left, state.right)
    if not function_side:
        return None
    function, constant = function_side
    if function.name not in {"sin", "cos", "tan"} or _variables(constant):
        return None
    argument = function.arguments[-1]
    affine, phase = _affine(argument, context.variable), _trig_phase(function.name, constant)
    numeric = _numeric_value(constant)
    if (
        function.name in {"sin", "cos"} and numeric is not None
        and (isinstance(numeric, complex) or not -1 <= numeric <= 1)
    ):
        return _SolverRuleOutcome(EMPTY)
    if phase is None:
        phase = _function(
            {"sin": "asin", "cos": "acos", "tan": "atan"}[function.name],
            constant,
        )
    if not affine or not affine[0] or phase is None:
        return None
    a, b = affine
    if function.name == "sin":
        if constant == ZERO:
            families = _parameter_family(context.variable, a, b, ZERO, PI)
        elif constant in {NEG_ONE, ONE}:
            families = _parameter_family(
                context.variable, a, b, phase, _mul(ExactNumber(2), PI),
            )
        else:
            families = UnionSolutionSet((
                _parameter_family(
                    context.variable, a, b, phase, _mul(ExactNumber(2), PI),
                ),
                _parameter_family(
                    context.variable, a, b, _add(PI, _neg(phase)),
                    _mul(ExactNumber(2), PI),
                ),
            ))
    elif function.name == "cos":
        if constant in {NEG_ONE, ONE}:
            families = _parameter_family(
                context.variable, a, b, phase, _mul(ExactNumber(2), PI),
            )
        else:
            families = UnionSolutionSet((
                _parameter_family(
                    context.variable, a, b, phase, _mul(ExactNumber(2), PI),
                ),
                _parameter_family(
                    context.variable, a, b, phase, _mul(ExactNumber(2), PI),
                    sign=-1,
                ),
            ))
    else:
        families = _parameter_family(context.variable, a, b, phase, PI)
    explanation = (
        "Return the complete integer-parameterized periodic family."
        if context.interval is None
        else "Enumerate the exact family over the requested interval."
    )
    return _SolverRuleOutcome(
        _filter_parametric(families, context.interval), explanation=explanation,
    )


def _solver_invert_power(state, context):
    power_side = (
        (state.left, state.right)
        if isinstance(state.left, Power) and not _variables(state.right)
        else (state.right, state.left)
        if isinstance(state.right, Power) and not _variables(state.left)
        else None
    )
    if not power_side:
        return None
    power, constant = power_side
    affine = _affine(power.exponent, context.variable)
    if not affine or not affine[0] or _variables(power.base) or _variables(constant):
        return None
    base_value, constant_value = _numeric_value(power.base), _numeric_value(constant)
    if (
        base_value is None or constant_value is None
        or isinstance(base_value, complex) or isinstance(constant_value, complex)
        or base_value <= 0
    ):
        return None
    if base_value == 1:
        return _SolverRuleOutcome(
            UniversalSolutionSet(context.domain) if constant_value == 1 else EMPTY,
        )
    if constant_value <= 0:
        return _SolverRuleOutcome(EMPTY)
    if (
        isinstance(constant, Power) and constant.base == power.base
        and not _variables(constant.exponent)
    ):
        target = constant.exponent
    elif power.base == constant:
        target = ONE
    else:
        target = _exact_log_ratio(power.base, constant) or _mul(
            _function("ln", constant), _pow(_function("ln", power.base), NEG_ONE),
        )
    return _SolverRuleOutcome(EquationState(power.exponent, target))


def _solver_invert_any_exponential(state, context):
    """Handle both ``exp(u)`` and constant-base ``a^u`` with one rule."""
    outcome = _solver_invert_exponential(state, context)
    return outcome if outcome is not None else _solver_invert_power(state, context)


_SOLVER_RULES = (
    _SolverRule(
        "clear_denominators", _solver_clear_denominators,
        _solver_clear_denominators_guard,
        "Clear denominators while retaining their exclusions.",
    ),
    _SolverRule(
        "trigonometric_reduction", _solver_trigonometric_reduction,
        explanation="Apply bounded Pythagorean or low multiple-angle identities.",
        domains=("real",),
    ),
    _SolverRule(
        "algebraic_substitution", _solver_algebraic_substitution,
        explanation="Solve a polynomial in a repeated algebraic subexpression.",
    ),
    _SolverRule(
        "solve_polynomial", _solver_polynomial,
        explanation="Solve the exact polynomial numerator.",
    ),
    _SolverRule(
        "solve_zero_product", _solver_zero_product,
        explanation="Apply the zero-product property to exact factors.",
    ),
    _SolverRule(
        "solve_symbolic_quadratic", _solver_symbolic_quadratic,
        explanation="Apply the exact parameter-aware quadratic formula.",
    ),
    _SolverRule("solve_symbolic_linear", _solver_symbolic_linear),
    _SolverRule(
        "combine_logarithms", _solver_combine_logarithms,
        _solver_combine_logarithms_guard,
        "Combine logarithms with the same base and retain every "
        "argument-domain restriction.",
        domains=("real",),
    ),
    _SolverRule(
        "invert_exponential", _solver_invert_any_exponential,
        _solver_invert_exponential_guard,
        "Apply logarithms to isolate the exponent and retain the required real-domain conditions.",
        domains=("real",),
    ),
    _SolverRule(
        "split_absolute", _solver_split_absolute,
        explanation="Split an absolute-value equation into its positive and negative branches.",
        domains=("real",),
    ),
    _SolverRule(
        "isolate_radical", _solver_isolate_radical,
        _solver_isolate_radical_guard,
        "Square the isolated principal radical; candidates will be checked "
        "in the original equation.",
        domains=("real",),
    ),
    _SolverRule(
        "isolate_additive_radical", _solver_isolate_additive_radical,
        explanation="Isolate one additive square-root term before guarded squaring.",
        domains=("real",),
    ),
    _SolverRule(
        "invert_logarithm", _solver_invert_logarithm,
        _solver_invert_logarithm_guard,
        "Apply the matching exponential function and preserve the logarithm domain.",
        domains=("real",),
    ),
    _SolverRule(
        "invert_trigonometric", _solver_invert_trigonometric,
        explanation="Return the complete integer-parameterized periodic family.",
        domains=("real",),
    ),
)


def _solver_rule_conditions(rule, state, outcome, context):
    if rule.guard is None:
        return outcome.conditions
    matched, required = _rewrite_guard_conditions(rule.guard(state, outcome, context))
    return required.merge(outcome.conditions) if matched else None


def _symbolic_dispatch(left, right, variable, domain, interval, trace,
                       assumptions=AssumptionSet(), max_transformations=32):
    """Run the ordered guarded solver-rule registry to a terminal result."""
    state = EquationState(left, right)
    normalized = EquationState(_solver_residual(state), ZERO)
    trace("normalize", state, normalized, "Move all terms to the left side.")
    accumulated = AssumptionSet()
    active = _coerce_assumption_set(assumptions)
    seen = {state}
    for _ in range(max_transformations):
        context = _SolverRuleContext(variable, domain, interval, active)
        applied = False
        for rule in _SOLVER_RULES:
            if domain not in rule.domains:
                continue
            outcome = rule.transform(state, context)
            if outcome is None:
                continue
            required = _solver_rule_conditions(rule, state, outcome, context)
            if required is None:
                continue
            if any(active.refutes(condition) for condition in required):
                return EMPTY, accumulated, True, "symbolic"
            accumulated = accumulated.merge(required)
            active = active.merge(required)
            explanation = outcome.explanation or rule.explanation
            trace(rule.identifier, state, outcome.after, explanation, required)
            if outcome.terminal:
                return outcome.after, accumulated, outcome.complete, outcome.method
            state = outcome.after
            if state in seen:
                return None, accumulated, False, "symbolic"
            seen.add(state)
            applied = True
            break
        if not applied:
            return None, accumulated, False, "symbolic"
    return None, accumulated, False, "symbolic"


def _poly_string(polynomial, variable):
    terms = []
    for degree in sorted(polynomial, reverse=True):
        coefficient = polynomial[degree]
        if degree == 0:
            terms.append(str(ExactNumber(coefficient)))
        else:
            symbol = variable if degree == 1 else f"{variable}^{degree}"
            terms.append(symbol if coefficient == 1 else f"-{symbol}" if coefficient == -1 else f"{ExactNumber(coefficient)}*{symbol}")
    return " + ".join(terms).replace("+ -", "- ") or "0"


def _polynomial_expression(polynomial, variable):
    """Build a structural expression from an exact polynomial mapping."""
    symbol = Symbol(variable)
    terms = []
    for degree, coefficient in polynomial.items():
        coefficient = ExactNumber(coefficient)
        factor = ONE if degree == 0 else symbol if degree == 1 else _pow(symbol, ExactNumber(degree))
        terms.append(coefficient if degree == 0 else factor if coefficient == ONE else _mul(coefficient, factor))
    return _add(*terms) if terms else ZERO


def _poly_eval_numeric(polynomial, value):
    result = 0j
    for coefficient in _coefficient_list(polynomial):
        result = result * value + complex(float(coefficient))
    return result


def _numeric_isolate(left, right, variable, interval, tolerance, max_iterations, assignments=None):
    """Adaptively isolate real roots over a caller-supplied finite interval.

    Crossing roots require a finite sign-change bracket.  Even-multiplicity
    roots require a sampled local minimum of ``abs(f)`` followed by bounded
    minimization.  These independent evidence intervals are retained through
    deduplication so nearby roots are not merged solely by distance.
    """
    lower, upper = interval
    evaluations = 0
    assignments = dict(assignments or {})
    cache = {}
    exhausted = False
    span = upper - lower
    max_depth = min(24, max(8, int(math.log2(max_iterations + 1)) + 6))
    max_evaluations = min(32769, max(257, 65 + max_iterations * 8))
    x_tolerance = max(
        tolerance * max(1.0, abs(lower), abs(upper)),
        np.finfo(float).eps * max(1.0, abs(lower), abs(upper)) * 16,
    )
    min_width = max(x_tolerance, span * (2.0 ** -max_depth))

    def evaluate(x):
        nonlocal evaluations, exhausted
        x = float(x)
        if x in cache:
            return cache[x]
        if evaluations >= max_evaluations:
            exhausted = True
            return math.nan
        evaluations += 1
        try:
            values = dict(assignments)
            values[variable] = x
            value = complex(_evaluate(left, values) - _evaluate(right, values))
            if abs(value.imag) > 1e-8 or not math.isfinite(value.real):
                result = math.nan
            else:
                result = value.real
        except (ArithmeticError, ValueError, OverflowError):
            result = math.nan
        cache[x] = result
        return result

    # A small deterministic seed mesh discovers broad features. Subdivision is
    # then concentrated around curvature, valleys, oscillation, and finite/
    # non-finite boundaries instead of scaling the entire mesh with iteration
    # count.
    seed_points = tuple(float(value) for value in np.linspace(lower, upper, 33))
    sampled = set(seed_points)
    seed_values = {point: evaluate(point) for point in seed_points}
    pending = [
        (seed_points[index], seed_points[index + 1],
         seed_values[seed_points[index]], seed_values[seed_points[index + 1]], 0)
        for index in range(len(seed_points) - 1)
    ]
    while pending and not exhausted:
        a, b, fa, fb, depth = pending.pop()
        if depth >= max_depth or b - a <= min_width:
            continue
        midpoint = (a + b) / 2
        fm = evaluate(midpoint)
        sampled.add(midpoint)
        finite = tuple(math.isfinite(value) for value in (fa, fm, fb))
        refine = False
        if not all(finite):
            # Refine only a transition boundary. A region that is wholly
            # undefined cannot contain an admissible real root.
            refine = any(finite) and not all(finite)
        else:
            absolute = tuple(abs(value) for value in (fa, fm, fb))
            scale = max(1.0, *absolute)
            curvature = abs(fm - (fa + fb) / 2) / scale
            valley = absolute[1] < 0.35 * min(absolute[0], absolute[2])
            left_slope = (fm - fa) / (midpoint - a)
            right_slope = (fb - fm) / (b - midpoint)
            turning = left_slope * right_slope <= 0 and absolute[1] < 0.8 * max(absolute[0], absolute[2])
            two_crossings = fa * fm <= 0 and fm * fb <= 0
            steep = max(absolute) > 1e8 * max(min(absolute), np.finfo(float).tiny)
            refine = curvature > 0.05 or valley or turning or two_crossings or steep
        if refine:
            pending.append((midpoint, b, fm, fb, depth + 1))
            pending.append((a, midpoint, fa, fm, depth + 1))

    points = sorted(sampled)
    candidates = []

    def accept(value, residual, bracket, kind, scale):
        if not math.isfinite(residual):
            return
        limit = max(tolerance * 10, 1e-12) * max(1.0, scale)
        if residual <= limit:
            candidates.append((float(value), float(residual), bracket, kind))

    def refine_bracket(a, b, fa, fb):
        best_x, best_value = (a, fa) if abs(fa) <= abs(fb) else (b, fb)
        scale = max(1.0, abs(fa), abs(fb))
        for _ in range(max_iterations):
            if b - a <= x_tolerance:
                break
            midpoint = (a + b) / 2
            # Use a bracketed secant proposal when it is safely interior;
            # otherwise retain bisection's convergence guarantee.
            if fb != fa:
                proposal = b - fb * (b - a) / (fb - fa)
                margin = 0.1 * (b - a)
                x = proposal if a + margin < proposal < b - margin else midpoint
            else:
                x = midpoint
            fx = evaluate(x)
            if not math.isfinite(fx):
                x = midpoint
                fx = evaluate(x)
                if not math.isfinite(fx):
                    break
            if abs(fx) < abs(best_value):
                best_x, best_value = x, fx
            if fx == 0:
                best_x, best_value = x, fx
                break
            if fa * fx < 0:
                b, fb = x, fx
            elif fx * fb < 0:
                a, fa = x, fx
            else:  # pragma: no cover - protected by a strict sign bracket
                break
        accept(best_x, abs(best_value), (a, b), "crossing", scale)

    # Every strict sign-change interval supplies independent crossing evidence.
    for a, b in zip(points, points[1:]):
        fa, fb = evaluate(a), evaluate(b)
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            refine_bracket(a, b, fa, fb)

    # Exact sampled zeroes also cover roots located at interval boundaries.
    for index, point in enumerate(points):
        value = evaluate(point)
        if math.isfinite(value) and value == 0:
            a = points[max(0, index - 1)]
            b = points[min(len(points) - 1, index + 1)]
            candidates.append((point, 0.0, (a, b), "sampled"))

    # Floating evaluation rarely returns an exact zero for transcendental
    # roots at a boundary (for example sin(20*pi)). Accept a small endpoint
    # only when a one-sided secant predicts the root at that same boundary;
    # this rejects merely small tails such as exp(x) far to the left.
    for endpoint_index, neighbor_index in ((0, 1), (-1, -2)):
        endpoint, neighbor = points[endpoint_index], points[neighbor_index]
        endpoint_value, neighbor_value = evaluate(endpoint), evaluate(neighbor)
        if not all(math.isfinite(value) for value in (endpoint_value, neighbor_value)):
            continue
        slope = (neighbor_value - endpoint_value) / (neighbor - endpoint)
        scale = max(1.0, abs(endpoint_value), abs(neighbor_value))
        if slope and abs(endpoint_value) <= max(tolerance * 10, 1e-12) * scale:
            projected = endpoint - endpoint_value / slope
            if abs(projected - endpoint) <= x_tolerance * 4:
                candidates.append((
                    endpoint, abs(endpoint_value),
                    (min(endpoint, neighbor), max(endpoint, neighbor)), "boundary",
                ))

    def refine_minimum(a, b, center):
        # Golden-section minimization of |f| is safeguarded inside the sampled
        # valley. It finds tangent roots without assuming a usable derivative.
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
        f1, f2 = abs(evaluate(x1)), abs(evaluate(x2))
        center_value = abs(evaluate(center))
        best_x, best_value = center, center_value
        scale = max(1.0, abs(evaluate(a)), center_value, abs(evaluate(b)))
        for _ in range(min(max_iterations, 80)):
            if b - a <= x_tolerance or exhausted:
                break
            if math.isfinite(f1) and f1 < best_value:
                best_x, best_value = x1, f1
            if math.isfinite(f2) and f2 < best_value:
                best_x, best_value = x2, f2
            if not math.isfinite(f1) and not math.isfinite(f2):
                break
            if not math.isfinite(f2) or math.isfinite(f1) and f1 <= f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - ratio * (b - a)
                f1 = abs(evaluate(x1))
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + ratio * (b - a)
                f2 = abs(evaluate(x2))
        accept(best_x, best_value, (a, b), "tangent", scale)

    # Local minima of |f| are the evidence required for an even-multiplicity
    # root. Strictness prevents flat nonzero plateaus from spawning candidates.
    for index in range(1, len(points) - 1):
        a, center, b = points[index - 1:index + 2]
        values = tuple(evaluate(point) for point in (a, center, b))
        if not all(math.isfinite(value) for value in values):
            continue
        absolute = tuple(abs(value) for value in values)
        if (
            absolute[1] <= absolute[0] and absolute[1] <= absolute[2]
            and (absolute[1] < absolute[0] or absolute[1] < absolute[2])
        ):
            refine_minimum(a, b, center)

    # Merge only candidates whose evidence intervals overlap and whose refined
    # locations agree. Disjoint brackets remain distinct even at small scales.
    candidates.sort(key=lambda item: (item[0], item[2][0], item[2][1], item[3]))
    unique = []
    for candidate in candidates:
        value, residual, bracket, _ = candidate
        duplicate = None
        for index, existing in enumerate(unique):
            existing_value, _, existing_bracket, _ = existing
            overlaps = max(bracket[0], existing_bracket[0]) <= min(bracket[1], existing_bracket[1])
            close = abs(value - existing_value) <= max(
                tolerance * 16 * max(1.0, abs(value), abs(existing_value)),
                np.finfo(float).eps * 64 * max(1.0, abs(value), abs(existing_value)),
            )
            if overlaps and close:
                duplicate = index
                break
        if duplicate is None:
            unique.append(candidate)
        elif residual < unique[duplicate][1]:
            unique[duplicate] = candidate
    unique.sort(key=lambda item: item[0])
    roots = tuple(item[0] for item in unique)
    residuals = tuple(item[1] for item in unique)
    return FiniteSolutionSet(roots) if roots else EMPTY, residuals, evaluations


def _solve_equation_impl(equation, variable=None, *, domain="real", interval=None,
                         method="auto", numeric_fallback=True, tolerance=1e-10,
                         max_iterations=1000, steps=False, assumptions=None) -> EquationSolution:
    """Solve one equation with native symbolic rules and bounded fallback."""
    if domain not in {"real", "complex"}:
        raise ValueError("domain must be 'real' or 'complex'")
    if method not in {"auto", "symbolic", "numeric"}:
        raise ValueError("method must be 'auto', 'symbolic', or 'numeric'")
    if not isinstance(numeric_fallback, (bool, np.bool_)) or not isinstance(steps, (bool, np.bool_)):
        raise TypeError("numeric_fallback and steps must be booleans")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(tolerance, (int, float, np.number)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    interval = _validate_interval(interval)
    if domain == "complex" and interval is not None:
        raise ValueError("interval is a real domain constraint and cannot be used with domain='complex'")
    if method == "numeric" and interval is None:
        raise ValueError("numeric solving requires a finite interval")
    if method == "numeric" and domain != "real":
        raise ValueError("bounded numerical fallback currently supports the real domain")
    user_assumptions = _normalize_user_assumptions(assumptions)
    source_variables = _source_variables(equation)
    left, right = _equation_input(equation)
    variables = sorted(_variables(left) | _variables(right) | source_variables)
    if variable is None:
        if len(variables) != 1:
            raise AmbiguousVariableError(f"Specify variable= explicitly; found {variables or 'no variables'}")
        variable = variables[0]
    elif hasattr(variable, "name"):
        variable = variable.name
    if not isinstance(variable, str) or not variable:
        raise TypeError("variable must be a non-empty string or named variable")
    original_conditions = _merge_conditions(
        _domain_conditions(left, domain, variable),
        _domain_conditions(right, domain, variable),
        user_assumptions,
    )
    if original_conditions.contradictory:
        return EquationSolution(variable, EMPTY, "solved", "symbolic", True, True, conditions=original_conditions, message="The equation assumptions or original domain restrictions are contradictory.")
    parameter_substitutions = {
        name: value for name, value in original_conditions.substitutions.items()
        if name != variable
    }
    working_left = _substitute(left, parameter_substitutions)
    working_right = _substitute(right, parameter_substitutions)
    if variable not in variables:
        residual = _numeric_value(_add(left, _neg(right)))
        solution_set = UniversalSolutionSet(domain) if residual == 0 else EMPTY
        solution_set = _apply_interval(solution_set, interval)
        return EquationSolution(variable, solution_set, "solved", "symbolic", True, True, original_conditions, message="The equation is constant with respect to the target variable.")
    recorded = []

    def trace(rule, before, after, explanation, conditions=()):
        if steps:
            recorded.append(SolutionStep(rule, before, after, explanation, tuple(conditions)))

    solution_set = conditions = None
    complete = False
    if method != "numeric":
        solution_set, conditions, complete, used_method = _symbolic_dispatch(
            working_left, working_right, variable, domain, interval, trace,
            original_conditions,
        )
        if solution_set is not None:
            solution_set = _apply_interval(solution_set, interval)
            conditions = _merge_conditions(original_conditions, conditions)
            solution_set = _substitute_solution_set(solution_set, conditions.substitutions)
            solution_set = _apply_assumptions(solution_set, conditions)
            solution_set, residuals = _verify_finite(solution_set, left, right, variable, domain, tolerance, conditions)
            return EquationSolution(variable, solution_set, "solved", used_method, True, complete, conditions, residuals, tuple(recorded), "Exact symbolic solution.")
        if method == "symbolic" or not numeric_fallback or interval is None:
            return EquationSolution(variable, EMPTY, "unresolved", "symbolic", False, False, conditions=original_conditions, steps=tuple(recorded), message="No supported complete symbolic transformation was found." + (" Supply a finite interval to enable numerical fallback." if interval is None and numeric_fallback else ""))
    if interval is None:
        return EquationSolution(variable, EMPTY, "unresolved", "symbolic", False, False, conditions=original_conditions, steps=tuple(recorded), message="Numerical fallback requires a finite interval.")
    numeric_parameters = {
        name: (_numeric_value(value) if _numeric_value(value) is not None else value)
        for name, value in original_conditions.substitutions.items()
    }
    solution_set, residuals, evaluations = _numeric_isolate(
        left, right, variable, interval, float(tolerance), int(max_iterations),
        numeric_parameters,
    )
    solution_set, residuals = _verify_finite(
        solution_set, left, right, variable, domain, tolerance, original_conditions,
    )
    trace("numeric_isolation", f"{left} = {right}", str(solution_set), "Adaptively isolate and safeguard real roots over the requested finite interval.")
    return EquationSolution(variable, solution_set, "solved", "numeric" if method == "numeric" else "hybrid", False, False, conditions=original_conditions, residuals=residuals, steps=tuple(recorded), message="Approximate roots found by adaptive isolation over the requested interval; completeness is not guaranteed for arbitrary functions.", evaluations=evaluations)


def solve_equation(equation, variable=None, *, domain="real", interval=None,
                   method="auto", numeric_fallback=True, tolerance=1e-10,
                   max_iterations=1000, steps=False) -> EquationSolution:
    """Solve one equation while preserving the unified API contract."""
    return _solve_equation_impl(
        equation, variable, domain=domain, interval=interval, method=method,
        numeric_fallback=numeric_fallback, tolerance=tolerance,
        max_iterations=max_iterations, steps=steps,
    )


def solve_equation_assuming(equation, assumptions, variable=None, *, domain="real",
                            interval=None, method="auto", numeric_fallback=True,
                            tolerance=1e-10, max_iterations=1000,
                            steps=False) -> EquationSolution:
    """Solve an equation under explicit structural assumptions.

    ``assumptions`` accepts an :class:`AssumptionSet`, one condition, an
    iterable of conditions/atomic condition strings, or a mapping of symbols
    to exact values or domains.
    """
    return _solve_equation_impl(
        equation, variable, domain=domain, interval=interval, method=method,
        numeric_fallback=numeric_fallback, tolerance=tolerance,
        max_iterations=max_iterations, steps=steps, assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Exact real polynomial and rational inequalities


def _split_symbolic_inequality(inequality):
    if not isinstance(inequality, str):
        raise TypeError("inequality must be a string")
    matches = list(re.finditer(r"<=|>=|!=|<|>", inequality))
    if len(matches) != 1:
        raise EquationParseError(
            "An inequality must contain exactly one of <, <=, >, >=, or !="
        )
    match = matches[0]
    left_text, right_text = inequality[:match.start()], inequality[match.end():]
    if not left_text.strip() or not right_text.strip():
        raise EquationParseError("Both sides of an inequality must be non-empty")
    return parse_symbolic(left_text), match.group(), parse_symbolic(right_text)


def _real_polynomial_roots(polynomial, variable):
    if not polynomial or max(polynomial, default=0) == 0:
        return ()
    solution_set, complete = _solve_polynomial_exact(
        polynomial, variable, "real", None,
    )
    if not complete or not isinstance(solution_set, (FiniteSolutionSet, EmptySolutionSet)):
        raise UnsupportedExpressionError("Could not isolate every real critical point")
    if isinstance(solution_set, EmptySolutionSet):
        return ()
    return tuple(zip(solution_set.values, solution_set.multiplicities))


def _inequality_relation_holds(sign, operator):
    return {
        "<": sign < 0,
        "<=": sign <= 0,
        ">": sign > 0,
        ">=": sign >= 0,
        "!=": sign != 0,
    }[operator]


def _inequality_critical_points(numerator, denominator, source_denominator,
                                variable):
    points = {}

    def add(values, *, flip=False, zero=False, hole=False):
        for value, multiplicity in values:
            item = points.setdefault(value, {
                "value": value, "flip": 0, "zero": False, "hole": False,
            })
            if flip:
                item["flip"] += multiplicity
            item["zero"] = item["zero"] or zero
            item["hole"] = item["hole"] or hole

    add(_real_polynomial_roots(numerator, variable), flip=True, zero=True)
    add(_real_polynomial_roots(denominator, variable), flip=True, hole=True)
    add(_real_polynomial_roots(source_denominator, variable), hole=True)
    ordered = list(points.values())
    ordered.sort(key=lambda item: (
        float(_numeric_value(item["value"])), str(item["value"]),
    ))
    return ordered


def _inequality_solution_from_cells(points, region_signs, operator):
    cells = []
    for index in range(len(points) + 1):
        cells.append(_inequality_relation_holds(region_signs[index], operator))
        if index < len(points):
            point = points[index]
            cells.append(
                not point["hole"] and point["zero"]
                and operator in {"<=", ">="}
            )
    parts, index = [], 0
    while index < len(cells):
        if not cells[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(cells) and cells[index + 1]:
            index += 1
        end = index
        if start == end and start % 2 == 1:
            parts.append(FiniteSolutionSet((points[start // 2]["value"],)))
        else:
            if start == 0:
                lower, lower_closed = None, False
            elif start % 2:
                lower, lower_closed = points[start // 2]["value"], True
            else:
                lower, lower_closed = points[start // 2 - 1]["value"], False
            if end == len(cells) - 1:
                upper, upper_closed = None, False
            elif end % 2:
                upper, upper_closed = points[end // 2]["value"], True
            else:
                upper, upper_closed = points[end // 2]["value"], False
            parts.append(IntervalSolutionSet(
                lower, upper, lower_closed, upper_closed,
            ))
        index += 1
    return EMPTY if not parts else parts[0] if len(parts) == 1 else UnionSolutionSet(tuple(parts))


def solve_inequality(inequality, variable=None, *, domain="real",
                     assumptions=None, steps=False) -> EquationSolution:
    """Solve an exact univariate polynomial or rational inequality.

    The first release is intentionally real-only and requires exact rational
    coefficients after applying equality substitutions from ``assumptions``.
    Critical points and removable holes are retained exactly.
    """
    if domain != "real":
        raise ValueError("inequality solving currently supports only the real domain")
    if not isinstance(steps, (bool, np.bool_)):
        raise TypeError("steps must be boolean")
    left, operator, right = _split_symbolic_inequality(inequality)
    discovered = sorted(_variables(left) | _variables(right))
    if variable is None:
        if len(discovered) != 1:
            raise AmbiguousVariableError(
                f"Specify variable= explicitly; found {discovered or 'no variables'}"
            )
        variable = discovered[0]
    elif hasattr(variable, "name"):
        variable = variable.name
    if not isinstance(variable, str) or not variable:
        raise TypeError("variable must be a non-empty string or named variable")
    user_assumptions = _normalize_user_assumptions(assumptions)
    substitutions = {
        name: value for name, value in user_assumptions.substitutions.items()
        if name != variable
    }
    original_residual = _add(left, _neg(right))
    residual = _substitute(original_residual, substitutions)
    domain_conditions = _merge_conditions(
        _domain_conditions(left, domain, variable),
        _domain_conditions(right, domain, variable),
        user_assumptions,
    )
    if domain_conditions.contradictory:
        return EquationSolution(
            variable, EMPTY, "solved", "symbolic", True, True,
            conditions=domain_conditions,
            message="The inequality assumptions are contradictory.",
        )
    try:
        parsed = _poly_fraction(residual, variable, max_degree=100)
    except UnsupportedExpressionError:
        parsed = None
    if parsed is None:
        return EquationSolution(
            variable, EMPTY, "unresolved", "symbolic", False, False,
            conditions=domain_conditions,
            message=(
                "Exact inequalities currently require a univariate polynomial "
                "or rational expression with rational coefficients."
            ),
        )
    numerator, source_denominator = map(_poly_clean, parsed)
    if not source_denominator:
        raise ZeroDivisionError("inequality has an identically zero denominator")
    common = _poly_gcd(numerator, source_denominator)
    denominator = source_denominator
    if common and max(common, default=0) > 0:
        numerator, numerator_remainder = _poly_divmod(numerator, common)
        denominator, denominator_remainder = _poly_divmod(source_denominator, common)
        if numerator_remainder or denominator_remainder:  # pragma: no cover
            raise ArithmeticError("polynomial GCD did not divide exactly")
    points = _inequality_critical_points(
        numerator, denominator, source_denominator, variable,
    )
    if numerator:
        numerator_leading = numerator[max(numerator)]
        denominator_leading = denominator[max(denominator)]
        right_sign = 1 if numerator_leading * denominator_leading > 0 else -1
    else:
        right_sign = 0
    region_signs = [0] * (len(points) + 1)
    region_signs[-1] = right_sign
    for index in range(len(points) - 1, -1, -1):
        region_signs[index] = (
            -region_signs[index + 1]
            if points[index]["flip"] % 2 else region_signs[index + 1]
        )
    solution_set = _inequality_solution_from_cells(
        points, region_signs, operator,
    )
    solution_set = _apply_assumptions(solution_set, domain_conditions)
    recorded = ()
    if steps:
        normalized = f"{_polynomial_expression(numerator, variable)} / {_polynomial_expression(denominator, variable)} {operator} 0"
        recorded = (
            SolutionStep(
                "normalize_inequality", inequality, normalized,
                "Move all terms to the left, normalize the exact rational form, and retain denominator exclusions.",
            ),
            SolutionStep(
                "rational_sign_chart", normalized, solution_set,
                "Order every exact zero and pole, propagate signs by multiplicity, and assemble the satisfying intervals.",
            ),
        )
    return EquationSolution(
        variable, solution_set, "solved", "symbolic", True, True,
        conditions=domain_conditions, steps=recorded,
        message="Exact real polynomial/rational inequality sign chart.",
    )


# ---------------------------------------------------------------------------
# Exact linear systems and explicit local numerical fallback


@dataclass(frozen=True)
class EquationSystemSolution:
    variables: Tuple[str, ...]
    solutions: Tuple[Mapping[str, Any], ...]
    status: str
    method: str
    exact: bool
    complete: bool
    parameters: Tuple[str, ...] = ()
    residuals: Tuple[float, ...] = ()
    steps: Tuple[SolutionStep, ...] = ()
    message: str = ""

    def __post_init__(self):
        variables = tuple(self.variables)
        parameters = tuple(self.parameters)
        solutions = tuple(dict(item) for item in self.solutions)
        if not variables or any(not isinstance(name, str) or not name for name in variables) or len(set(variables)) != len(variables):
            raise ValueError("system variables must contain distinct nonempty names")
        if self.status not in {"solved", "unresolved", "inconsistent"}:
            raise ValueError("system solution status is invalid")
        if self.method not in {"symbolic", "numeric", "hybrid"}:
            raise ValueError("system solution method is invalid")
        if any(not isinstance(value, bool) for value in (self.exact, self.complete)):
            raise TypeError("exact and complete must be booleans")
        if any(set(item) != set(variables) or any(not _valid_solution_value(value) for value in item.values()) for item in solutions):
            raise ValueError("every solution mapping must contain one immutable value per variable")
        if any(not isinstance(name, str) or not name for name in parameters) or len(set(parameters)) != len(parameters):
            raise ValueError("system parameters must contain distinct nonempty names")
        residuals, recorded = tuple(self.residuals), tuple(self.steps)
        if any(isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or value < 0 for value in residuals):
            raise ValueError("system residuals must contain nonnegative finite values")
        if any(not isinstance(value, SolutionStep) for value in recorded):
            raise TypeError("system steps must contain SolutionStep values")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "solutions", tuple(MappingProxyType(item) for item in solutions))
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "steps", recorded)

    def to_dict(self):
        return {
            "variables": list(self.variables),
            "solutions": [{name: _encode_value(value) for name, value in item.items()} for item in self.solutions],
            "status": self.status, "method": self.method, "exact": self.exact,
            "complete": self.complete, "parameters": list(self.parameters),
            "residuals": list(self.residuals),
            "steps": [_step_to_dict(step) for step in self.steps],
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            tuple(data["variables"]),
            tuple({name: _decode_value(value) for name, value in item.items()} for item in data["solutions"]),
            data["status"], data["method"], data["exact"], data["complete"],
            tuple(data.get("parameters", ())), tuple(data.get("residuals", ())),
            tuple(_step_from_dict(item) for item in data.get("steps", ())),
            data.get("message", ""),
        )


def _linear_form(expression, variables):
    size = len(variables)
    if isinstance(expression, ExactNumber):
        return [Rational(0)] * size, expression.value
    if isinstance(expression, Symbol):
        if expression.name not in variables:
            return None
        coefficients = [Rational(0)] * size
        coefficients[variables.index(expression.name)] = Rational(1)
        return coefficients, Rational(0)
    if isinstance(expression, Add):
        coefficients, constant = [Rational(0)] * size, Rational(0)
        for term in expression.terms:
            parsed = _linear_form(term, variables)
            if parsed is None:
                return None
            coefficients = [first + second for first, second in zip(coefficients, parsed[0])]
            constant += parsed[1]
        return coefficients, constant
    if isinstance(expression, Multiply):
        scalar, nonconstant = Rational(1), None
        for factor in expression.factors:
            if isinstance(factor, ExactNumber):
                scalar *= factor.value
            elif nonconstant is None:
                nonconstant = factor
            else:
                return None
        if nonconstant is None:
            return [Rational(0)] * size, scalar
        parsed = _linear_form(nonconstant, variables)
        if parsed is None:
            return None
        return [scalar * value for value in parsed[0]], scalar * parsed[1]
    if isinstance(expression, Power) and expression.exponent == ONE:
        return _linear_form(expression.base, variables)
    return None


def _rref(matrix, columns):
    matrix = [list(row) for row in matrix]
    pivot_columns, row = [], 0
    for column in range(columns):
        pivot = next((index for index in range(row, len(matrix)) if matrix[index][column]), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        scale = matrix[row][column]
        matrix[row] = [value / scale for value in matrix[row]]
        for index in range(len(matrix)):
            if index != row and matrix[index][column]:
                factor = matrix[index][column]
                matrix[index] = [value - factor * pivot_value for value, pivot_value in zip(matrix[index], matrix[row])]
        pivot_columns.append(column)
        row += 1
        if row == len(matrix):
            break
    return matrix, pivot_columns


def _solve_triangular_system(sides, variables, domain, tolerance, max_branches=256):
    branches = []

    def recurse(current_sides, remaining, assignments):
        if len(branches) >= max_branches:
            raise UnsupportedExpressionError(f"Polynomial system exceeds the {max_branches}-branch limit")
        simplified = [(_substitute(left, assignments), _substitute(right, assignments)) for left, right in current_sides]
        active = []
        for left, right in simplified:
            residual = _add(left, _neg(right))
            involved = _variables(residual) & set(remaining)
            if not involved:
                value = _numeric_value(residual)
                # Once every system variable in an equation is assigned, a
                # non-evaluable constant denotes an undefined candidate (for
                # example division by zero), not an unresolved free branch.
                if value is None or abs(complex(value)) > tolerance:
                    return True
            else:
                active.append((left, right, involved))
        if not remaining:
            resolved = dict(assignments)
            for _ in range(len(resolved)):
                updated = {
                    name: _substitute(value, resolved)
                    if isinstance(value, SymbolicExpression) else value
                    for name, value in resolved.items()
                }
                if updated == resolved:
                    break
                resolved = updated
            branches.append(resolved)
            return True
        # Prefer an actually univariate equation.  If none exists, eliminate a
        # variable that occurs affinely.  A symbolic coefficient is safe when
        # the constant part is a known nonzero constant: the equation itself
        # then proves that the coefficient cannot vanish (for example xy=2).
        choice = next((
            ("univariate", left, right, next(name for name in remaining if name in involved), None)
            for left, right, involved in active if len(involved) == 1
        ), None)
        if choice is None:
            for left, right, involved in active:
                residual = _add(left, _neg(right))
                for variable in remaining:
                    if variable not in involved:
                        continue
                    affine = _affine_symbolic(residual, variable)
                    if affine is None or affine[0] == ZERO:
                        continue
                    coefficient_value = _numeric_value(affine[0])
                    constant_value = (
                        _numeric_value(affine[1]) if not _variables(affine[1]) else None
                    )
                    if coefficient_value not in {None, 0} or constant_value not in {None, 0}:
                        choice = ("affine", left, right, variable, affine)
                        break
                if choice is not None:
                    break
        if choice is None:
            return False
        kind, left, right, variable, affine = choice
        if kind == "univariate":
            result = solve_equation(
                (left, right), variable=variable, domain=domain,
                method="symbolic", tolerance=tolerance,
            )
            if not result.complete:
                return False
            if isinstance(result.solution_set, EmptySolutionSet):
                return True
            if not isinstance(result.solution_set, FiniteSolutionSet):
                return False
            values = result.solution_set.values
        else:
            coefficient, constant = affine
            values = (_mul(_neg(constant), _pow(coefficient, NEG_ONE)),)
        next_remaining = [name for name in remaining if name != variable]
        complete = True
        for value in values:
            numeric = _numeric_value(value)
            if (
                domain == "real" and numeric is not None
                and isinstance(numeric, complex) and abs(numeric.imag) > tolerance
            ):
                continue
            complete = recurse(
                simplified, next_remaining,
                dict(assignments, **{variable: value}),
            ) and complete
        return complete

    if not recurse(sides, list(variables), {}):
        return None
    return tuple({variable: branch[variable] for variable in variables} for branch in branches)


def _mv_poly_clean(polynomial):
    return {powers: coefficient for powers, coefficient in polynomial.items() if coefficient}


def _mv_poly_add(first, second, factor=Rational(1), *, max_terms=256):
    result = dict(first)
    for powers, coefficient in second.items():
        result[powers] = result.get(powers, Rational(0)) + factor * coefficient
    result = _mv_poly_clean(result)
    if len(result) > max_terms:
        raise UnsupportedExpressionError(
            f"Multivariate polynomial exceeds the {max_terms}-term limit"
        )
    return result


def _mv_poly_mul(first, second, *, max_degree=8, max_terms=256):
    result = {}
    for first_powers, first_coefficient in first.items():
        for second_powers, second_coefficient in second.items():
            powers = tuple(a + b for a, b in zip(first_powers, second_powers))
            if sum(powers) > max_degree:
                raise UnsupportedExpressionError(
                    f"Multivariate polynomial degree exceeds the {max_degree} limit"
                )
            result[powers] = (
                result.get(powers, Rational(0))
                + first_coefficient * second_coefficient
            )
    result = _mv_poly_clean(result)
    if len(result) > max_terms:
        raise UnsupportedExpressionError(
            f"Multivariate polynomial exceeds the {max_terms}-term limit"
        )
    return result


def _mv_poly_pow(polynomial, exponent, *, max_degree=8, max_terms=256):
    dimensions = len(next(iter(polynomial), (0, 0)))
    result = {(0,) * dimensions: Rational(1)}
    factor = polynomial
    while exponent:
        if exponent & 1:
            result = _mv_poly_mul(
                result, factor, max_degree=max_degree, max_terms=max_terms,
            )
        exponent //= 2
        if exponent:
            factor = _mv_poly_mul(
                factor, factor, max_degree=max_degree, max_terms=max_terms,
            )
    return result


def _multivariate_polynomial(expression, variables, *, max_degree=8, max_terms=256):
    """Return a bounded exact multivariate polynomial or ``None``."""
    dimensions = len(variables)
    zero_powers = (0,) * dimensions
    if isinstance(expression, ExactNumber):
        return {zero_powers: expression.value} if expression.value else {}
    if isinstance(expression, Symbol):
        if expression.name not in variables:
            return None
        powers = [0] * dimensions
        powers[variables.index(expression.name)] = 1
        return {tuple(powers): Rational(1)}
    if isinstance(expression, Add):
        result = {}
        for term in expression.terms:
            parsed = _multivariate_polynomial(
                term, variables, max_degree=max_degree, max_terms=max_terms,
            )
            if parsed is None:
                return None
            result = _mv_poly_add(result, parsed, max_terms=max_terms)
        return result
    if isinstance(expression, Multiply):
        result = {zero_powers: Rational(1)}
        for factor in expression.factors:
            parsed = _multivariate_polynomial(
                factor, variables, max_degree=max_degree, max_terms=max_terms,
            )
            if parsed is None:
                return None
            result = _mv_poly_mul(
                result, parsed, max_degree=max_degree, max_terms=max_terms,
            )
        return result
    if (
        isinstance(expression, Power)
        and isinstance(expression.exponent, ExactNumber)
        and expression.exponent.denominator == 1
        and expression.exponent.numerator >= 0
    ):
        parsed = _multivariate_polynomial(
            expression.base, variables,
            max_degree=max_degree, max_terms=max_terms,
        )
        if parsed is None:
            return None
        return _mv_poly_pow(
            parsed, expression.exponent.numerator,
            max_degree=max_degree, max_terms=max_terms,
        )
    return None


def _resultant_coefficients(polynomial, eliminated_index, retained_index):
    coefficients = {}
    for powers, coefficient in polynomial.items():
        eliminated_degree, retained_degree = (
            powers[eliminated_index], powers[retained_index]
        )
        current = coefficients.setdefault(eliminated_degree, {})
        current[retained_degree] = current.get(retained_degree, Rational(0)) + coefficient
    return {degree: _poly_clean(value) for degree, value in coefficients.items()}


def _polynomial_determinant(matrix, *, max_degree=64):
    """Division-free determinant for the small Sylvester matrices we allow."""
    size = len(matrix)
    cache = {}

    def determinant(row, columns):
        key = row, columns
        if key in cache:
            return cache[key]
        if row == size:
            return {0: Rational(1)}
        result = {}
        for position, column in enumerate(columns):
            entry = matrix[row][column]
            if not entry:
                continue
            remainder = columns[:position] + columns[position + 1:]
            term = _poly_mul(
                entry, determinant(row + 1, remainder),
                max_degree=max_degree,
            )
            result = _poly_add(
                result, term,
                factor=Rational(-1 if position % 2 else 1),
            )
        cache[key] = result
        return result

    return determinant(0, tuple(range(size)))


def _bivariate_resultant(first, second, eliminated_index, retained_index,
                         *, max_matrix=8):
    first_coefficients = _resultant_coefficients(
        first, eliminated_index, retained_index,
    )
    second_coefficients = _resultant_coefficients(
        second, eliminated_index, retained_index,
    )
    first_degree = max(first_coefficients, default=0)
    second_degree = max(second_coefficients, default=0)
    if first_degree == 0 or second_degree == 0:
        return None
    size = first_degree + second_degree
    if size > max_matrix:
        raise UnsupportedExpressionError(
            f"Polynomial resultant exceeds the {max_matrix}-row matrix limit"
        )
    first_descending = [
        first_coefficients.get(degree, {})
        for degree in range(first_degree, -1, -1)
    ]
    second_descending = [
        second_coefficients.get(degree, {})
        for degree in range(second_degree, -1, -1)
    ]
    matrix = []
    for shift in range(second_degree):
        matrix.append(
            [{} for _ in range(shift)] + first_descending
            + [{} for _ in range(size - shift - len(first_descending))]
        )
    for shift in range(first_degree):
        matrix.append(
            [{} for _ in range(shift)] + second_descending
            + [{} for _ in range(size - shift - len(second_descending))]
        )
    return _polynomial_determinant(matrix)


def _system_candidate_valid(sides, assignment, variables, domain, tolerance):
    numeric_assignment = {}
    for variable in variables:
        numeric = _numeric_value(assignment[variable])
        if numeric is None:
            return None
        numeric = complex(numeric)
        if domain == "real" and abs(numeric.imag) > tolerance:
            return False
        numeric_assignment[variable] = (
            numeric.real if abs(numeric.imag) <= tolerance else numeric
        )
    for left, right in sides:
        exact_residual = _substitute(_add(left, _neg(right)), assignment)
        if exact_residual == ZERO:
            continue
        try:
            first = complex(_evaluate(left, numeric_assignment))
            second = complex(_evaluate(right, numeric_assignment))
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            return False
        if not all(math.isfinite(value) for value in (
            first.real, first.imag, second.real, second.imag,
        )):
            return False
        residual = abs(first - second) / max(1.0, abs(first), abs(second))
        if residual > max(tolerance * 100, 1e-8):
            return False
    return True


def _solve_bivariate_resultant_system(sides, variables, domain, tolerance,
                                      max_branches=256):
    """Solve a bounded zero-dimensional two-variable polynomial system."""
    if len(variables) != 2:
        return None
    try:
        polynomials = [
            _multivariate_polynomial(_add(left, _neg(right)), list(variables))
            for left, right in sides
        ]
    except UnsupportedExpressionError:
        return None
    usable = [index for index, polynomial in enumerate(polynomials) if polynomial]
    if len(usable) < 2 or any(polynomials[index] is None for index in usable):
        return None
    for eliminated_index, retained_index in ((0, 1), (1, 0)):
        eliminated = variables[eliminated_index]
        retained = variables[retained_index]
        for first_position, first_index in enumerate(usable):
            for second_index in usable[first_position + 1:]:
                try:
                    resultant = _bivariate_resultant(
                        polynomials[first_index], polynomials[second_index],
                        eliminated_index, retained_index,
                    )
                except UnsupportedExpressionError:
                    continue
                if resultant is None or not resultant:
                    continue
                if max(resultant, default=0) == 0:
                    return () if resultant.get(0, Rational(0)) else None
                retained_set, complete = _solve_polynomial_exact(
                    resultant, retained, domain, None,
                )
                if not complete or not isinstance(retained_set, (FiniteSolutionSet, EmptySolutionSet)):
                    continue
                if isinstance(retained_set, EmptySolutionSet):
                    return ()
                if len(retained_set.values) > max_branches:
                    continue
                candidates, unresolved_branch = [], False
                for retained_value in retained_set.values:
                    assignment = {retained: retained_value}
                    eliminated_values = None
                    for left, right in sides:
                        substituted = (
                            _substitute(left, assignment),
                            _substitute(right, assignment),
                        )
                        if eliminated not in (
                            _variables(substituted[0]) | _variables(substituted[1])
                        ):
                            continue
                        result = solve_equation(
                            substituted, variable=eliminated, domain=domain,
                            method="symbolic", tolerance=tolerance,
                        )
                        if result.complete and isinstance(
                            result.solution_set, EmptySolutionSet
                        ):
                            eliminated_values = ()
                            break
                        if result.complete and isinstance(
                            result.solution_set, FiniteSolutionSet
                        ):
                            eliminated_values = result.solution_set.values
                            break
                    if eliminated_values is None:
                        unresolved_branch = True
                        break
                    for eliminated_value in eliminated_values:
                        candidate = dict(assignment, **{eliminated: eliminated_value})
                        valid = _system_candidate_valid(
                            sides, candidate, variables, domain, tolerance,
                        )
                        if valid is None:
                            unresolved_branch = True
                            break
                        if valid:
                            candidates.append({name: candidate[name] for name in variables})
                    if unresolved_branch:
                        break
                if unresolved_branch:
                    continue
                unique = {}
                for candidate in candidates:
                    key = tuple(str(candidate[name]) for name in variables)
                    unique[key] = candidate
                ordered = tuple(unique[key] for key in sorted(unique))
                return ordered
    return None


def solve_equation_system(equations, variables=None, *, domain="real",
                          numeric_fallback=False, initial=None, tolerance=1e-10,
                          max_iterations=1000, steps=False) -> EquationSystemSolution:
    """Solve an exact linear or bounded nonlinear system.

    Nonlinear symbolic solving uses complete triangular/affine substitution
    first, followed by a bounded bivariate polynomial resultant.  Unsupported
    or positive-dimensional systems remain explicitly unresolved; numerical
    fallback is still local and opt-in.
    """
    if isinstance(equations, (str, bytes)):
        raise TypeError("equations must be a nonempty sequence of equations")
    if not isinstance(numeric_fallback, (bool, np.bool_)) or not isinstance(steps, (bool, np.bool_)):
        raise TypeError("numeric_fallback and steps must be booleans")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(tolerance, (int, float, np.number)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    equations = list(equations)
    if not equations:
        raise ValueError("At least one equation is required")
    if domain not in {"real", "complex"}:
        raise ValueError("domain must be 'real' or 'complex'")
    if domain == "complex" and numeric_fallback:
        raise ValueError("nonlinear numerical fallback currently supports the real domain")
    sides = [_equation_input(equation) for equation in equations]
    discovered = sorted(set().union(*(_variables(left) | _variables(right) for left, right in sides)))
    if variables is None:
        variables = discovered
    else:
        if isinstance(variables, (str, bytes)):
            raise ValueError("variables must be a sequence of distinct non-empty names")
        variables = [value.name if hasattr(value, "name") else value for value in variables]
    if not variables or len(set(variables)) != len(variables) or any(not isinstance(value, str) or not value for value in variables):
        raise ValueError("variables must contain distinct non-empty names")
    rows = []
    for left, right in sides:
        parsed = _linear_form(_add(left, _neg(right)), list(variables))
        if parsed is None:
            rows = None
            break
        rows.append(parsed[0] + [-parsed[1]])
    recorded = ()
    if steps:
        recorded = (SolutionStep(
            "normalize_system", str(tuple(equations)),
            str(tuple(f"{left} = {right}" for left, right in sides)),
            "Normalize every equation before solving.",
        ),)
    if rows is not None:
        reduced, pivots = _rref(rows, len(variables))
        if steps:
            recorded += (SolutionStep(
                "rational_row_reduction", str(rows), str(reduced),
                "Compute exact reduced row-echelon form using rational arithmetic.",
            ),)
        if any(not any(row[:-1]) and row[-1] for row in reduced):
            return EquationSystemSolution(tuple(variables), (), "inconsistent", "symbolic", True, True, message="The exact row reduction contains a contradiction.")
        free = [column for column in range(len(variables)) if column not in pivots]
        parameters = tuple(f"t{index}" for index in range(len(free)))
        values = {variables[column]: Symbol(parameters[index]) for index, column in enumerate(free)}
        for row_index in range(len(pivots) - 1, -1, -1):
            column = pivots[row_index]
            value = ExactNumber(reduced[row_index][-1])
            for free_index in free:
                if reduced[row_index][free_index]:
                    value = _add(value, _neg(_mul(ExactNumber(reduced[row_index][free_index]), values[variables[free_index]])))
            values[variables[column]] = value
        ordered = {variable: values[variable] for variable in variables}
        return EquationSystemSolution(tuple(variables), (ordered,), "solved", "symbolic", True, True, parameters, steps=recorded, message="Exact rational row reduction." if not free else "Exact parametric solution for an underdetermined system.")
    triangular = _solve_triangular_system(sides, variables, domain, float(tolerance))
    if triangular is not None:
        if steps:
            recorded += (SolutionStep(
                "triangular_substitution", str(tuple(f"{left} = {right}" for left, right in sides)), str(triangular),
                "Solve one variable at a time, including safe affine elimination, and substitute each exact branch.",
            ),)
        if not triangular:
            return EquationSystemSolution(
                tuple(variables), (), "inconsistent", "symbolic", True, True,
                steps=recorded,
                message="Exact nonlinear substitution proves that the system is inconsistent.",
            )
        return EquationSystemSolution(
            tuple(variables), triangular, "solved", "symbolic", True, True,
            steps=recorded,
            message="Exact triangular and affine nonlinear substitution.",
        )
    resultant = _solve_bivariate_resultant_system(
        sides, variables, domain, float(tolerance),
    )
    if resultant is not None:
        if steps:
            recorded += (SolutionStep(
                "polynomial_resultant",
                str(tuple(f"{left} = {right}" for left, right in sides)),
                str(resultant),
                "Eliminate one variable with an exact bounded Sylvester resultant, solve the retained polynomial, and verify every candidate in the original system.",
            ),)
        if not resultant:
            return EquationSystemSolution(
                tuple(variables), (), "inconsistent", "symbolic", True, True,
                steps=recorded,
                message="The exact polynomial resultant proves that the system is inconsistent.",
            )
        return EquationSystemSolution(
            tuple(variables), resultant, "solved", "symbolic", True, True,
            steps=recorded,
            message="Exact zero-dimensional bivariate polynomial resultant.",
        )
    if not numeric_fallback:
        return EquationSystemSolution(
            tuple(variables), (), "unresolved", "symbolic", False, False,
            steps=recorded,
            message=(
                "No supported complete exact system transformation was found. "
                "The native solver supports rational linear systems, finite "
                "triangular/affine substitutions, and bounded zero-dimensional "
                "bivariate polynomial resultants; enable numeric_fallback with "
                "initial values for a local nonlinear solution."
            ),
        )
    if initial is None:
        raise ValueError("Nonlinear numerical fallback requires initial values")
    if isinstance(initial, Mapping):
        if set(initial) != set(variables):
            raise ValueError("initial keys must match variables")
        initial_values = [initial[name] for name in variables]
    else:
        initial_values = list(initial)
    from kiwicalc.numeric.multivariable import solve_system

    def residual(*point):
        assignment = dict(zip(variables, point))
        result = []
        for left, right in sides:
            value = complex(_evaluate(left, assignment) - _evaluate(right, assignment))
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("Equation residual is not finite at the current approximation")
            if abs(value.imag) > max(float(tolerance) * 10, 1e-12):
                raise ValueError("Equation is not real-valued at the current approximation")
            result.append(float(value.real))
        return result

    information = solve_system(residual, initial_values, tolerance=tolerance, max_iterations=max_iterations, return_info=True)
    mapping = {name: float(value) for name, value in zip(variables, information.value)}
    if steps:
        recorded += (SolutionStep(
            "local_newton", str(dict(zip(variables, initial_values))), str(mapping),
            "Apply damped Newton iteration from the supplied initial values.",
        ),)
    return EquationSystemSolution(tuple(variables), (mapping,), "solved" if information.converged else "unresolved", "numeric", False, False, residuals=(information.residual,), steps=recorded, message=information.message)


__all__ = [
    "SymbolicExpression", "ExactNumber", "Symbol", "SymbolicConstant", "Add",
    "Multiply", "Power", "SymbolicFunction", "RootOf", "parse_symbolic",
    "to_symbolic", "to_legacy_expression", "simplify_symbolic", "is_canonical_symbolic",
    "structurally_equal", "differentiate_symbolic", "RewriteContext",
    "RewriteRule", "RewriteApplication", "RewriteResult",
    "available_rewrite_rules", "rewrite_symbolic",
    "normalize_polynomial_symbolic", "normalize_rational_symbolic",
    "Condition", "TruthCondition", "RelationCondition", "DefinedCondition",
    "BetweenCondition", "OpaqueCondition", "CompoundCondition", "AssumptionSet",
    "parse_condition", "condition_from_dict", "simplify_condition", "negate_condition",
    "SolutionSet", "EmptySolutionSet", "UniversalSolutionSet",
    "FiniteSolutionSet", "IntervalSolutionSet", "ParametricSolutionSet",
    "UnionSolutionSet", "ConditionalSolutionSet", "EquationState", "SolutionStep",
    "EquationSolution", "EquationSystemSolution", "symbolic_from_dict",
    "solve_equation", "solve_equation_assuming", "solve_inequality",
    "solve_equation_system",
]
