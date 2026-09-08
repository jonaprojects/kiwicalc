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
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

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
        return False

    def refutes(self, condition):
        condition = simplify_condition(condition)
        if isinstance(condition, TruthCondition):
            return not condition.value
        negated = negate_condition(condition)
        return self.entails(negated)

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
            flat.append(Multiply((ZERO, base)))
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
            return ONE
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
    """Return the deterministic canonical form produced by the native factories."""
    value = to_symbolic(value)
    if isinstance(value, Add): return _add(*(simplify_symbolic(item) for item in value.terms))
    if isinstance(value, Multiply): return _mul(*(simplify_symbolic(item) for item in value.factors))
    if isinstance(value, Power): return _pow(simplify_symbolic(value.base), simplify_symbolic(value.exponent))
    if isinstance(value, SymbolicFunction): return _function(value.name, *(simplify_symbolic(item) for item in value.arguments))
    return value


def structurally_equal(first, second):
    return simplify_symbolic(first) == simplify_symbolic(second)


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
    numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return ExactNumber(numerator, denominator)
    return _function("sqrt", ExactNumber(value))


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
        if lower_numeric is None or upper_numeric is None:
            conditions = _interval_conditions(solution_set.lower, interval) + _interval_conditions(solution_set.upper, interval)
            return ConditionalSolutionSet(solution_set, conditions)
        lower, upper = max(float(lower_numeric), interval[0]), min(float(upper_numeric), interval[1])
        if lower > upper:
            return EMPTY
        return IntervalSolutionSet(lower, upper, solution_set.lower_closed, solution_set.upper_closed)
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


def _symbolic_dispatch(left, right, variable, domain, interval, trace):
    residual = _collapse_guarded_zeros(_add(left, _neg(right)))
    trace("normalize", f"{left} = {right}", f"{residual} = 0", "Move all terms to the left side.")
    rational = _poly_fraction(residual, variable)
    if rational is not None:
        numerator, denominator = rational
        conditions = AssumptionSet()
        if denominator != {0: Rational(1)}:
            denominator_expression = _polynomial_expression(denominator, variable)
            conditions = AssumptionSet((RelationCondition(
                denominator_expression, "!=", ZERO,
                f"{_poly_string(denominator, variable)} != 0",
            ),))
            trace("clear_denominators", f"{residual} = 0", f"{_poly_string(numerator, variable)} = 0", "Clear denominators while retaining their exclusions.", conditions)
        solution_set, complete = _solve_polynomial_exact(numerator, variable, domain, interval)
        if isinstance(solution_set, FiniteSolutionSet) and denominator != {0: Rational(1)}:
            values, mults = [], []
            for value, multiplicity in zip(solution_set.values, solution_set.multiplicities):
                numeric = _numeric_value(value)
                if numeric is None or abs(complex(_poly_eval_numeric(denominator, numeric))) > 1e-10:
                    values.append(value); mults.append(multiplicity)
            solution_set = FiniteSolutionSet(tuple(values), tuple(mults)) if values else EMPTY
        trace("solve_polynomial", f"{_poly_string(numerator, variable)} = 0", str(solution_set), "Solve the exact polynomial numerator.")
        return solution_set, conditions, complete, "symbolic"

    symbolic_affine = _affine_symbolic(residual, variable)
    if symbolic_affine is not None and symbolic_affine[0] != ZERO:
        coefficient, constant = symbolic_affine
        root = _mul(_neg(constant), _pow(coefficient, NEG_ONE))
        coefficient_value = _numeric_value(coefficient)
        if coefficient_value is not None and coefficient_value != 0:
            result = FiniteSolutionSet((root,)) if _in_interval(root, interval) else EMPTY
            trace("solve_symbolic_linear", f"{residual} = 0", str(result), "Divide by the known nonzero symbolic coefficient.")
            return result, AssumptionSet(), True, "symbolic"
        branches = UnionSolutionSet((
            ConditionalSolutionSet(FiniteSolutionSet((root,)), (RelationCondition(coefficient, "!=", ZERO),)),
            ConditionalSolutionSet(UniversalSolutionSet(domain), (
                RelationCondition(coefficient, "=", ZERO),
                RelationCondition(constant, "=", ZERO),
            )),
        ))
        trace("solve_conditional_linear", f"{residual} = 0", str(branches), "Divide by a symbolic coefficient only on its nonzero branch; retain the identity branch.")
        return branches, AssumptionSet(), True, "symbolic"

    # Non-polynomial complex inversion requires explicit logarithm branches and
    # other multivalued-function machinery.  Never return a principal branch as
    # though it were the complete complex solution.
    if domain == "complex":
        return None, AssumptionSet(), False, "symbolic"

    left_logs, right_logs = _log_terms(left), _log_terms(right)
    if left_logs and right_logs and left_logs[0] == right_logs[0]:
        left_argument = _mul(*left_logs[1])
        right_argument = _mul(*right_logs[1])
        combined = _add(left_argument, _neg(right_argument))
        parsed = _poly_fraction(combined, variable)
        if parsed is not None:
            result, complete = _solve_polynomial_exact(parsed[0], variable, domain, interval)
            restrictions = AssumptionSet(tuple(_render_condition(argument, "> 0", variable) for argument in left_logs[1] + right_logs[1]))
            trace("combine_logarithms", f"{left} = {right}", f"{left_argument} = {right_argument}", "Combine logarithms with the same base and retain every argument-domain restriction.", restrictions)
            return result, restrictions, complete, "symbolic"

    function_side = _same_function_side(left, right)
    if function_side:
        function, constant = function_side
        argument = function.arguments[-1]
        if function.name == "exp" and not _variables(constant):
            affine = _affine_symbolic(argument, variable)
            numeric = _numeric_value(constant)
            if numeric is not None and (isinstance(numeric, complex) or numeric <= 0):
                return EMPTY, AssumptionSet(), True, "symbolic"
            if affine is not None and affine[0] != ZERO:
                coefficient, offset = affine
                inverse = _function("ln", constant)
                root = _mul(_add(inverse, _neg(offset)), _pow(coefficient, NEG_ONE))
                result = FiniteSolutionSet((root,))
                condition = AssumptionSet() if numeric is not None else AssumptionSet((RelationCondition(constant, ">", ZERO),))
                trace("invert_exponential", f"{function} = {constant}", f"{argument} = {inverse}", "Apply the natural logarithm and retain positivity of the right side.", condition)
                return result, condition, True, "symbolic"
        if function.name == "abs" and isinstance(constant, ExactNumber):
            if constant.value < 0:
                return EMPTY, AssumptionSet(), True, "symbolic"
            roots = []
            for target in (constant, ExactNumber(-constant.value)):
                root = _solve_affine_equal(argument, target, variable)
                if root is not None and root not in roots:
                    roots.append(root)
            if roots:
                result = FiniteSolutionSet(tuple(roots))
                trace("split_absolute", f"abs({argument}) = {constant}", str(result), "Split an absolute-value equation into its positive and negative branches.")
                return result, AssumptionSet(), True, "symbolic"
        if function.name == "sqrt":
            if isinstance(constant, ExactNumber) and constant.value < 0 and domain == "real":
                return EMPTY, AssumptionSet(), True, "symbolic"
            squared = _add(argument, _neg(_pow(constant, ExactNumber(2))))
            parsed = _poly_fraction(squared, variable)
            if parsed is not None and parsed[1] == {0: Rational(1)}:
                result, complete = _solve_polynomial_exact(parsed[0], variable, domain, interval)
                conditions = AssumptionSet((RelationCondition(constant, ">=", ZERO),)) if domain == "real" else AssumptionSet()
                trace("isolate_radical", f"sqrt({argument}) = {constant}", f"{squared} = 0", "Square the isolated principal radical; candidates will be checked in the original equation.", conditions)
                return result, conditions, complete, "symbolic"
        if function.name in {"ln", "log"} and len(function.arguments) in {1, 2}:
            affine = _affine(argument, variable)
            if affine and affine[0] and not _variables(constant):
                inverse = _function("exp", constant) if function.name == "ln" or len(function.arguments) == 1 else _pow(function.arguments[0], constant)
                root = ExactNumber(-affine[1] / affine[0]) if inverse == ZERO else _mul(ExactNumber(1 / affine[0]), _add(inverse, ExactNumber(-affine[1])))
                result = FiniteSolutionSet((root,)) if _in_interval(root, interval) else EMPTY
                condition = _render_condition(argument, "> 0", variable)
                conditions = AssumptionSet((condition,))
                trace("invert_logarithm", f"{function} = {constant}", f"{argument} = {inverse}", "Apply the matching exponential function and preserve the logarithm domain.", conditions)
                return result, conditions, True, "symbolic"
        if function.name in {"sin", "cos", "tan"} and not _variables(constant):
            affine, phase = _affine(argument, variable), _trig_phase(function.name, constant)
            numeric = _numeric_value(constant)
            if function.name in {"sin", "cos"} and numeric is not None and (isinstance(numeric, complex) or not -1 <= numeric <= 1):
                return EMPTY, AssumptionSet(), True, "symbolic"
            if phase is None:
                phase = _function({"sin": "asin", "cos": "acos", "tan": "atan"}[function.name], constant)
            if affine and affine[0] and phase is not None:
                a, b = affine
                if function.name == "sin":
                    if constant == ZERO:
                        families = _parameter_family(variable, a, b, ZERO, PI)
                    elif constant in {NEG_ONE, ONE}:
                        families = _parameter_family(variable, a, b, phase, _mul(ExactNumber(2), PI))
                    else:
                        families = UnionSolutionSet((_parameter_family(variable, a, b, phase, _mul(ExactNumber(2), PI)), _parameter_family(variable, a, b, _add(PI, _neg(phase)), _mul(ExactNumber(2), PI))))
                elif function.name == "cos":
                    if constant in {NEG_ONE, ONE}:
                        families = _parameter_family(variable, a, b, phase, _mul(ExactNumber(2), PI))
                    else:
                        families = UnionSolutionSet((_parameter_family(variable, a, b, phase, _mul(ExactNumber(2), PI)), _parameter_family(variable, a, b, phase, _mul(ExactNumber(2), PI), sign=-1)))
                else:
                    families = _parameter_family(variable, a, b, phase, PI)
                result = _filter_parametric(families, interval)
                trace("invert_trigonometric", f"{function} = {constant}", str(result), "Return the complete integer-parameterized periodic family." if interval is None else "Enumerate the exact family over the requested interval.")
                return result, AssumptionSet(), True, "symbolic"

    # exp(affine)=constant is represented as a function; constant-base powers
    # are represented by Power.
    power_side = (left, right) if isinstance(left, Power) and not _variables(right) else (right, left) if isinstance(right, Power) and not _variables(left) else None
    if power_side:
        power, constant = power_side
        affine = _affine(power.exponent, variable)
        if affine and affine[0] and not _variables(power.base) and not _variables(constant):
            base_value, constant_value = _numeric_value(power.base), _numeric_value(constant)
            if base_value is None or constant_value is None or isinstance(base_value, complex) or isinstance(constant_value, complex):
                return None, AssumptionSet(), False, "symbolic"
            if base_value == 1:
                return (UniversalSolutionSet(domain) if constant_value == 1 else EMPTY), AssumptionSet(), True, "symbolic"
            if base_value <= 0:
                return None, AssumptionSet(), False, "symbolic"
            if constant_value <= 0:
                return EMPTY, AssumptionSet(), True, "symbolic"
            if isinstance(constant, Power) and constant.base == power.base and not _variables(constant.exponent):
                target = constant.exponent
            elif power.base == constant:
                target = ONE
            else:
                target = _exact_log_ratio(power.base, constant) or _mul(_function("ln", constant), _pow(_function("ln", power.base), NEG_ONE))
            root = _mul(ExactNumber(1 / affine[0]), _add(target, ExactNumber(-affine[1])))
            result = FiniteSolutionSet((root,)) if _in_interval(root, interval) else EMPTY
            trace("invert_exponential", f"{power} = {constant}", str(result), "Apply logarithms to isolate the exponent.")
            return result, AssumptionSet(), True, "symbolic"
    return None, AssumptionSet(), False, "symbolic"


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
    lower, upper = interval
    evaluations = 0
    assignments = dict(assignments or {})

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        try:
            values = dict(assignments)
            values[variable] = x
            value = complex(_evaluate(left, values) - _evaluate(right, values))
            if abs(value.imag) > 1e-8 or not math.isfinite(value.real):
                return math.nan
            return value.real
        except (ArithmeticError, ValueError, OverflowError):
            return math.nan

    sample_count = min(4097, max(257, max_iterations * 4 + 1))
    points = np.linspace(lower, upper, sample_count)
    values = np.asarray([evaluate(float(point)) for point in points])
    candidates = []
    for index, (x, value) in enumerate(zip(points, values)):
        if math.isfinite(value) and abs(value) <= tolerance:
            candidates.append(float(x))
        if index == 0 or not math.isfinite(value) or not math.isfinite(values[index - 1]) or value * values[index - 1] >= 0:
            continue
        a, b, fa, fb = float(points[index - 1]), float(x), float(values[index - 1]), float(value)
        for _ in range(max_iterations):
            midpoint = (a + b) / 2
            fm = evaluate(midpoint)
            if not math.isfinite(fm):
                break
            if abs(fm) <= tolerance:
                candidates.append(midpoint); break
            if b - a <= tolerance * max(1.0, abs(midpoint)):
                break
            if fa * fm <= 0:
                b, fb = midpoint, fm
            else:
                a, fa = midpoint, fm
    # Safeguarded Newton refinement catches sampled tangent roots.
    absolute = np.abs(values)
    for index in range(1, sample_count - 1):
        if not math.isfinite(values[index]) or absolute[index] > absolute[index - 1] or absolute[index] > absolute[index + 1]:
            continue
        x = float(points[index])
        for _ in range(min(max_iterations, 50)):
            fx = evaluate(x)
            h = math.sqrt(np.finfo(float).eps) * max(1.0, abs(x))
            derivative = (evaluate(x + h) - evaluate(x - h)) / (2 * h)
            if not math.isfinite(fx) or not math.isfinite(derivative) or abs(derivative) < 1e-14:
                break
            candidate = x - fx / derivative
            if candidate < lower or candidate > upper:
                break
            if abs(candidate - x) <= tolerance * max(1.0, abs(candidate)):
                x = candidate; break
            x = candidate
        fx = evaluate(x)
        if math.isfinite(fx) and abs(fx) <= max(tolerance * 10, 1e-8):
            candidates.append(x)
    candidates.sort()
    unique = []
    for value in candidates:
        if not unique or abs(value - unique[-1]) > max(tolerance * 10, 1e-9) * max(1.0, abs(value)):
            unique.append(value)
        elif abs(evaluate(value)) < abs(evaluate(unique[-1])):
            unique[-1] = value
    residuals = tuple(abs(evaluate(value)) for value in unique)
    return FiniteSolutionSet(tuple(unique)) if unique else EMPTY, residuals, evaluations


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
        solution_set, conditions, complete, used_method = _symbolic_dispatch(working_left, working_right, variable, domain, interval, trace)
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
    trace("numeric_isolation", f"{left} = {right}", str(solution_set), "Isolate and refine real roots over the requested finite interval.")
    return EquationSolution(variable, solution_set, "solved", "numeric" if method == "numeric" else "hybrid", False, False, conditions=original_conditions, residuals=residuals, steps=tuple(recorded), message="Approximate roots found over the requested interval; completeness is not guaranteed for arbitrary functions.", evaluations=evaluations)


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
                if value is None or abs(complex(value)) > tolerance:
                    return
            else:
                active.append((left, right, involved))
        if not remaining:
            branches.append(dict(assignments)); return
        choice = next(((left, right, next(iter(involved))) for left, right, involved in active if len(involved) == 1), None)
        if choice is None:
            return
        left, right, variable = choice
        result = solve_equation((left, right), variable=variable, domain=domain, method="symbolic", tolerance=tolerance)
        if not result.complete or not isinstance(result.solution_set, FiniteSolutionSet):
            return
        next_remaining = [name for name in remaining if name != variable]
        for value in result.solution_set.values:
            recurse(simplified, next_remaining, dict(assignments, **{variable: value}))

    recurse(sides, list(variables), {})
    if not branches:
        return None
    return tuple({variable: branch[variable] for variable in variables} for branch in branches)


def solve_equation_system(equations, variables=None, *, domain="real",
                          numeric_fallback=False, initial=None, tolerance=1e-10,
                          max_iterations=1000, steps=False) -> EquationSystemSolution:
    """Solve an exact linear system, or explicitly fall back to local Newton."""
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
                "Solve one variable at a time and substitute each exact branch.",
            ),)
        return EquationSystemSolution(tuple(variables), triangular, "solved", "symbolic", True, True, steps=recorded, message="Exact triangular polynomial substitution.")
    if not numeric_fallback:
        return EquationSystemSolution(tuple(variables), (), "unresolved", "symbolic", False, False, message="The native symbolic system solver currently supports linear systems; enable numeric_fallback with initial values for a local nonlinear solution.")
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
    "to_symbolic", "to_legacy_expression", "simplify_symbolic",
    "structurally_equal", "differentiate_symbolic",
    "Condition", "TruthCondition", "RelationCondition", "DefinedCondition",
    "BetweenCondition", "OpaqueCondition", "CompoundCondition", "AssumptionSet",
    "parse_condition", "condition_from_dict", "simplify_condition", "negate_condition",
    "SolutionSet", "EmptySolutionSet", "UniversalSolutionSet",
    "FiniteSolutionSet", "IntervalSolutionSet", "ParametricSolutionSet",
    "UnionSolutionSet", "ConditionalSolutionSet", "EquationState", "SolutionStep",
    "EquationSolution", "EquationSystemSolution", "symbolic_from_dict",
    "solve_equation", "solve_equation_assuming", "solve_equation_system",
]
