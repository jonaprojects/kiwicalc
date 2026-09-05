"""Fixed and adaptive sampling for two-dimensional function plots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

from kiwicalc.core.interfaces import IExpression
from kiwicalc.functions.function import Function


@dataclass(frozen=True)
class PlotSample:
    """Sampled plot coordinates together with bounded-work diagnostics."""

    x: np.ndarray
    y: np.ndarray
    sampling: str
    evaluations: int
    initial_points: int
    refined_points: int
    discontinuities: int
    truncated: bool

    @property
    def point_count(self) -> int:
        return int(self.x.size)


def normalize_sampling(sampling: str) -> str:
    if not isinstance(sampling, str):
        raise TypeError("sampling must be 'fixed' or 'adaptive'")
    mode = sampling.strip().lower()
    if mode not in {'fixed', 'adaptive'}:
        raise ValueError("sampling must be 'fixed' or 'adaptive'")
    return mode


def _positive_finite(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number")
    if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _nonnegative_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _callable(source: Union[str, Callable, IExpression]) -> Callable:
    if isinstance(source, str):
        source = Function(source)
    if isinstance(source, Function):
        return source.lambda_expression if callable(source.lambda_expression) else source
    if isinstance(source, IExpression):
        converted = source.to_lambda()
        if callable(converted):
            return converted
        if callable(source):
            return source
        raise ValueError("expression could not be converted to a callable function")
    if callable(source):
        return source
    raise TypeError("source must be a callable, formula string, or expression")


def _seed_values(start, stop, step, values, *, adaptive):
    if values is not None:
        try:
            x = np.asarray(tuple(values), dtype=float)
        except (TypeError, ValueError):
            raise ValueError("values must be finite numeric coordinates")
        minimum = 2 if adaptive else 1
        if x.ndim != 1 or x.size < minimum or not np.isfinite(x).all():
            qualifier = "at least two" if adaptive else "at least one"
            raise ValueError(f"values must contain {qualifier} finite coordinates")
        if adaptive and np.any(np.diff(x) <= 0):
            raise ValueError("adaptive values must be strictly increasing")
        return x

    step = _positive_finite(step, 'step')
    try:
        start, stop = float(start), float(stop)
    except (TypeError, ValueError):
        raise ValueError("start and stop must be finite numbers")
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        prefix = "adaptive sampling" if adaptive else "sampling"
        raise ValueError(f"{prefix} requires finite start < stop")
    x = np.arange(start, stop, step, dtype=float)
    if not x.size or not math.isclose(float(x[-1]), stop, rel_tol=1e-12, abs_tol=step * 1e-12):
        x = np.append(x, stop)
    else:
        x[-1] = stop
    return x


class _Evaluator:
    def __init__(self, source):
        self.function = _callable(source)
        self.cache = {}

    @staticmethod
    def _coerce(value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return np.nan
        return value if math.isfinite(value) else np.nan

    def evaluate_many(self, coordinates):
        missing = list(dict.fromkeys(float(value) for value in coordinates if float(value) not in self.cache))
        if not missing:
            return
        array = np.asarray(missing, dtype=float)
        try:
            with np.errstate(all='ignore'):
                result = np.asarray(self.function(array), dtype=float)
            if result.shape == ():
                result = np.full(array.shape, float(result))
            result = np.broadcast_to(result, array.shape)
            for x, y in zip(missing, result):
                self.cache[x] = self._coerce(y)
            return
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            pass
        for x in missing:
            try:
                self.cache[x] = self._coerce(self.function(x))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                self.cache[x] = np.nan


def sample_for_plot(
    source: Union[str, Callable, IExpression], start: float=-10,
    stop: float=10, step: float=0.01, *, values: Optional[Iterable[float]]=None,
    sampling: str='fixed', tolerance: float=1e-3, max_points: int=5000,
    max_depth: int=12,
) -> PlotSample:
    """Sample a scalar function for plotting.

    ``sampling='fixed'`` evaluates only the seed coordinates. Adaptive mode
    additionally probes interval midpoints and subdivides where midpoint
    interpolation error exceeds ``tolerance``. Work is bounded by
    ``max_points`` and ``max_depth``.
    """
    mode = normalize_sampling(sampling)
    seeds = _seed_values(start, stop, step, values, adaptive=mode == 'adaptive')
    evaluator = _Evaluator(source)
    evaluator.evaluate_many(seeds)
    if mode == 'fixed':
        x = np.asarray(seeds, dtype=float)
        y = np.asarray([evaluator.cache[float(value)] for value in x], dtype=float)
        return PlotSample(
            x=x, y=y, sampling=mode, evaluations=len(evaluator.cache),
            initial_points=int(seeds.size), refined_points=0,
            discontinuities=int(np.count_nonzero(~np.isfinite(y))),
            truncated=False,
        )

    tolerance = _positive_finite(tolerance, 'tolerance')
    max_points = _nonnegative_integer(max_points, 'max_points')
    max_depth = _nonnegative_integer(max_depth, 'max_depth')
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if seeds.size > max_points:
        raise ValueError("max_points cannot be smaller than the initial sample count")

    truncated = False
    intervals = [(float(left), float(right), 0) for left, right in zip(seeds[:-1], seeds[1:])]
    while intervals:
        remaining = max_points - len(evaluator.cache)
        if remaining <= 0:
            truncated = True
            break
        selected, deferred = intervals[:remaining], intervals[remaining:]
        if deferred:
            truncated = True
        middles = [(left + right) / 2 for left, right, _ in selected]
        evaluator.evaluate_many(middles)
        next_intervals = []
        for (left, right, depth), middle in zip(selected, middles):
            left_y = evaluator.cache[left]
            middle_y = evaluator.cache[middle]
            right_y = evaluator.cache[right]
            finite = np.isfinite((left_y, middle_y, right_y))
            if not finite.all():
                needs_refinement = bool(np.isfinite(middle_y) and finite.any())
            else:
                scale = 1 + max(abs(left_y), abs(middle_y), abs(right_y))
                error = abs(middle_y - (left_y + right_y) / 2) / scale
                pole_like = (
                    left_y * right_y < 0
                    and min(abs(left_y), abs(right_y)) > 1 / tolerance
                )
                needs_refinement = error > tolerance or pole_like
                if pole_like and depth >= max_depth:
                    evaluator.cache[middle] = np.nan
            if not needs_refinement:
                continue
            if depth >= max_depth:
                truncated = True
                continue
            next_intervals.extend(((left, middle, depth + 1), (middle, right, depth + 1)))
        intervals = next_intervals

    x = np.asarray(sorted(evaluator.cache), dtype=float)
    y = np.asarray([evaluator.cache[value] for value in x], dtype=float)
    discontinuities = int(np.count_nonzero(~np.isfinite(y)))
    return PlotSample(
        x=x, y=y, sampling=mode, evaluations=len(evaluator.cache),
        initial_points=int(seeds.size), refined_points=int(x.size - seeds.size),
        discontinuities=discontinuities, truncated=truncated,
    )


__all__ = ['PlotSample', 'sample_for_plot']
