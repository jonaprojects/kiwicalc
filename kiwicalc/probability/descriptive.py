"""Friendly, NumPy-backed descriptive statistics.

The functions in this module share four conventions:

* ``axis=None`` reduces all observations; an integer axis reduces that axis.
* weights may have the input shape, be broadcastable to it, or be one-dimensional
  along the reduction axis.  Zero-weight observations are ignored.
* ``nan_policy`` is one of ``"raise"``, ``"omit"``, or ``"propagate"``.
* empty reductions and reductions with no positive total weight raise
  :class:`ValueError` (unless a NaN is being propagated).
"""

from dataclasses import dataclass
import builtins
import math
from numbers import Integral, Real

import numpy as np


_NAN_POLICIES = {'raise', 'omit', 'propagate'}


@dataclass(frozen=True)
class FrequencyTable:
    """Values and their weighted or unweighted frequencies."""

    values: np.ndarray
    counts: np.ndarray
    proportions: np.ndarray

    def to_dict(self):
        return {
            value.item() if isinstance(value, np.generic) else value:
                count.item() if isinstance(count, np.generic) else count
            for value, count in zip(self.values, self.counts)
        }

    def __len__(self):
        return len(self.values)


@dataclass(frozen=True)
class FiveNumberSummary:
    """Minimum, quartiles, and maximum for one or more data slices."""

    minimum: object
    q1: object
    median: object
    q3: object
    maximum: object

    def as_dict(self):
        return {
            'min': self.minimum,
            'q1': self.q1,
            'median': self.median,
            'q3': self.q3,
            'max': self.maximum,
        }


@dataclass(frozen=True)
class OutlierFences:
    """Tukey lower and upper fences."""

    lower: object
    upper: object

    def __iter__(self):
        yield self.lower
        yield self.upper


@dataclass(frozen=True)
class ContingencyTable:
    """A two-way categorical frequency table."""

    row_values: np.ndarray
    column_values: np.ndarray
    counts: np.ndarray
    proportions: np.ndarray

    def to_dict(self):
        result = {}
        for row_index, row_value in enumerate(self.row_values):
            key = row_value.item() if isinstance(row_value, np.generic) else row_value
            result[key] = {}
            for column_index, column_value in enumerate(self.column_values):
                column_key = (
                    column_value.item()
                    if isinstance(column_value, np.generic)
                    else column_value
                )
                value = self.counts[row_index, column_index]
                result[key][column_key] = (
                    value.item() if isinstance(value, np.generic) else value
                )
        return result


@dataclass(frozen=True)
class DescriptiveSummary:
    """Unified output returned by :func:`describe`."""

    count: object
    mean: object
    std: object
    minimum: object
    q1: object
    median: object
    q3: object
    maximum: object
    iqr: object

    def as_dict(self):
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.minimum,
            'q1': self.q1,
            'median': self.median,
            'q3': self.q3,
            'max': self.maximum,
            'iqr': self.iqr,
        }

    def __getitem__(self, key):
        return self.as_dict()[key]


def _policy(nan_policy):
    if not isinstance(nan_policy, str) or nan_policy not in _NAN_POLICIES:
        raise ValueError("nan_policy must be 'raise', 'omit', or 'propagate'")
    return nan_policy


def _axis(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError('axis must be an integer or None')
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f'axis {axis} is out of bounds for an array with {ndim} dimensions')
    return axis


def _numeric_array(data):
    try:
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('data must contain real numeric values') from exc
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _object_array(data):
    array = np.asarray(data, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _broadcast_weights(weights, shape, axis):
    if weights is None:
        return None
    try:
        array = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('weights must contain real numeric values') from exc
    if array.ndim == 1 and axis is not None and array.size == shape[axis]:
        reshape = [1] * len(shape)
        reshape[axis] = array.size
        array = array.reshape(reshape)
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError('weights must be broadcastable to the data shape') from exc


def _is_missing(value):
    if isinstance(value, (tuple, list)):
        return any(_is_missing(item) for item in value)
    if value is None:
        return True
    try:
        missing = np.isnan(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _rows(data, axis=None, weights=None, nan_policy='propagate', numeric=True):
    policy = _policy(nan_policy)
    array = _numeric_array(data) if numeric else _object_array(data)
    reduction_axis = _axis(axis, array.ndim)
    weight_array = _broadcast_weights(weights, array.shape, reduction_axis)

    if reduction_axis is None:
        matrix = array.reshape(1, -1)
        weight_matrix = None if weight_array is None else weight_array.reshape(1, -1)
        output_shape = ()
    else:
        matrix = np.moveaxis(array, reduction_axis, -1)
        output_shape = matrix.shape[:-1]
        matrix = matrix.reshape(-1, matrix.shape[-1])
        if weight_array is None:
            weight_matrix = None
        else:
            weight_matrix = np.moveaxis(weight_array, reduction_axis, -1)
            weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])

    if matrix.shape[1] == 0:
        raise ValueError('data must contain at least one observation')

    prepared = []
    for index, row in enumerate(matrix):
        row_weights = None if weight_matrix is None else weight_matrix[index]
        if numeric:
            missing = np.isnan(row)
        else:
            missing = np.fromiter((_is_missing(value) for value in row), dtype=bool)
        if row_weights is not None:
            weight_missing = np.isnan(row_weights)
            missing = missing | weight_missing
            finite_weights = row_weights[~weight_missing]
            if np.any(~np.isfinite(finite_weights)):
                raise ValueError('weights must be finite')
            if np.any(finite_weights < 0):
                raise ValueError('weights cannot be negative')
        if policy == 'raise' and np.any(missing):
            raise ValueError('data contains missing values')
        if policy == 'propagate' and np.any(missing):
            prepared.append((None, None, True))
            continue
        if policy == 'omit':
            row = row[~missing]
            if row_weights is not None:
                row_weights = row_weights[~missing]
        if row_weights is not None:
            if float(np.sum(row_weights)) <= 0:
                raise ValueError('weights must have a positive total')
            positive = row_weights > 0
            row = row[positive]
            row_weights = row_weights[positive]
        if row.size == 0:
            raise ValueError('no observations remain after applying the data policy')
        prepared.append((row, row_weights, False))
    return prepared, output_shape, array.shape, reduction_axis


def _finish(values, shape, dtype=float):
    array = np.asarray(values, dtype=dtype).reshape(shape)
    return array.item() if array.ndim == 0 else array


def _reduce(data, reducer, *, axis=None, weights=None, nan_policy='propagate'):
    rows, shape, _, _ = _rows(data, axis, weights, nan_policy, numeric=True)
    values = []
    for row, row_weights, propagated in rows:
        values.append(np.nan if propagated else reducer(row, row_weights))
    return _finish(values, shape)


def count(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the observation count, or total weight when weights are supplied."""
    _validate_ddof(ddof)
    return _reduce(
        data,
        lambda row, row_weights: (
            float(row.size) if row_weights is None else float(np.sum(row_weights))
        ),
        axis=axis,
        weights=weights,
        nan_policy=nan_policy,
    )


def mean(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return an arithmetic mean."""
    _validate_ddof(ddof)
    return _reduce(
        data,
        lambda row, row_weights: (
            float(np.mean(row))
            if row_weights is None
            else float(np.average(row, weights=row_weights))
        ),
        axis=axis,
        weights=weights,
        nan_policy=nan_policy,
    )


def weighted_mean(data, weights, axis=None, ddof=0, nan_policy='propagate'):
    """Explicit weighted-mean spelling."""
    return mean(data, axis=axis, weights=weights, ddof=ddof, nan_policy=nan_policy)


def _weighted_quantile(row, q, row_weights):
    q_array = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_array)) or np.any((q_array < 0) | (q_array > 1)):
        raise ValueError('quantiles must be between zero and one')
    if row_weights is None:
        return np.quantile(row, q_array)
    order = np.argsort(row, kind='stable')
    values = row[order]
    ordered_weights = row_weights[order]
    if np.allclose(ordered_weights, ordered_weights[0]):
        return np.quantile(values, q_array)
    positions = np.cumsum(ordered_weights) / np.sum(ordered_weights)
    return np.interp(q_array, positions, values, left=values[0], right=values[-1])


def quantile(data, q, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return quantiles in ``[0, 1]``, including interpolated weighted quantiles."""
    _validate_ddof(ddof)
    q_array = np.asarray(q, dtype=float)
    rows, shape, _, _ = _rows(data, axis, weights, nan_policy, numeric=True)
    values = []
    for row, row_weights, propagated in rows:
        if propagated:
            values.append(np.full(q_array.shape, np.nan))
        else:
            values.append(_weighted_quantile(row, q_array, row_weights))
    result = np.asarray(values).reshape(shape + q_array.shape)
    if q_array.ndim and shape:
        result = np.moveaxis(result, tuple(range(len(shape), result.ndim)), tuple(range(q_array.ndim)))
    elif q_array.ndim and not shape:
        result = result.reshape(q_array.shape)
    return result.item() if result.ndim == 0 else result


def percentile(data, q, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return percentiles in ``[0, 100]``."""
    q_array = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_array)) or np.any((q_array < 0) | (q_array > 100)):
        raise ValueError('percentiles must be between zero and one hundred')
    return quantile(
        data, q_array / 100.0, axis=axis, weights=weights, ddof=ddof,
        nan_policy=nan_policy,
    )


def median(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the median (the weighted median when weights are supplied)."""
    return quantile(data, 0.5, axis=axis, weights=weights, ddof=ddof, nan_policy=nan_policy)


def quartiles(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return Q1, Q2, and Q3 along the first result dimension."""
    return quantile(
        data, [0.25, 0.5, 0.75], axis=axis, weights=weights, ddof=ddof,
        nan_policy=nan_policy,
    )


def iqr(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the interquartile range."""
    values = quartiles(data, axis=axis, weights=weights, ddof=ddof, nan_policy=nan_policy)
    return values[2] - values[0]


def mode(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the first encountered value among equally frequent modes."""
    _validate_ddof(ddof)
    rows, shape, _, _ = _rows(data, axis, weights, nan_policy, numeric=False)
    results = []
    for row, row_weights, propagated in rows:
        if propagated:
            results.append(np.nan)
            continue
        values = []
        totals = []
        for index, value in enumerate(row):
            match = next((i for i, known in enumerate(values) if value == known), None)
            amount = 1.0 if row_weights is None else float(row_weights[index])
            if match is None:
                values.append(value)
                totals.append(amount)
            else:
                totals[match] += amount
        results.append(values[int(np.argmax(totals))])
    return _finish(results, shape, dtype=object)


def minimum(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the minimum among positive-weight observations."""
    _validate_ddof(ddof)
    return _reduce(data, lambda row, _: float(np.min(row)), axis=axis, weights=weights,
                   nan_policy=nan_policy)


def maximum(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the maximum among positive-weight observations."""
    _validate_ddof(ddof)
    return _reduce(data, lambda row, _: float(np.max(row)), axis=axis, weights=weights,
                   nan_policy=nan_policy)


def data_range(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return ``maximum - minimum``."""
    return maximum(data, axis, weights, ddof, nan_policy) - minimum(
        data, axis, weights, ddof, nan_policy
    )


def _validate_ddof(ddof):
    if isinstance(ddof, bool) or not isinstance(ddof, Real) or not math.isfinite(float(ddof)):
        raise TypeError('ddof must be a finite real number')
    if ddof < 0:
        raise ValueError('ddof cannot be negative')
    return float(ddof)


def _variance(row, row_weights, ddof):
    if row_weights is None:
        denominator = row.size - ddof
        center = np.mean(row)
        numerator = np.sum((row - center) ** 2)
    else:
        denominator = np.sum(row_weights) - ddof
        center = np.average(row, weights=row_weights)
        numerator = np.sum(row_weights * (row - center) ** 2)
    if denominator <= 0:
        raise ValueError('ddof must be smaller than the observation count or total weight')
    return float(numerator / denominator)


def variance(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return variance; use ``ddof=0`` for population and ``ddof=1`` for sample."""
    ddof = _validate_ddof(ddof)
    return _reduce(data, lambda row, row_weights: _variance(row, row_weights, ddof),
                   axis=axis, weights=weights, nan_policy=nan_policy)


def standard_deviation(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the square root of :func:`variance`."""
    return np.sqrt(variance(data, axis, weights, ddof, nan_policy))


def population_variance(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    return variance(data, axis, weights, ddof, nan_policy)


def sample_variance(data, axis=None, weights=None, ddof=1, nan_policy='propagate'):
    return variance(data, axis, weights, ddof, nan_policy)


def population_standard_deviation(data, axis=None, weights=None, ddof=0,
                                  nan_policy='propagate'):
    return standard_deviation(data, axis, weights, ddof, nan_policy)


def sample_standard_deviation(data, axis=None, weights=None, ddof=1,
                              nan_policy='propagate'):
    return standard_deviation(data, axis, weights, ddof, nan_policy)


def mean_absolute_deviation(data, axis=None, weights=None, ddof=0,
                            nan_policy='propagate'):
    """Return mean absolute deviation from the arithmetic mean."""
    _validate_ddof(ddof)
    def reducer(row, row_weights):
        center = np.mean(row) if row_weights is None else np.average(row, weights=row_weights)
        deviations = np.abs(row - center)
        return float(np.mean(deviations) if row_weights is None else
                     np.average(deviations, weights=row_weights))
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def median_absolute_deviation(data, axis=None, weights=None, ddof=0,
                              nan_policy='propagate'):
    """Return median absolute deviation from the median."""
    _validate_ddof(ddof)
    def reducer(row, row_weights):
        center = _weighted_quantile(row, 0.5, row_weights)
        return float(_weighted_quantile(np.abs(row - center), 0.5, row_weights))
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def geometric_mean(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the geometric mean of strictly positive observations."""
    _validate_ddof(ddof)
    def reducer(row, row_weights):
        if np.any(row <= 0):
            raise ValueError('geometric mean requires positive observations')
        logs = np.log(row)
        return float(np.exp(np.mean(logs) if row_weights is None else
                            np.average(logs, weights=row_weights)))
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def harmonic_mean(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the harmonic mean of strictly positive observations."""
    _validate_ddof(ddof)
    def reducer(row, row_weights):
        if np.any(row <= 0):
            raise ValueError('harmonic mean requires positive observations')
        if row_weights is None:
            return float(row.size / np.sum(1.0 / row))
        return float(np.sum(row_weights) / np.sum(row_weights / row))
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def trimmed_mean(data, proportion_to_cut=0.1, axis=None, weights=None, ddof=0,
                 nan_policy='propagate'):
    """Return a mean after trimming each tail by the requested proportion."""
    _validate_ddof(ddof)
    if (isinstance(proportion_to_cut, bool) or not isinstance(proportion_to_cut, Real)
            or not 0 <= proportion_to_cut < 0.5):
        raise ValueError('proportion_to_cut must be in [0, 0.5)')
    def reducer(row, row_weights):
        order = np.argsort(row, kind='stable')
        row = row[order]
        if row_weights is None:
            cut = int(math.floor(row.size * proportion_to_cut))
            trimmed = row[cut:row.size - cut] if cut else row
            return float(np.mean(trimmed))
        remaining = row_weights[order].astype(float, copy=True)
        tail_weight = float(np.sum(remaining) * proportion_to_cut)
        left_to_remove = tail_weight
        for index in range(remaining.size):
            removed = builtins.min(remaining[index], left_to_remove)
            remaining[index] -= removed
            left_to_remove -= removed
            if left_to_remove <= 0:
                break
        right_to_remove = tail_weight
        for index in range(remaining.size - 1, -1, -1):
            removed = builtins.min(remaining[index], right_to_remove)
            remaining[index] -= removed
            right_to_remove -= removed
            if right_to_remove <= 0:
                break
        return float(np.average(row, weights=remaining))
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def _standardized_moment(data, order, axis, weights, ddof, nan_policy, excess=False):
    ddof = _validate_ddof(ddof)
    def reducer(row, row_weights):
        total = row.size if row_weights is None else np.sum(row_weights)
        denominator = total - ddof
        if denominator <= 0:
            raise ValueError('ddof must be smaller than the observation count or total weight')
        center = np.mean(row) if row_weights is None else np.average(row, weights=row_weights)
        deviations = row - center
        if row_weights is None:
            second = np.sum(deviations ** 2) / denominator
            moment = np.sum(deviations ** order) / denominator
        else:
            second = np.sum(row_weights * deviations ** 2) / denominator
            moment = np.sum(row_weights * deviations ** order) / denominator
        if second == 0:
            raise ValueError('standardized moments are undefined for constant data')
        value = moment / second ** (order / 2.0)
        return float(value - 3 if excess else value)
    return _reduce(data, reducer, axis=axis, weights=weights, nan_policy=nan_policy)


def skewness(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return the third standardized central moment."""
    return _standardized_moment(data, 3, axis, weights, ddof, nan_policy)


def kurtosis(data, axis=None, weights=None, ddof=0, nan_policy='propagate', excess=True):
    """Return excess kurtosis by default; pass ``excess=False`` for Pearson form."""
    return _standardized_moment(data, 4, axis, weights, ddof, nan_policy, excess)


def coefficient_of_variation(data, axis=None, weights=None, ddof=0,
                             nan_policy='propagate'):
    """Return standard deviation divided by the absolute mean."""
    average = mean(data, axis, weights, ddof, nan_policy)
    if np.any(np.asarray(average) == 0):
        raise ValueError('coefficient of variation is undefined when the mean is zero')
    return standard_deviation(data, axis, weights, ddof, nan_policy) / np.abs(average)


def _frequency_row(row, row_weights):
    values = []
    totals = []
    for index, value in enumerate(row):
        match = next((i for i, known in enumerate(values) if value == known), None)
        amount = 1 if row_weights is None else float(row_weights[index])
        if match is None:
            values.append(value)
            totals.append(amount)
        else:
            totals[match] += amount
    counts = np.asarray(totals, dtype=int if row_weights is None else float)
    return FrequencyTable(np.asarray(values), counts, counts / np.sum(counts))


def frequency_table(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return a :class:`FrequencyTable`, or an object array of tables with ``axis``."""
    _validate_ddof(ddof)
    rows, shape, _, _ = _rows(data, axis, weights, nan_policy, numeric=False)
    tables = [None if propagated else _frequency_row(row, row_weights)
              for row, row_weights, propagated in rows]
    return _finish(tables, shape, dtype=object)


def proportion_table(data, axis=None, weights=None, ddof=0, nan_policy='propagate'):
    """Return category-to-proportion mappings."""
    tables = frequency_table(data, axis, weights, ddof, nan_policy)
    def mapping(table):
        if table is None:
            return None
        return {
            value.item() if isinstance(value, np.generic) else value: float(proportion)
            for value, proportion in zip(table.values, table.proportions)
        }
    if isinstance(tables, np.ndarray):
        result = np.empty(tables.shape, dtype=object)
        for index in np.ndindex(tables.shape):
            result[index] = mapping(tables[index])
        return result
    return mapping(tables)


def five_number_summary(data, axis=None, weights=None, ddof=0,
                        nan_policy='propagate'):
    values = quantile(data, [0, 0.25, 0.5, 0.75, 1], axis, weights, ddof, nan_policy)
    return FiveNumberSummary(values[0], values[1], values[2], values[3], values[4])


def outlier_fences(data, axis=None, weights=None, ddof=0, nan_policy='propagate',
                   factor=1.5):
    """Return Tukey fences using ``Q1 - factor*IQR`` and ``Q3 + factor*IQR``."""
    if (isinstance(factor, bool) or not isinstance(factor, Real)
            or not math.isfinite(float(factor)) or factor < 0):
        raise ValueError('factor must be a non-negative real number')
    values = quartiles(data, axis, weights, ddof, nan_policy)
    spread = values[2] - values[0]
    return OutlierFences(values[0] - factor * spread, values[2] + factor * spread)


def detect_outliers(data, axis=None, weights=None, ddof=0, nan_policy='propagate',
                    factor=1.5):
    """Return a Boolean mask identifying values outside the Tukey fences."""
    array = _numeric_array(data)
    reduction_axis = _axis(axis, array.ndim)
    fences = outlier_fences(array, axis, weights, ddof, nan_policy, factor)
    if reduction_axis is None:
        return (array < fences.lower) | (array > fences.upper)
    lower = np.expand_dims(fences.lower, reduction_axis)
    upper = np.expand_dims(fences.upper, reduction_axis)
    return (array < lower) | (array > upper)


def _paired_rows(x, y, axis, weights, nan_policy):
    x_array = _numeric_array(x)
    y_array = _numeric_array(y)
    try:
        x_array, y_array = np.broadcast_arrays(x_array, y_array)
    except ValueError as exc:
        raise ValueError('x and y must have broadcastable shapes') from exc
    combined = np.stack((x_array, y_array), axis=-1)
    reduction_axis = _axis(axis, x_array.ndim)
    if reduction_axis is None:
        combined = combined.reshape(-1, 2)
        combined_axis = 0
    else:
        combined_axis = reduction_axis
    # Prepare each variable with a shared missing mask in the covariance reducer.
    weight_array = _broadcast_weights(weights, x_array.shape, reduction_axis)
    moved = np.moveaxis(combined, combined_axis, -2)
    output_shape = moved.shape[:-2]
    matrix = moved.reshape(-1, moved.shape[-2], 2)
    if weight_array is None:
        weight_matrix = None
    else:
        if reduction_axis is None:
            weight_matrix = weight_array.reshape(1, -1)
        else:
            moved_weights = np.moveaxis(weight_array, reduction_axis, -1)
            weight_matrix = moved_weights.reshape(-1, moved_weights.shape[-1])
    policy = _policy(nan_policy)
    results = []
    for index, pair in enumerate(matrix):
        row_weights = None if weight_matrix is None else weight_matrix[index]
        missing = np.isnan(pair).any(axis=1)
        if row_weights is not None:
            missing = missing | np.isnan(row_weights)
            valid_weights = row_weights[~np.isnan(row_weights)]
            if np.any(~np.isfinite(valid_weights)):
                raise ValueError('weights must be finite')
            if np.any(valid_weights < 0):
                raise ValueError('weights cannot be negative')
        if policy == 'raise' and np.any(missing):
            raise ValueError('data contains missing values')
        if policy == 'propagate' and np.any(missing):
            results.append((None, None, True))
            continue
        if policy == 'omit':
            pair = pair[~missing]
            if row_weights is not None:
                row_weights = row_weights[~missing]
        if row_weights is not None:
            positive = row_weights > 0
            pair, row_weights = pair[positive], row_weights[positive]
        if pair.shape[0] == 0:
            raise ValueError('no paired observations remain')
        results.append((pair, row_weights, False))
    return results, output_shape


def _paired_statistic(x, y, axis, weights, ddof, nan_policy, correlation=False,
                      rank=False):
    ddof = _validate_ddof(ddof)
    rows, shape = _paired_rows(x, y, axis, weights, nan_policy)
    values = []
    for pair, row_weights, propagated in rows:
        if propagated:
            values.append(np.nan)
            continue
        left, right = pair[:, 0], pair[:, 1]
        if rank:
            left, right = _rank(left), _rank(right)
        total = left.size if row_weights is None else np.sum(row_weights)
        denominator = total - ddof
        if denominator <= 0:
            raise ValueError('ddof must be smaller than the paired observation count')
        left_mean = np.mean(left) if row_weights is None else np.average(left, weights=row_weights)
        right_mean = np.mean(right) if row_weights is None else np.average(right, weights=row_weights)
        if row_weights is None:
            numerator = np.sum((left - left_mean) * (right - right_mean))
        else:
            numerator = np.sum(row_weights * (left - left_mean) * (right - right_mean))
        covariance_value = numerator / denominator
        if not correlation:
            values.append(float(covariance_value))
            continue
        left_variance = _variance(left, row_weights, ddof)
        right_variance = _variance(right, row_weights, ddof)
        if left_variance == 0 or right_variance == 0:
            raise ValueError('correlation is undefined for constant data')
        values.append(float(covariance_value / math.sqrt(left_variance * right_variance)))
    return _finish(values, shape)


def covariance(x, y=None, axis=0, weights=None, ddof=1, nan_policy='propagate'):
    """Return pair covariance or a covariance matrix when ``y`` is omitted."""
    if y is not None:
        return _paired_statistic(x, y, axis, weights, ddof, nan_policy)
    array = _numeric_array(x)
    reduction_axis = _axis(axis, array.ndim)
    if reduction_axis is None or array.ndim == 1:
        return variance(array, axis=reduction_axis, weights=weights, ddof=ddof,
                        nan_policy=nan_policy)
    moved = np.moveaxis(array, reduction_axis, 0)
    variables = moved.reshape(moved.shape[0], -1)
    columns = variables.shape[1]
    result = np.empty((columns, columns), dtype=float)
    for i in range(columns):
        for j in range(columns):
            result[i, j] = _paired_statistic(
                variables[:, i], variables[:, j], 0, weights, ddof, nan_policy
            )
    return result


def pearson_correlation(x, y=None, axis=0, weights=None, ddof=1,
                        nan_policy='propagate'):
    """Return Pearson correlation or a correlation matrix when ``y`` is omitted."""
    if y is not None:
        return _paired_statistic(x, y, axis, weights, ddof, nan_policy, correlation=True)
    array = _numeric_array(x)
    reduction_axis = _axis(axis, array.ndim)
    if reduction_axis is None or array.ndim == 1:
        return _paired_statistic(array, array, reduction_axis, weights, ddof, nan_policy,
                                 correlation=True)
    moved = np.moveaxis(array, reduction_axis, 0).reshape(array.shape[reduction_axis], -1)
    columns = moved.shape[1]
    result = np.empty((columns, columns), dtype=float)
    for i in range(columns):
        for j in range(columns):
            result[i, j] = _paired_statistic(
                moved[:, i], moved[:, j], 0, weights, ddof, nan_policy,
                correlation=True,
            )
    return result


def _rank(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1
        start = stop
    return ranks


def spearman_correlation(x, y=None, axis=0, weights=None, ddof=1,
                         nan_policy='propagate'):
    """Return Spearman rank correlation, using average ranks for ties."""
    if y is not None:
        return _paired_statistic(x, y, axis, weights, ddof, nan_policy,
                                 correlation=True, rank=True)
    array = _numeric_array(x)
    reduction_axis = _axis(axis, array.ndim)
    if reduction_axis is None or array.ndim == 1:
        return _paired_statistic(array, array, reduction_axis, weights, ddof, nan_policy,
                                 correlation=True, rank=True)
    moved = np.moveaxis(array, reduction_axis, 0).reshape(array.shape[reduction_axis], -1)
    columns = moved.shape[1]
    result = np.empty((columns, columns), dtype=float)
    for i in range(columns):
        for j in range(columns):
            result[i, j] = _paired_statistic(
                moved[:, i], moved[:, j], 0, weights, ddof, nan_policy,
                correlation=True, rank=True,
            )
    return result


def contingency_table(x, y, axis=None, weights=None, ddof=0,
                      nan_policy='propagate'):
    """Return a categorical two-way table or an object array of tables."""
    _validate_ddof(ddof)
    x_array = _object_array(x)
    y_array = _object_array(y)
    try:
        x_array, y_array = np.broadcast_arrays(x_array, y_array)
    except ValueError as exc:
        raise ValueError('x and y must have broadcastable shapes') from exc
    reduction_axis = _axis(axis, x_array.ndim)
    combined = np.empty(x_array.shape, dtype=object)
    for index in np.ndindex(x_array.shape):
        combined[index] = (x_array[index], y_array[index])
    rows, shape, _, _ = _rows(combined, reduction_axis, weights, nan_policy, numeric=False)
    tables = []
    for row, row_weights, propagated in rows:
        if propagated:
            tables.append(None)
            continue
        row_values, column_values = [], []
        for left, right in row:
            if left not in row_values:
                row_values.append(left)
            if right not in column_values:
                column_values.append(right)
        counts = np.zeros((len(row_values), len(column_values)), dtype=(
            int if row_weights is None else float
        ))
        for index, (left, right) in enumerate(row):
            amount = 1 if row_weights is None else row_weights[index]
            counts[row_values.index(left), column_values.index(right)] += amount
        tables.append(ContingencyTable(
            np.asarray(row_values), np.asarray(column_values), counts, counts / np.sum(counts)
        ))
    return _finish(tables, shape, dtype=object)


def describe(data, axis=None, weights=None, ddof=1, nan_policy='propagate'):
    """Return a compact, unified numerical summary."""
    values = five_number_summary(data, axis, weights, ddof, nan_policy)
    return DescriptiveSummary(
        count(data, axis, weights, ddof, nan_policy),
        mean(data, axis, weights, ddof, nan_policy),
        standard_deviation(data, axis, weights, ddof, nan_policy),
        values.minimum,
        values.q1,
        values.median,
        values.q3,
        values.maximum,
        values.q3 - values.q1,
    )


# Familiar aliases without shadowing built-ins inside this module's implementation.
min = minimum
max = maximum
standard_dev = standard_deviation
std = standard_deviation
pearson = pearson_correlation
spearman = spearman_correlation


__all__ = [
    'FrequencyTable', 'FiveNumberSummary', 'OutlierFences', 'ContingencyTable',
    'DescriptiveSummary', 'count', 'mean', 'weighted_mean', 'median', 'mode',
    'minimum', 'maximum', 'min', 'max', 'data_range', 'quantile', 'percentile',
    'quartiles', 'iqr', 'variance', 'standard_deviation', 'standard_dev', 'std',
    'population_variance', 'sample_variance', 'population_standard_deviation',
    'sample_standard_deviation', 'mean_absolute_deviation',
    'median_absolute_deviation', 'geometric_mean', 'harmonic_mean', 'trimmed_mean',
    'skewness', 'kurtosis', 'coefficient_of_variation', 'frequency_table',
    'proportion_table', 'five_number_summary', 'outlier_fences', 'detect_outliers',
    'covariance', 'pearson_correlation', 'spearman_correlation', 'pearson',
    'spearman', 'contingency_table', 'describe',
]
