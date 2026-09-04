"""Reproducible NumPy comparisons; run with python -m scripts.benchmark_numeric.

Writes generated results to the ignored examples directory. No SciPy required.
Timings are local medians, not CI performance assertions.
"""
import argparse
import contextlib
import io
import json
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import kiwicalc as kw


def root_error(actual, expected):
    """Minimum worst relative error across all one-to-one root matchings."""
    actual = np.asarray(list(actual), dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return None
    distances = abs(actual[:, None] - expected) / np.maximum(1, abs(expected))
    thresholds = np.unique(distances)

    def matches(threshold):
        assigned = [-1] * len(expected)

        def augment(row, seen):
            for column in np.flatnonzero(distances[row] <= threshold):
                if column in seen:
                    continue
                seen.add(column)
                if assigned[column] == -1 or augment(assigned[column], seen):
                    assigned[column] = row
                    return True
            return False

        return all(augment(row, set()) for row in range(len(actual)))

    left, right = 0, len(thresholds) - 1
    while left < right:
        middle = (left + right) // 2
        if matches(thresholds[middle]):
            right = middle
        else:
            left = middle + 1
    return float(thresholds[left])


def cases():
    known = [
        ('quadratic_real', [-2, 3]),
        ('cubic_real', [-2, 1, 3]),
        ('quartic_mixed', [-2, 3, 1j, -1j]),
        ('degree6_real', [-3, -2, -1, 1, 2, 3]),
        ('degree8_mixed', [-3, -1, 1, 3, 1j, -1j, 2j, -2j]),
        ('repeated_real', [2, 2, 2, 2]),
        ('repeated_complex', [1j, 1j, -1j, -1j]),
        ('clustered_real', [1, 1.001, 1.002, 1.003]),
        ('zero_roots', [0, 0, -2, 1, 1j, -1j]),
        ('wide_roots', [1e-4, 1, 100, -10]),
    ]
    rng = np.random.default_rng(20260904)
    for degree in (4, 6, 10):
        for index in range(3):
            roots = list(rng.uniform(-3, 3, degree - 2)) + [1+2j, 1-2j]
            known.append((f'seeded_d{degree}_{index}', roots))
    for name, roots in known:
        yield name, np.real(np.poly(roots)), np.asarray(roots, dtype=complex)
    for scale in (1e-100, 1e100):
        yield f'scale_{scale:g}', np.array([1., -2., -5., 6.]) * scale, np.array([-2, 1, 3])


def measure(function, repeats):
    durations, messages = [], set()
    result, error = None, None
    for iteration in range(repeats + 1):
        with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('always')
            start = time.perf_counter()
            try:
                result = function()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            elapsed = time.perf_counter() - start
        messages.update(str(w.message) for w in captured)
        if iteration:
            durations.append(elapsed * 1000)
        if error:
            break
    return result, statistics.median(durations) if durations else elapsed * 1000, error, sorted(messages)


def run(repeats=3):
    rows = []
    for name, coefficients, expected in cases():
        derivative = np.polyder(coefficients)
        f = lambda x: np.polyval(coefficients, x)
        df = lambda x: np.polyval(derivative, x)
        methods = {
            'numpy.roots': lambda: np.roots(coefficients),
            'bairstow': lambda: kw.bairstow_method(coefficients, epsilon=1e-12, nmax=1000),
            'durand_kerner2': lambda: kw.durand_kerner2(coefficients, epsilon=1e-12, nmax=1000),
            'aberth': lambda: kw.aberth_method(f, df, coefficients, epsilon=1e-12, nmax=1000),
        }
        for method, function in methods.items():
            result, elapsed, error, messages = measure(function, repeats)
            row = dict(case=name, method=method, degree=len(expected), milliseconds=elapsed,
                       exception=error, warnings=messages, root_count=None,
                       root_error=None, backward_residual=None, reconstruction_error=None)
            if error is None:
                roots = np.asarray(list(result), dtype=complex)
                row['root_count'] = len(roots)
                row['root_error'] = root_error(roots, expected)
                if len(roots) and np.isfinite(roots).all():
                    normalized = coefficients / np.max(np.abs(coefficients))
                    with np.errstate(all='ignore'):
                        numerator = abs(np.polyval(normalized, roots))
                        denominator = np.polyval(abs(normalized), abs(roots))
                        residual = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
                        row['backward_residual'] = float(max(residual))
                        if len(roots) == len(expected):
                            reconstruction = np.poly(roots) * normalized[0]
                            row['reconstruction_error'] = float(max(abs(reconstruction-normalized)))
            rows.append(row)
    # NumPy has no general derivative, Simpson, or root-iteration equivalent.
    # Compare quadrature with sampled NumPy trapezoids and analytic integrals.
    calculus = []
    trap = getattr(np, 'trapezoid', None) or np.trapz
    for name, f, a, b, exact in [('constant', lambda x: np.ones_like(x), 0, 1, 1),
                                  ('square', lambda x: x*x, 0, 1, 1/3),
                                  ('sine', np.sin, 0, np.pi, 2)]:
        for intervals in (10, 1000):
            methods = {
                'numpy.trapezoid': lambda: trap(f(np.linspace(a, b, intervals+1)), np.linspace(a, b, intervals+1)),
                'trapz': lambda: kw.trapz(f, a, b, intervals),
                'reinman': lambda: kw.reinman(f, a, b, intervals+1),
                'simpson_odd_samples': lambda: kw.simpson(f, a, b, intervals+1),
                'simpson_even_samples': lambda: kw.simpson(f, a, b, intervals),
            }
            for method, function in methods.items():
                value, elapsed, error, messages = measure(function, repeats)
                calculus.append(dict(case=name, intervals=intervals, method=method, milliseconds=elapsed,
                                     absolute_error=None if error else abs(float(value)-exact), exception=error))
    return dict(environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                                 repeats=repeats, warmups=1, root_iteration_budget=1000),
                roots=rows, integration=calculus)


def report(data):
    lines = ['# Numeric benchmark', '', '## Methodology', '',
             f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}.",
             f"Median of {data['environment']['repeats']} timed runs after one warm-up. Exceptions terminate that case.",
             'Timing includes each public call but excludes input preparation. No timing assertions are used in tests.',
             'All root methods receive identical descending-power coefficients. Iterative methods use epsilon=1e-12 and nmax=1000;',
             'their tolerance and iteration meanings differ. NumPy uses its own eigenvalue-based implementation, not Bairstow.',
             'Root error is the optimal one-to-one worst relative error against the nominal generating roots.',
             'Repeated/clustered roots are ill-conditioned; coefficient rounding also changes their true roots.',
             'Backward residual alone cannot detect missing or duplicate roots; root counts and reconstruction are also recorded.',
             'Integration compares scalar KiwiCalc callbacks with vectorized NumPy evaluation, representing typical API usage.',
             'NumPy has no equivalent general callable differentiation, Simpson, optimization, or scalar root-solver API;',
             'these timings must not be interpreted as comparisons of every numeric algorithm.', '',
             '## Summary', '',
             '| Method | Full root count | Matched error <= 1e-6 | Median per-case time / NumPy |',
             '|---|---:|---:|---:|']
    baselines = {row['case']: row['milliseconds'] for row in data['roots'] if row['method'] == 'numpy.roots'}
    for method in ('numpy.roots', 'bairstow', 'durand_kerner2', 'aberth'):
        rows = [row for row in data['roots'] if row['method'] == method]
        complete = sum(row['root_count'] == row['degree'] for row in rows)
        accurate = sum(row['root_error'] is not None and row['root_error'] <= 1e-6 for row in rows)
        ratio = statistics.median(row['milliseconds'] / baselines[row['case']] for row in rows)
        lines.append(f'| {method} | {complete}/{len(rows)} | {accurate}/{len(rows)} | {ratio:.1f}x |')
    lines += ['', 'The accuracy threshold is a diagnostic, not a promised tolerance. Repeated-root cases are included.',
              'Durand-Kerner and Aberth currently return sets: missing multiplicity is an API limitation, not always convergence failure.',
              'The numeric stress tests cover Riemann interval weights, Simpson even-sample endpoints,',
              'and Aberth behavior with distinct clustered roots and tiny coefficient scales.', '',
              '## Polynomial roots', '', '| Case | Method | ms | Count | Matched root error | Backward residual | Exception |',
              '|---|---|---:|---:|---:|---:|---|']
    def number(value):
        return '—' if value is None else f'{value:.3g}'
    for row in data['roots']:
        lines.append(f"| {row['case']} | {row['method']} | {row['milliseconds']:.3f} | {row['root_count']} | {number(row['root_error'])} | {number(row['backward_residual'])} | {row['exception'] or ''} |")
    lines += ['', '## Integration', '', '| Case | Intervals | Method | ms | Absolute error |', '|---|---:|---|---:|---:|']
    for row in data['integration']:
        lines.append(f"| {row['case']} | {row['intervals']} | {row['method']} | {row['milliseconds']:.3f} | {number(row['absolute_error'])} |")
    lines += ['', 'Warnings, reconstruction errors, and full environment details are in the accompanying JSON report.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-prefix', default='numeric_benchmark')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not args.output_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.output_prefix):
        parser.error('--output-prefix must contain only letters, numbers, underscores, or hyphens')
    data = run(args.repeats)
    output = Path(__file__).resolve().parents[1] / 'examples'
    output.mkdir(exist_ok=True)
    (output / (args.output_prefix + '.json')).write_text(json.dumps(data, indent=2), encoding='utf-8')
    (output / (args.output_prefix + '.md')).write_text(report(data), encoding='utf-8')
    print(f'Results: {output / (args.output_prefix + ".md")}')
